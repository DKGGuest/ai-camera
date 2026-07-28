import csv
import json
import os
import threading
import time
from collections import defaultdict

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
import cv2
import numpy as np

from src.monitors.box_bcounter import BoxCounter
import src.core.models as models

import config
import database
import models_loader
import math

MODES = ("access", "vehicle", "adaptive", "people", "worker", "queue", "box")
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO ids)
PERSON_CLASS = 0


def load_settings():
    if os.path.exists(config.SETTINGS_PATH):
        with open(config.SETTINGS_PATH) as f:
            return json.load(f)
    return {"camera_url": config.DEFAULT_CAMERA_URL, "mode": "access"}


def save_settings(settings):
    with open(config.SETTINGS_PATH, "w") as f:
        json.dump(settings, f)


def nms_boxes(boxes, scores, iou_threshold=0.4):
    """
    Simple 1D Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes.
    """
    if not boxes:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep


class CameraWorker:
    """
    Single background thread that:
      - pulls frames from the ESP32-S3-CAM MJPEG stream
      - runs whichever AI model is currently selected
      - draws overlays and keeps the latest annotated frame as JPEG bytes
      - writes detection events into the SQLite database
    """

    def __init__(self):
        settings = load_settings()
        self.camera_url = settings.get("camera_url", config.DEFAULT_CAMERA_URL)
        self._requested_camera_url = None
        self.mode = settings.get("mode", "access")
        self.queue_roi = settings.get("queue_roi", None)

        self._frame_lock = threading.Lock()
        self._latest_jpeg = None
        self._status_lock = threading.Lock()
        self._connected = False
        self._last_error = ""

        self._cap = None
        self._stop = False

        # Models (lazy-loaded)
        self._yolo = None
        self._face_detector = None
        self._face_recognizer = None
        self._known_embeddings = {}

        # Access-control state
        self._last_access_log = {}   # name -> last log timestamp (cooldown)

        # Vehicle state
        self._track_history = defaultdict(list)
        self._counted_ids = set()
        self._upward_count = 0
        self._lane_changers = set()

        # Adaptive-stream state
        self._active_mode = False
        self._last_human_seen = 0
        self._adaptive_last_change = time.time()
        self._COOLDOWN = 5.0

        # People-counter state
        self._people_tracks = {}
        self._people_next_id = 0
        self._people_entry_count = 0
        self._people_exit_count = 0
        self._last_people_photo = 0.0

        # Worker-tracker state
        self._gender_net = None
        self._face_cascade = None
        self._profile_cascade = None
        self._worker_states = {}        # track_id -> {status, work_s, idle_s, phone_s, talk_s, streak, pending_status}
        self._worker_frame_count = 0
        self._worker_last_laptop_boxes = []
        self._worker_last_phone_boxes = []
        self._worker_last_tick = None
        self._last_worker_photo = 0.0
        self._worker_pair_close_since = {}   # frozenset({id1, id2}) -> timestamp first seen close & not working
        self._worker_csv_path = os.path.join(config.BASE_DIR, "worker_logs.csv")

        # Queue-monitor state
        self._queue_last_alarm = 0.0
        self._last_queue_photo = 0.0

        # Queue setup
        self._queue_roi = []
        
        # Box Counter setup
        self._box_line = []

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._raw_frame_lock = threading.Lock()
        self._raw_frame = None

    # ------------------------------------------------------------------ #
    # Public control API (called from Flask routes)
    # ------------------------------------------------------------------ #
    def start(self):
        self._ensure_models_loaded()
        self._capture_thread.start()
        self._thread.start()

    def reload_settings(self):
        settings = load_settings()
        self.camera_url = settings.get("camera_url", config.DEFAULT_CAMERA_URL)
        self.mode = settings.get("mode", "access")
        self.queue_roi = settings.get("queue_roi", None)

    def load_settings_dict(self):
        return load_settings()

    def save_settings_dict(self, settings):
        save_settings(settings)

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        s = load_settings()
        s["mode"] = mode
        save_settings(s)
        database.log_event("system", "mode_change", f"Switched to '{mode}' model", "info")

    def set_camera_url(self, url):
        self._requested_camera_url = url
        s = load_settings()
        s["camera_url"] = url
        save_settings(s)

    def set_queue_roi(self, points):
        with self._status_lock:
            self._queue_roi = points
            
    def set_box_line(self, points):
        with self._status_lock:
            self._box_line = points

    def reload_known_faces(self):
        self._ensure_models_loaded()
        self._known_embeddings = models_loader.load_known_faces(
            config.KNOWN_FACES_DIR, self._face_detector, self._face_recognizer
        )

    def get_status(self):
        with self._status_lock:
            return {"connected": self._connected, "error": self._last_error, "mode": self.mode,
                    "camera_url": self.camera_url}

    def get_latest_frame_bgr(self):
        """Returns the last raw BGR frame (used by the 'capture' / enroll feature)."""
        with self._frame_lock:
            return None if self._last_raw_frame is None else self._last_raw_frame.copy()

    def mjpeg_generator(self):
        boundary = b"--frame"
        last_jpeg = None
        while True:
            with self._frame_lock:
                jpeg = self._latest_jpeg
            if jpeg is not None and jpeg != last_jpeg:
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                last_jpeg = jpeg
            time.sleep(0.02)  # Check quickly, but only send new frames

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    _last_raw_frame = None

    def _ensure_models_loaded(self):
        if self._yolo is None:
            self._yolo = models_loader.load_yolo_model()
        self._box_counter = BoxCounter()
        self._face_detector, self._face_recognizer = models_loader.load_face_models()
        self._known_embeddings = models_loader.load_known_faces(
            config.KNOWN_FACES_DIR, self._face_detector, self._face_recognizer
        )

    def _open_capture(self):
        src = int(self.camera_url) if str(self.camera_url).isdigit() else self.camera_url
        # Always use FFMPEG to enforce nobuffer options and avoid MSMF lag on Windows
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self):
        """Runs in its own thread, doing nothing but keeping self._raw_frame as fresh
        as possible. This is what actually fixes multi-second RTSP lag: if this were
        combined with the (slower) AI processing loop, frames would pile up in the
        camera's network buffer every time a model took longer than one frame interval
        to run. By reading continuously here, no backlog can ever accumulate — the
        processing loop below just always grabs whatever is newest, however far behind
        it currently is."""
        while not self._stop:
            # Safely handle camera URL switch in the capture thread
            if self._requested_camera_url is not None:
                self.camera_url = self._requested_camera_url
                self._requested_camera_url = None
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                with self._raw_frame_lock:
                    self._raw_frame = None

            if self._cap is None:
                self._cap = self._open_capture()
                if not self._cap.isOpened():
                    with self._status_lock:
                        self._connected = False
                        self._last_error = f"Could not open camera stream: {self.camera_url}"
                    time.sleep(2.0)
                    self._cap = None
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                with self._status_lock:
                    self._connected = False
                    self._last_error = "Lost connection to camera stream, retrying..."
                self._cap.release()
                self._cap = None
                time.sleep(1.0)
                continue

            with self._status_lock:
                self._connected = True
                self._last_error = ""

            with self._raw_frame_lock:
                self._raw_frame = frame
            self._last_raw_frame = frame

    def _loop(self):
        while not self._stop:
            with self._raw_frame_lock:
                frame = None if self._raw_frame is None else self._raw_frame.copy()

            if frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting to camera...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                ok, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok:
                    with self._frame_lock:
                        self._latest_jpeg = buf.tobytes()
                time.sleep(0.1)
                continue

            # Downscale heavy 4K frames before AI processing to keep pace with 25fps input
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))

            try:
                if self.mode == "access":
                    out = self._process_access(frame)
                elif self.mode == "vehicle":
                    out = self._process_vehicle(frame)
                elif self.mode == "people":
                    out = self._process_people(frame)
                elif self.mode == "worker":
                    out = self._process_worker(frame)
                elif self.mode == "queue":
                    out = self._process_queue(frame)
                elif self.mode == "box":
                    out = self._process_box(frame)
                else:
                    out = self._process_adaptive(frame)
            except Exception as e:
                out = frame
                print(f"[{self.mode}] Model error: {e}")
                cv2.putText(out, f"Model error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._frame_lock:
                    self._latest_jpeg = buf.tobytes()

            # Yield CPU briefly so the capture thread can eagerly empty the network buffer
            time.sleep(0.01)

    # ------------------------------------------------------------------ #
    # Model 1: Access Control (entry/exit, granted/denied)
    # ------------------------------------------------------------------ #
    def _process_access(self, frame):
        (h, w) = frame.shape[:2]
        self._face_detector.setInputSize((w, h))
        _, faces = self._face_detector.detect(frame)

        largest_area, largest_face, largest_box = 0, None, None
        if faces is not None:
            for face in faces:
                box = face[0:4].astype(int)
                x, y, fw, fh = box
                area = fw * fh
                if area > largest_area and fw > 40 and fh > 40:
                    largest_area, largest_face, largest_box = area, face, box

        status_text, status_color = "SCANNING...", (150, 150, 150)

        if largest_face is not None:
            x, y, fw, fh = largest_box
            try:
                aligned = self._face_recognizer.alignCrop(frame, largest_face)
                feature = self._face_recognizer.feature(aligned)

                best_match, best_score = "Unknown", -1.0
                for name, emb in self._known_embeddings.items():
                    score = models_loader.recognizer_match(feature[0], emb)
                    if score > best_score:
                        best_score = score
                        if score > config.FACE_SIMILARITY_THRESHOLD:
                            best_match = name

                now = time.time()
                if best_match != "Unknown":
                    status_text = f"ACCESS GRANTED: {best_match.upper()}"
                    status_color, box_color = (0, 200, 0), (0, 255, 0)
                    last = self._last_access_log.get(best_match, 0)
                    if now - last > 5.0:
                        database.log_event("access", "entry", best_match, "granted")
                        photo_path = f"static/events/{int(now)}_{best_match}.jpg"
                        cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                        database.log_access(best_match, "granted", photo_path)
                        self._last_access_log[best_match] = now
                else:
                    status_text = "ACCESS DENIED"
                    status_color, box_color = (0, 0, 200), (0, 0, 255)
                    last = self._last_access_log.get("Unknown", 0)
                    if now - last > 5.0:
                        database.log_event("access", "entry", "Unknown person", "denied")
                        photo_path = f"static/events/{int(now)}_unknown.jpg"
                        cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                        database.log_access("Unknown person", "denied", photo_path)
                        self._last_access_log["Unknown"] = now

                cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 3)
            except Exception:
                pass

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        return frame

    # ------------------------------------------------------------------ #
    # Model 2: Vehicle Detection / Counting
    # ------------------------------------------------------------------ #
    def _process_vehicle(self, frame):
        height, width = frame.shape[:2]
        results = self._yolo.track(frame, persist=True, classes=VEHICLE_CLASSES, verbose=False)

        counting_line_y = int(height * 0.5)
        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                hist = self._track_history[track_id]
                hist.append((cx, cy))
                if len(hist) > 10:
                    hist.pop(0)

                if len(hist) >= 2:
                    prev_cy = hist[-2][1]
                    if prev_cy > counting_line_y and cy <= counting_line_y and track_id not in self._counted_ids:
                        self._upward_count += 1
                        self._counted_ids.add(track_id)
                        database.log_event("vehicle", "vehicle_crossed", f"Track ID {track_id}", "info")

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.rectangle(frame, (10, 10), (300, 55), (0, 0, 0), -1)
        cv2.putText(frame, f"Vehicle Count: {self._upward_count}", (18, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame

    # ------------------------------------------------------------------ #
    # Model 3: Adaptive Bitrate (drops resolution/bitrate when idle)
    # ------------------------------------------------------------------ #
    def _process_adaptive(self, frame):
        height, width = frame.shape[:2]
        low_w, low_h = 320, int(320 * (height / width)) if width else 240

        detect_frame = cv2.resize(frame, (low_w, low_h))
        results = self._yolo(detect_frame, classes=[PERSON_CLASS], verbose=False)

        human_detected = False
        if results[0].boxes is not None:
            for conf in results[0].boxes.conf.cpu():
                if conf > 0.4:
                    human_detected = True
                    break

        now = time.time()
        was_active = self._active_mode
        if human_detected:
            self._last_human_seen = now
            self._active_mode = True
        elif now - self._last_human_seen > self._COOLDOWN:
            self._active_mode = False

        if was_active != self._active_mode:
            state = "high" if self._active_mode else "low"
            duration = now - self._adaptive_last_change
            self._adaptive_last_change = now
            
            database.log_event("adaptive", "bitrate_change",
                                "Switched to HIGH bitrate" if self._active_mode else "Switched to LOW bitrate",
                                "active" if self._active_mode else "idle")
            photo_path = f"static/events/{int(now)}_adaptive_{state}.jpg"
            cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
            database.log_adaptive(state, duration, photo_path)

        out = frame.copy()
        if not self._active_mode:
            pixelated = cv2.resize(out, (low_w, low_h), interpolation=cv2.INTER_NEAREST)
            out = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)
            status_text, status_color, bitrate_text = "IDLE - LOW BITRATE", (0, 0, 255), "~150 kbps"
        else:
            status_text, status_color, bitrate_text = "ACTIVE - HIGH BITRATE", (0, 255, 0), "~6000 kbps"
            full = self._yolo(out, classes=[PERSON_CLASS], verbose=False)
            if full[0].boxes is not None:
                for box, conf in zip(full[0].boxes.xyxy.cpu(), full[0].boxes.conf.cpu()):
                    if conf > 0.4:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.rectangle(out, (10, 10), (430, 90), (0, 0, 0), -1)
        cv2.putText(out, status_text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        cv2.putText(out, bitrate_text, (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    # ------------------------------------------------------------------ #
    # Model 4: People Counter (entry/exit line crossing)
    # ------------------------------------------------------------------ #
    def _process_people(self, frame):
        height, width = frame.shape[:2]
        counting_line_y = height // 3
        buffer_px = 40
        upper_y, lower_y = counting_line_y - buffer_px, counting_line_y + buffer_px

        results = self._yolo(frame, classes=[PERSON_CLASS], verbose=False)
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu():
                x1, y1, x2, y2 = map(int, box)
                detections.append(((x1 + x2) // 2, (y1 + y2) // 2))

        # Match detections to existing tracks by nearest centroid
        used_ids = set()
        for cx, cy in detections:
            best_id, best_dist = None, float("inf")
            for tid, data in self._people_tracks.items():
                if tid in used_ids:
                    continue
                dist = math.hypot(cx - data["cx"], cy - data["cy"])
                if dist < 80 and dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is None:
                best_id = self._people_next_id
                self._people_next_id += 1
                zone = "above" if cy < upper_y else "below" if cy > lower_y else "buffer"
                self._people_tracks[best_id] = {"cx": cx, "cy": cy, "state": zone, "lost": 0}
            else:
                data = self._people_tracks[best_id]
                prev_zone = data["state"]
                zone = "above" if cy < upper_y else "below" if cy > lower_y else "buffer"
                if prev_zone in ("above",) and zone == "below":
                    self._people_entry_count += 1
                    database.log_event("people", "entry", f"Track {best_id}", "info")
                    photo_path = f"static/events/{int(time.time())}_people_entry.jpg"
                    cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                    database.log_people("entry", self._people_entry_count, self._people_exit_count, photo_path)
                elif prev_zone in ("below",) and zone == "above":
                    self._people_exit_count += 1
                    database.log_event("people", "exit", f"Track {best_id}", "info")
                    photo_path = f"static/events/{int(time.time())}_people_exit.jpg"
                    cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                    database.log_people("exit", self._people_entry_count, self._people_exit_count, photo_path)
                data.update({"cx": cx, "cy": cy, "state": zone, "lost": 0})

            used_ids.add(best_id)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        for tid in list(self._people_tracks.keys()):
            if tid not in used_ids:
                self._people_tracks[tid]["lost"] += 1
                if self._people_tracks[tid]["lost"] > 15:
                    del self._people_tracks[tid]

        now = time.time()
        if now - self._last_people_photo > 5.0:
            self._last_people_photo = now
            photo_path = f"static/events/{int(now)}_people_periodic.jpg"
            cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
            database.log_people("periodic", self._people_entry_count, self._people_exit_count, photo_path)

        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (255, 0, 0), 2)
        cv2.rectangle(frame, (10, 10), (250, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Entries: {self._people_entry_count}", (18, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {self._people_exit_count}", (18, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    # ------------------------------------------------------------------ #
    # Model 5: Worker Activity Tracker
    # Stable IDs (ByteTrack) + fast face-facing check + phone-near-face +
    # laptop association + talking detection + dashboard + CSV event log.
    # ------------------------------------------------------------------ #
    def _log_worker_event(self, tid, s, new_status, now):
        try:
            file_exists = os.path.exists(self._worker_csv_path)
            with open(self._worker_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "worker_id", "status", "work_seconds",
                                      "idle_seconds", "phone_seconds", "talk_seconds",
                                      "attention_pct", "productivity_pct"])
                total_s = s.get("work_s", 0) + s.get("idle_s", 0) + s.get("phone_s", 0) + s.get("talk_s", 0)
                pct = (s.get("work_s", 0) / total_s * 100) if total_s else 0.0
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                    tid, new_status,
                    f"{s.get('work_s', 0):.1f}", f"{s.get('idle_s', 0):.1f}",
                    f"{s.get('phone_s', 0):.1f}", f"{s.get('talk_s', 0):.1f}",
                    f"{pct:.1f}", f"{pct:.1f}",
                ])
            # Also log to new database
            worker_name = f"Worker {tid}"
            photo_path = f"static/events/{int(now)}_worker_{tid}.jpg"
            # Note: frame is not easily available in _log_worker_event without passing it down.
            # We will use the worker_last_frame or just write a blank image if we can't access it easily.
            # Wait, `self._raw_frame` is available.
            with self._raw_frame_lock:
                current_frame = self._raw_frame.copy() if self._raw_frame is not None else None
            if current_frame is not None:
                cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), current_frame)
            else:
                photo_path = ""
            database.log_worker(worker_name, s.get("working_s", 0), s.get("not_working_s", 0), photo_path)
        except Exception as e:
            print(f"[worker] event log error: {e}")

    def _process_worker(self, frame):
        # Ensure YuNet face detector is loaded (for OpenCV 5+ compatibility)
        if not hasattr(self, '_worker_face_det') or self._worker_face_det is None:
            from src.core.models import load_worker_face_detector
            self._worker_face_det = load_worker_face_detector()

        height, width = frame.shape[:2]
        self._worker_frame_count += 1

        now = time.time()
        dt = now - self._worker_last_tick if self._worker_last_tick is not None else 0.0
        self._worker_last_tick = now

        # Detection + stable-ID tracking (Centroid tracking) every 2nd frame.
        if self._worker_frame_count % 2 == 1:
            # Run YOLO with a lower confidence threshold (0.05) to ensure small/low-confidence cell phones are detected.
            results = self._yolo(frame, classes=[0, 62, 63, 64, 66, 67], conf=0.05, verbose=False)
            r = results[0]
            person_candidates, person_scores = [], []
            laptop_candidates, laptop_scores, laptop_names = [], [], []
            phone_candidates, phone_scores = [], []
            if r.boxes is not None:
                for box, conf, cls in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu()):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = self._yolo.names[int(cls)]
                    c = float(conf)
                    
                    if cls_name == "person" and c >= 0.20:
                        person_candidates.append([x1, y1, x2, y2])
                        person_scores.append(c)
                    elif cls_name in ("laptop", "tv", "mouse", "keyboard") and c >= 0.20:
                        laptop_candidates.append([x1, y1, x2, y2])
                        laptop_scores.append(c)
                        laptop_names.append(cls_name)
                    # Class ID 67 represents 'cell phone' in the YOLO COCO dataset
                    elif cls_name == "cell phone" and c >= 0.05:
                        phone_candidates.append([x1, y1, x2, y2])
                        phone_scores.append(c)

            # Apply Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes
            keep_persons = nms_boxes(person_candidates, person_scores, iou_threshold=0.4)
            person_raw_dets = [(person_candidates[i], person_scores[i]) for i in keep_persons]
            
            # Combine TV and Laptop NMS to avoid double boxes on the same device
            keep_laptops = nms_boxes(laptop_candidates, laptop_scores, iou_threshold=0.3)
            laptop_boxes = [(laptop_candidates[i], laptop_scores[i], laptop_names[i]) for i in keep_laptops]
            
            keep_phones = nms_boxes(phone_candidates, phone_scores, iou_threshold=0.3)
            phone_boxes = [(phone_candidates[i], phone_scores[i]) for i in keep_phones]

            # Track persons using centroid tracking
            person_dets = []
            next_id = 0
            if self._worker_states:
                next_id = max(self._worker_states.keys()) + 1

            used_tids = set()
            for p_box, p_conf in person_raw_dets:
                x1, y1, x2, y2 = p_box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                best_tid = None
                min_dist = 200 # max distance threshold
                for tid, st in self._worker_states.items():
                    if tid in used_tids or st.get("lost_frames", 0) > 0:
                        continue
                    old_box = st["box"]
                    old_cx = (old_box[0] + old_box[2]) / 2
                    old_cy = (old_box[1] + old_box[3]) / 2
                    dist = math.hypot(cx - old_cx, cy - old_cy)
                    if dist < min_dist:
                        min_dist = dist
                        best_tid = tid
                        
                if best_tid is None:
                    best_tid = next_id
                    next_id += 1
                else:
                    used_tids.add(best_tid)
                
                person_dets.append((best_tid, p_box, p_conf))

            seen_ids = set()
            for tid, p_box, p_conf in person_dets:
                seen_ids.add(tid)
                x1, y1, x2, y2 = p_box
                px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
                px2, py2 = min(width, x2 + 20), min(height, y2 + 20)
                roi = frame[py1:py2, px1:px2]

                # --- YuNet Face Detection + Head Pose Estimation ---
                face_box_abs = None
                looking_at_screen = False
                if roi.size > 0:
                    try:
                        rh, rw = roi.shape[:2]
                        if rw > 30 and rh > 30:
                            self._worker_face_det.setInputSize((rw, rh))
                            _, detected_faces = self._worker_face_det.detect(roi)
                            if detected_faces is not None and len(detected_faces) > 0:
                                # Pick the largest face in ROI
                                best_face = max(detected_faces, key=lambda f: f[2] * f[3])
                                fx, fy, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                                face_box_abs = [px1 + fx, py1 + fy, px1 + fx + fw, py1 + fy + fh]

                                # Extract 5 landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
                                lm = best_face[4:14].reshape(5, 2)
                                right_eye, left_eye = lm[0], lm[1]
                                nose = lm[2]
                                right_mouth, left_mouth = lm[3], lm[4]

                                eye_dist = float(np.linalg.norm(left_eye - right_eye))
                                if eye_dist > 1.0:
                                    eye_cx = (right_eye[0] + left_eye[0]) / 2.0
                                    eye_cy = (right_eye[1] + left_eye[1]) / 2.0
                                    mouth_cy = (right_mouth[1] + left_mouth[1]) / 2.0

                                    # Yaw: nose offset from eye center (looking sideways)
                                    yaw_ratio = (nose[0] - eye_cx) / eye_dist
                                    # Pitch: vertical face proportion (eyes to mouth distance)
                                    # Low pitch = head tilted down (sleeping/slumped)
                                    pitch_ratio = (mouth_cy - eye_cy) / eye_dist

                                    # Face vertical position in person bounding box
                                    face_center_y_abs = py1 + fy + fh / 2.0
                                    person_h = py2 - py1
                                    face_rel_y = (face_center_y_abs - py1) / person_h if person_h > 0 else 0.5

                                    # WORKING = face upright, looking forward, not slumped
                                    looking_at_screen = (
                                        abs(yaw_ratio) < 0.35 and   # not looking too far left/right
                                        pitch_ratio > 0.25 and      # not head down (sleeping/frustrated)
                                        face_rel_y < 0.55           # not slumped over desk
                                    )
                    except Exception:
                        pass

                # Phone-near-face (falls back to phone-near-body if no face was found)
                phone_near = False
                for pb, _pconf in phone_boxes:
                    ref_box = face_box_abs if face_box_abs is not None else p_box
                    rx1, ry1, rx2, ry2 = ref_box
                    margin = 40 if face_box_abs is not None else 15
                    if not (pb[2] < rx1 - margin or pb[0] > rx2 + margin or
                            pb[3] < ry1 - margin or pb[1] > ry2 + margin):
                        phone_near = True
                        break

                # Nearest-laptop association
                has_laptop = False
                if laptop_boxes:
                    pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
                    for lb, _lconf, _lname in laptop_boxes:
                        lcx, lcy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
                        if math.hypot(pcx - lcx, pcy - lcy) < (x2 - x1) * 3.5:
                            has_laptop = True
                            break

                # WORKING = near laptop + looking at screen + no phone
                raw_working = has_laptop and looking_at_screen and not phone_near

                st = self._worker_states.get(tid)
                if st is None:
                    self._worker_states[tid] = {
                        "work_s": 0.0, "idle_s": 0.0, "phone_s": 0.0, "talk_s": 0.0,
                        "working_s": 0.0, "not_working_s": 0.0,
                        "status": "Working" if raw_working else ("Using Phone" if phone_near else "Idle"),
                        "streak": 0, "gender": "Unknown", "box": p_box, "conf": p_conf,
                        "lost_frames": 0, "raw_working": raw_working, "phone_near": phone_near,
                        "has_laptop": has_laptop,
                    }
                else:
                    st["box"], st["conf"], st["lost_frames"] = p_box, p_conf, 0
                    st["raw_working"], st["phone_near"], st["has_laptop"] = raw_working, phone_near, has_laptop

            # Talking: two currently-visible, not-working people standing close together for 5s+
            visible = [(tid, s) for tid, s in self._worker_states.items() if s.get("lost_frames", 0) == 0]
            talking_ids = set()
            for i in range(len(visible)):
                tid_a, sa = visible[i]
                if sa["raw_working"]:
                    continue
                ax1, ay1, ax2, ay2 = sa["box"]
                acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                aw = ax2 - ax1
                for j in range(i + 1, len(visible)):
                    tid_b, sb = visible[j]
                    if sb["raw_working"]:
                        continue
                    bx1, by1, bx2, by2 = sb["box"]
                    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    bw = bx2 - bx1
                    pair = frozenset((tid_a, tid_b))
                    if math.hypot(acx - bcx, acy - bcy) < (aw + bw) * 1.3:
                        first_seen = self._worker_pair_close_since.setdefault(pair, now)
                        if now - first_seen > 5.0:
                            talking_ids.add(tid_a)
                            talking_ids.add(tid_b)
                    else:
                        self._worker_pair_close_since.pop(pair, None)

            # Resolve final status with a short 2-cycle hysteresis to avoid single-frame flicker
            for tid, s in self._worker_states.items():
                if s.get("lost_frames", 0) > 0:
                    continue
                if s["phone_near"]:
                    target = "Using Phone"
                elif tid in talking_ids:
                    target = "Talking"
                elif s["raw_working"]:
                    target = "Working"
                elif not s["has_laptop"]:
                    target = "No Laptop"
                else:
                    target = "Looking Away"

                if target == s.get("pending", target):
                    s["streak"] = s.get("streak", 0) + 1
                else:
                    s["pending"], s["streak"] = target, 1

                if s["streak"] >= 2 and s.get("status") != target:
                    self._log_worker_event(tid, s, target, now)
                    s["status"] = target

            for tid in list(self._worker_states.keys()):
                s = self._worker_states[tid]
                if tid not in seen_ids:
                    s["lost_frames"] = s.get("lost_frames", 0) + 1
                    if s["lost_frames"] > 15:
                        del self._worker_states[tid]

            self._worker_last_laptop_boxes = laptop_boxes
            self._worker_last_phone_boxes = phone_boxes

        # Timer accumulation every frame (real wall-clock seconds) so on-screen timers tick smoothly
        for tid, s in self._worker_states.items():
            if s.get("lost_frames", 0) > 0:
                continue
            status = s.get("status", "Idle")
            # Fine-grained timers (kept for CSV logging)
            if status == "Working":
                s["work_s"] = s.get("work_s", 0.0) + dt
            elif status == "Using Phone":
                s["phone_s"] = s.get("phone_s", 0.0) + dt
            elif status == "Talking":
                s["talk_s"] = s.get("talk_s", 0.0) + dt
            else:
                s["idle_s"] = s.get("idle_s", 0.0) + dt
            # Binary working / not-working timers (mutually exclusive)
            if status == "Working":
                s["working_s"] = s.get("working_s", 0.0) + dt
            else:
                s["not_working_s"] = s.get("not_working_s", 0.0) + dt

        # --- Drawing ---
        def _fmt_time(seconds):
            h_ = int(seconds // 3600)
            m_ = int((seconds % 3600) // 60)
            s_ = int(seconds % 60)
            return f"{h_:02d}:{m_:02d}:{s_:02d}"

        for lb, lconf, lname in self._worker_last_laptop_boxes:
            x1, y1, x2, y2 = lb
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 0), 1)
            cv2.putText(frame, f"{lname}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1)
        for pb, pconf in self._worker_last_phone_boxes:
            x1, y1, x2, y2 = pb
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 105, 255), 1)
            cv2.putText(frame, "phone", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 105, 255), 1)

        COLOR_GREEN = (0, 255, 0)   # BGR - Working
        COLOR_RED = (0, 0, 255)     # BGR - Not Working

        working_count = 0
        not_working_count = 0

        for tid, s in self._worker_states.items():
            if s.get("lost_frames", 0) > 0:
                continue
            x1, y1, x2, y2 = s["box"]
            status = s.get("status", "Idle")
            is_working = (status == "Working")
            color = COLOR_GREEN if is_working else COLOR_RED
            status_label = "WORKING" if is_working else "NOT WORKING"

            if is_working:
                working_count += 1
            else:
                not_working_count += 1

            # Bounding box with padding
            px1, py1 = max(0, x1 - 12), max(0, y1 - 12)
            px2, py2 = min(width, x2 + 12), min(height, y2 + 12)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

            # ---- Write output / display ----
            
            if now - self._last_worker_photo > 5.0:
                self._last_worker_photo = now
                photo_path = f"static/events/{int(now)}_worker_periodic.jpg"
                cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                # Log an overall summary line if we want, or just a dummy entry
                database.log_worker("Periodic Snapshot", 0.0, 0.0, photo_path)

            w_time = _fmt_time(s.get("working_s", 0.0))
            nw_time = _fmt_time(s.get("not_working_s", 0.0))

            w_label = f"W: {w_time}"
            nw_label = f"NW: {nw_time}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (w_tw, w_th), _ = cv2.getTextSize(w_label, font, font_scale, thickness)
            (nw_tw, nw_th), _ = cv2.getTextSize(nw_label, font, font_scale, thickness)

            label_h = max(w_th, nw_th) + 10
            label_y_top = max(0, py1 - label_h - 2)

            # Working timer - top-left (green background)
            w_bg_x1 = px1
            w_bg_x2 = px1 + w_tw + 8
            overlay = frame.copy()
            cv2.rectangle(overlay, (w_bg_x1, label_y_top), (w_bg_x2, label_y_top + label_h), (0, 80, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (w_bg_x1, label_y_top), (w_bg_x2, label_y_top + label_h), COLOR_GREEN, 1)
            cv2.putText(frame, w_label, (w_bg_x1 + 4, label_y_top + label_h - 4),
                        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            # Not-Working timer - top-right (red background)
            nw_bg_x2 = px2
            nw_bg_x1 = px2 - nw_tw - 8
            overlay = frame.copy()
            cv2.rectangle(overlay, (nw_bg_x1, label_y_top), (nw_bg_x2, label_y_top + label_h), (0, 0, 80), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (nw_bg_x1, label_y_top), (nw_bg_x2, label_y_top + label_h), COLOR_RED, 1)
            cv2.putText(frame, nw_label, (nw_bg_x1 + 4, label_y_top + label_h - 4),
                        font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            # --- Bottom label: Worker ID + Status ---
            gender = s.get("gender", "Unknown")
            gender_tag = f" {gender}" if gender not in (None, "Unknown") else ""
            conf_pct = s.get('conf', 0) * 100
            bottom_label = f"#{tid}{gender_tag} | {status_label} {conf_pct:.0f}%"
            (bt_w, bt_h), _ = cv2.getTextSize(bottom_label, font, font_scale, thickness)

            bt_y = min(height - 2, py2 + bt_h + 8)
            overlay = frame.copy()
            cv2.rectangle(overlay, (px1, bt_y - bt_h - 4), (px1 + bt_w + 8, bt_y + 2), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (px1, bt_y - bt_h - 4), (px1 + bt_w + 8, bt_y + 2), color, 1)
            cv2.putText(frame, bottom_label, (px1 + 4, bt_y - 2),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # --- Top HUD bar ---
        total = working_count + not_working_count
        productivity = (working_count / total * 100) if total else 0.0
        hud_h = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, hud_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (0, 0), (width, hud_h), (60, 60, 60), 1)

        cv2.putText(frame, f"Workers: {total}   Working: {working_count}   Not Working: {not_working_count}   Productivity: {productivity:.0f}%",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Per-worker summary in HUD
        hud_x = 10
        for tid, s in sorted(self._worker_states.items()):
            if s.get("lost_frames", 0) > 0:
                continue
            w_str = _fmt_time(s.get("working_s", 0.0))
            nw_str = _fmt_time(s.get("not_working_s", 0.0))
            summary = f"#{tid} W:{w_str} NW:{nw_str}"
            cv2.putText(frame, summary, (hud_x, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            (sw, _), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            hud_x += sw + 20

        return frame

    # ------------------------------------------------------------------ #
    # Model 6: Queue Monitor (people count in fixed zone)
    # ------------------------------------------------------------------ #
    def _process_queue(self, frame):
        height, width = frame.shape[:2]
        
        # Load custom 4-point ROI if set, otherwise fallback to default rectangle
        if self.queue_roi and len(self.queue_roi) == 4:
            pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in self.queue_roi], np.int32)
        else:
            pts = np.array([
                [int(width * 0.1), int(height * 0.4)],
                [int(width * 0.9), int(height * 0.4)],
                [int(width * 0.9), int(height * 0.9)],
                [int(width * 0.1), int(height * 0.9)]
            ], np.int32)
            
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (255, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "QUEUE ZONE", (pts[0][0], max(30, pts[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Run YOLO with a lower confidence threshold (0.15) to detect heavily occluded people in the queue zone
        results = self._yolo(frame, classes=[PERSON_CLASS], conf=0.15, verbose=False)
        people_in_line = 0
        if results[0].boxes is not None:
            person_candidates, person_scores = [], []
            for box, conf in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.conf.cpu()):
                x1, y1, x2, y2 = map(int, box)
                person_candidates.append([x1, y1, x2, y2])
                person_scores.append(float(conf))
            
            # Apply NMS to remove overlapping/duplicate boxes
            keep = nms_boxes(person_candidates, person_scores, iou_threshold=0.4)
            
            for idx in keep:
                x1, y1, x2, y2 = person_candidates[idx]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Check if person's center is inside custom polygon ROI
                in_line = cv2.pointPolygonTest(pts, (cx, cy), False) >= 0
                color = (0, 255, 0) if in_line else (0, 0, 255)
                if in_line:
                    people_in_line += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)

        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        count_color = (0, 255, 0) if people_in_line <= 7 else (0, 0, 255)
        cv2.putText(frame, f"People in Line: {people_in_line}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, count_color, 3)

        if time.time() - self._queue_last_alarm > 2.0:
            if people_in_line > 7:
                os.system("afplay /System/Library/Sounds/Ping.aiff &")
            if time.time() - self._last_queue_photo > 5.0:
                self._last_queue_photo = time.time()
                photo_path = f"static/events/{int(time.time())}_queue_{people_in_line}.jpg"
                cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                database.log_queue(people_in_line, photo_path)
            
        if people_in_line >= 5 and time.time() - self._queue_last_alarm > 10.0:
            database.log_event("queue", "long_queue", f"{people_in_line} people in line!", "alert")

        return frame

    def _process_box(self, frame):
        # We'll use YOLOv8 class 28 (suitcase) as a proxy for box/sack for this demonstration
        results = self._yolo(frame, classes=[28], conf=0.2, verbose=False)
        r = results[0]
        boxes = []
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy()
            boxes.append({'bbox': b})
            
        with self._status_lock:
            line_pts = list(self._box_line)
            
        out_frame, event_occurred, event_type = self._box_counter.process_frame(frame.copy(), boxes, line_pts)
        
        if event_occurred:
            database.log_event("box", event_type, f"A box was {event_type}", "info")
            photo_path = f"static/events/{int(time.time())}_box_{event_type}.jpg"
            cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), out_frame)
            database.log_box(self._box_counter.loaded_count, self._box_counter.unloaded_count, photo_path)
            
        return out_frame



# Single shared instance used by the Flask app
worker = CameraWorker()