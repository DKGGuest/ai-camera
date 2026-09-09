import csv
import json
import os
import threading
import time
from collections import defaultdict

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
import cv2
import numpy as np

from src.monitors.box_counter import BoxCounter
import src.core.models as models

import config
import database
import models_loader
import math

MODES = ("access", "vehicle", "adaptive", "people", "worker", "queue", "box", "desk", "room")
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

def point_in_box(point, box, padding=0):
    """Check if a point (x, y) is inside a bounding box (x1, y1, x2, y2) with optional padding."""
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding)

def boxes_intersect(box1, box2, threshold=0.15):
    """Check if two bounding boxes intersect significantly (IoA)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x1_i < x2_i and y1_i < y2_i:
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # If the overlap covers more than `threshold` of EITHER box
        if intersection_area / float(max(area1, 1)) > threshold or intersection_area / float(max(area2, 1)) > threshold:
            return True
    return False

def estimate_head_pose(landmarks, fy, fh, py1, py2, camera_angle="elevated"):
    right_eye, left_eye = landmarks[0], landmarks[1]
    nose = landmarks[2]
    right_mouth, left_mouth = landmarks[3], landmarks[4]

    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    if eye_dist < 1.0:
        return 0.0, 0.0, 0.0, 0.5

    eye_cx = (right_eye[0] + left_eye[0]) / 2.0
    eye_cy = (right_eye[1] + left_eye[1]) / 2.0
    mouth_cy = (right_mouth[1] + left_mouth[1]) / 2.0

    yaw_ratio = (nose[0] - eye_cx) / eye_dist
    pitch_ratio = (mouth_cy - eye_cy) / eye_dist
    
    face_center_y = fy + fh / 2.0
    person_h = py2 - py1
    face_rel_y = face_center_y / person_h if person_h > 0 else 0.5

    if camera_angle == "elevated":
        yaw_max, pitch_min, face_y_max = 0.8, -0.5, 0.85
    elif camera_angle == "side":
        yaw_max, pitch_min, face_y_max = 1.2, -0.2, 0.75
    else:
        yaw_max, pitch_min, face_y_max = 0.5, -0.1, 0.65

    score = 1.0
    if abs(yaw_ratio) > yaw_max: score -= 0.5
    if pitch_ratio < pitch_min: score -= 0.8
    if face_rel_y > face_y_max: score -= 0.5

    return max(0.0, min(1.0, score)), yaw_ratio, pitch_ratio, face_rel_y


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
        self._desk_model = None
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
        self._people_id_mapping = {}
        self._people_track_history = {}
        self._people_entry_count = 0
        self._people_exit_count = 0
        self._last_people_photo = 0.0
        self._people_lines = settings.get("people_lines", [])

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
        self._box_line = settings.get("box_line", [])

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
        self._people_lines = settings.get("people_lines", [])
        self._box_line = settings.get("box_line", [])

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

    def set_people_lines(self, lines):
        with self._status_lock:
            self._people_lines = lines

    def reset_people_counts(self):
        with self._status_lock:
            self._people_entry_count = 0
            self._people_exit_count = 0
            self._people_tracks.clear()
            self._people_id_mapping.clear()
            self._people_track_history.clear()

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
        if not hasattr(self, '_desk_model') or self._desk_model is None:
            self._desk_model = models_loader.load_desk_model()
        if not hasattr(self, '_desk_pose_model') or self._desk_pose_model is None:
            from ultralytics import YOLO
            self._desk_pose_model = YOLO("yolov8m-pose.pt")
        if not hasattr(self, '_custom_box_model') or self._custom_box_model is None:
            self._custom_box_model = models_loader.load_custom_box_model()
        if not hasattr(self, '_worker_classifier') or self._worker_classifier is None:
            self._worker_classifier = models_loader.load_worker_classifier()
        if not hasattr(self, '_chair_classifier') or self._chair_classifier is None:
            import os
            from ultralytics import YOLO
            classifier_path = os.path.join(os.path.dirname(__file__), 'runs', 'classify', 'chair_classifier', 'weights', 'best.pt')
            if os.path.exists(classifier_path):
                self._chair_classifier = YOLO(classifier_path)
                print(f"[DeskOccupancy] Loaded chair classifier from {classifier_path}")
            else:
                self._chair_classifier = None
                print(f"[DeskOccupancy] WARNING: Chair classifier not found at {classifier_path}")
        self._box_counter = BoxCounter()
        self._face_detector, self._face_recognizer = models_loader.load_face_models()
        self._known_embeddings = models_loader.load_known_faces(
            config.KNOWN_FACES_DIR, self._face_detector, self._face_recognizer
        )

    def _open_capture(self):
        if str(self.camera_url).isdigit():
            src = int(self.camera_url)
            # For local USB webcams on Windows, DirectShow is the lowest latency backend
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Optional: force 30fps to avoid camera driver buffering
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        else:
            src = self.camera_url
            import urllib.parse
            parsed = urllib.parse.urlparse(src)
            # If the user enters a raw IP:PORT from an app like Android "IP Webcam", the actual stream is at /video
            if parsed.scheme in ('http', 'https') and parsed.path in ('', '/'):
                src = src.rstrip('/') + '/video'
                
            # Automatically convert Main Stream to Sub Stream for known IP cameras (CP Plus, Dahua, Hikvision)
            # CP Plus / Dahua
            if "subtype=0" in src:
                src = src.replace("subtype=0", "subtype=1")
            # Hikvision
            elif "Streaming/Channels/101" in src:
                src = src.replace("Streaming/Channels/101", "Streaming/Channels/102")
                
            # For network streams (RTSP/HTTP), use FFMPEG with no-buffer flags
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

    def _http_mjpeg_loop(self, src):
        import urllib.request
        import urllib.parse
        parsed = urllib.parse.urlparse(src)
        if parsed.path in ('', '/'):
            src = src.rstrip('/') + '/video'
            
        try:
            stream = urllib.request.urlopen(src, timeout=5)
            bytes_data = b''
            while not self._stop and self._requested_camera_url is None:
                chunk = stream.read(16384)
                if not chunk:
                    break
                bytes_data += chunk
                
                # Find the last complete JPEG frame in the accumulated buffer
                b = bytes_data.rfind(b'\xff\xd9')
                if b != -1:
                    a = bytes_data.rfind(b'\xff\xd8', 0, b)
                    if a != -1:
                        jpg = bytes_data[a:b+2]
                        # Discard old frames by keeping only the remainder
                        bytes_data = bytes_data[b+2:]
                        
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._status_lock:
                                self._connected = True
                                self._last_error = ""
                            with self._raw_frame_lock:
                                self._raw_frame = frame
                            self._last_raw_frame = frame
        except Exception as e:
            with self._status_lock:
                self._connected = False
                self._last_error = f"Stream error: {e}"
            time.sleep(2.0)

    def _capture_loop(self):
        """Runs in its own thread, keeping self._raw_frame as fresh as possible. 
        Features a custom zero-lag MJPEG reader for HTTP streams (ESP32-CAM) and 
        aggressive OpenCV grabbing for RTSP."""
        while not self._stop:
            if self._requested_camera_url is not None:
                self.camera_url = self._requested_camera_url
                self._requested_camera_url = None
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                with self._raw_frame_lock:
                    self._raw_frame = None

            src = str(self.camera_url)
            if src.startswith('http'):
                self._http_mjpeg_loop(src)
            else:
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
                elif self.mode == "desk":
                    out = self._process_desk(frame)
                elif self.mode == "room":
                    out = self._process_room(frame)
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
    # Model: Desk Occupancy
    # ------------------------------------------------------------------ #
    class TrackedChair:
        def __init__(self, box):
            self.box = box
            self.missed_frames = 0
            self.matched = True
            
        def update(self, box):
            # Smooth the box transitions slightly to prevent flickering
            alpha = 0.5
            x1 = int(self.box[0] * alpha + box[0] * (1 - alpha))
            y1 = int(self.box[1] * alpha + box[1] * (1 - alpha))
            x2 = int(self.box[2] * alpha + box[2] * (1 - alpha))
            y2 = int(self.box[3] * alpha + box[3] * (1 - alpha))
            
            # Enforce a minimum size to ensure it's a "large rectangle" even if partially occluded
            w = max(x2 - x1, 60)
            h = max(y2 - y1, 80)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            self.box = (cx - w//2, cy - h//2, cx + w//2, cy + h//2)
            self.missed_frames = 0

    def _process_desk(self, frame):
        import math
        
        def calculate_angle(p1, p2, p3):
            if p1[0] == 0 or p2[0] == 0 or p3[0] == 0: return 0
            v1 = [p1[0] - p2[0], p1[1] - p2[1]]
            v2 = [p3[0] - p2[0], p3[1] - p2[1]]
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.hypot(v1[0], v1[1])
            mag2 = math.hypot(v2[0], v2[1])
            if mag1 * mag2 == 0: return 0
            angle_rad = math.acos(max(-1.0, min(1.0, dot_product / (mag1 * mag2))))
            return math.degrees(angle_rad)

        def is_duplicate_box(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea <= 0: return False
            areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            minArea = min(areaA, areaB)
            if minArea <= 0: return False
            unionArea = areaA + areaB - interArea
            iou = interArea / float(unionArea)
            ioma = interArea / float(minArea) # Intersection over Minimum Area
            return iou > 0.35 or ioma > 0.45

        PERSON_CONF_THRESHOLD = 0.30
        CHAIR_CONF_THRESHOLD = 0.15 # Lower threshold to detect occluded/far chairs (like yellow circle)
        
        if not hasattr(self, '_tracked_chairs_list'):
            self._tracked_chairs_list = []
        
        # 1. Chairs Detection
        results_chairs = self._desk_model(frame, classes=[56], conf=CHAIR_CONF_THRESHOLD, iou=0.1, imgsz=1280, verbose=False)
        detected_chairs = []
        if results_chairs[0].boxes is not None:
            for box in results_chairs[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) > 250: # Filter unusually wide boxes (likely two merged chairs)
                    continue
                detected_chairs.append((x1, y1, x2, y2))
                
        # Filter raw detections to eliminate duplicate sub-boxes on the same chair
        filtered_detected_chairs = []
        for d_box in detected_chairs:
            overlap = False
            for f_box in filtered_detected_chairs:
                if is_duplicate_box(d_box, f_box):
                    overlap = True
                    break
            if not overlap:
                filtered_detected_chairs.append(d_box)
                
        # Update Tracked Chairs
        for tc in self._tracked_chairs_list:
            tc.matched = False
            
        for d_box in filtered_detected_chairs:
            best_tc = None
            for tc in self._tracked_chairs_list:
                if is_duplicate_box(d_box, tc.box):
                    best_tc = tc
                    break
            if best_tc:
                best_tc.update(d_box)
                best_tc.matched = True
            else:
                self._tracked_chairs_list.append(self.TrackedChair(d_box))
                
        # Retain active tracked chairs (reducing ghost lifetime to ~1s) and deduplicate active chairs
        active_chairs = []
        for tc in self._tracked_chairs_list:
            if not tc.matched: tc.missed_frames += 1
            if tc.missed_frames < 20: # Remember chairs for ~1 second max
                duplicate = False
                for existing in active_chairs:
                    if is_duplicate_box(tc.box, existing.box):
                        duplicate = True
                        break
                if not duplicate:
                    active_chairs.append(tc)
                    
        self._tracked_chairs_list = active_chairs
        chairs = [tc.box for tc in active_chairs]
        
        occupied_chair_indices = set()

        # 2. Person & Posture detection (Pose model for sitting detection)
        sitting_people = []
        if hasattr(self, '_desk_pose_model') and self._desk_pose_model is not None:
            results_pose = self._desk_pose_model(frame, conf=PERSON_CONF_THRESHOLD, imgsz=1280, verbose=False)
            
            if results_pose[0].keypoints is not None and results_pose[0].boxes is not None:
                for idx in range(len(results_pose[0].boxes)):
                    box = results_pose[0].boxes[idx]
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    center_x, center_y = int((px1 + px2) / 2), int((py1 + py2) / 2)
                    
                    keypoints = results_pose[0].keypoints.data[idx]
                    l_shoulder, l_hip, l_knee = keypoints[5], keypoints[11], keypoints[13]
                    r_shoulder, r_hip, r_knee = keypoints[6], keypoints[12], keypoints[14]
                    
                    knees_visible = False
                    angle = 180
                    kp_conf = 0.3
                    if l_shoulder[2] > kp_conf and l_hip[2] > kp_conf and l_knee[2] > kp_conf:
                        angle = calculate_angle(l_shoulder[:2], l_hip[:2], l_knee[:2])
                        knees_visible = True
                    elif r_shoulder[2] > kp_conf and r_hip[2] > kp_conf and r_knee[2] > kp_conf:
                        angle = calculate_angle(r_shoulder[:2], r_hip[:2], r_knee[:2])
                        knees_visible = True
                            
                    is_lying_down = (px2 - px1) > (py2 - py1) * 0.8
                    
                    if is_lying_down:
                        status = "Sitting"
                    elif knees_visible and 45 <= angle <= 165:
                        status = "Sitting"
                    elif not knees_visible and (py2 - py1) < (px2 - px1) * 3.0:
                        # Fallback: if knees are occluded (e.g., under desk) and bounding box is not extremely tall, assume sitting
                        status = "Sitting"
                    else:
                        status = "Standing"
                    
                    # Estimate chest location for chair matching
                    if l_shoulder[2] > kp_conf and r_shoulder[2] > kp_conf:
                        chest_x = int((l_shoulder[0] + r_shoulder[0]) / 2)
                        chest_y = int((l_shoulder[1] + r_shoulder[1]) / 2) + 20
                    elif l_shoulder[2] > kp_conf:
                        chest_x, chest_y = int(l_shoulder[0]), int(l_shoulder[1]) + 20
                    elif r_shoulder[2] > kp_conf:
                        chest_x, chest_y = int(r_shoulder[0]), int(r_shoulder[1]) + 20
                    else:
                        chest_x = center_x
                        chest_y = int(py1 + 0.3 * (py2 - py1))
                        
                    # Get all keypoints for head-to-toe polygon
                    poly_points = []
                    for kp_idx in range(17):
                        if keypoints[kp_idx][2] > 0.3:
                            poly_points.append([int(keypoints[kp_idx][0]), int(keypoints[kp_idx][1])])
                    
                    # Draw polygon structure instead of point marks
                    box_color = (255, 100, 100) if status == "Sitting" else (100, 255, 255)
                    if len(poly_points) >= 3:
                        pts = np.array(poly_points, np.int32).reshape((-1, 1, 2))
                        hull = cv2.convexHull(pts)
                        # Draw polygon structure
                        # cv2.polylines(frame, [hull], isClosed=True, color=box_color, thickness=3)
                        
                    label = f"{status} ({int(angle)} deg)"
                    cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    
                    if status == "Sitting":
                        sitting_people.append({
                            'box': (px1, py1, px2, py2),
                            'center': (center_x, center_y),
                            'chest': (chest_x, chest_y)
                        })

        # Calculate candidate matching scores between sitting people and chair boxes
        candidate_matches = []
        for p_idx, person in enumerate(sitting_people):
            px1, py1, px2, py2 = person['box']
            hx, hy = person['chest']
            p_area = (px2 - px1) * (py2 - py1)
            
            for c_idx, chair_box in enumerate(chairs):
                cx1, cy1, cx2, cy2 = chair_box
                c_area = (cx2 - cx1) * (cy2 - cy1)
                
                ix1, iy1 = max(px1, cx1), max(py1, cy1)
                ix2, iy2 = min(px2, cx2), min(py2, cy2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                
                # Check if person's chest is inside chair box (with slight padding)
                chest_in_chair = (cx1 - 15 <= hx <= cx2 + 15) and (cy1 - 15 <= hy <= cy2 + 25)
                
                if inter_area > 0.15 * p_area or inter_area > 0.15 * c_area or chest_in_chair:
                    iou = inter_area / float(p_area + c_area - inter_area) if (p_area + c_area - inter_area) > 0 else 0
                    score = iou + (2.5 if chest_in_chair else 0.5)
                    candidate_matches.append((score, p_idx, c_idx))

        # Greedy 1-to-1 assignment: 1 sitting person occupies at most 1 chair
        candidate_matches.sort(key=lambda x: x[0], reverse=True)
        assigned_persons = set()
        assigned_chairs = set()
        
        for score, p_idx, c_idx in candidate_matches:
            if p_idx not in assigned_persons and c_idx not in assigned_chairs:
                assigned_persons.add(p_idx)
                assigned_chairs.add(c_idx)
                occupied_chair_indices.add(c_idx)
        # Artificially create chairs for sitting people who missed chair detection
        for p_idx, person in enumerate(sitting_people):
            if p_idx not in assigned_persons:
                px1, py1, px2, py2 = person['box']
                cx1 = px1 - 15
                cy1 = py1 + int((py2 - py1) * 0.3)
                cx2 = px2 + 15
                cy2 = py2 + 20
                
                chairs.append((cx1, cy1, cx2, cy2))
                occupied_chair_indices.add(len(chairs) - 1)

        # 3. AI Classifier signal (for remaining unassigned chairs)
        if hasattr(self, '_chair_classifier') and self._chair_classifier is not None:
            h, w = frame.shape[:2]
            for i, chair_box in enumerate(chairs):
                if i in occupied_chair_indices:
                    continue # already marked occupied by a sitting person
                cx1, cy1, cx2, cy2 = chair_box
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                if cx2 - cx1 < 10 or cy2 - cy1 < 10:
                    continue
                chair_crop = frame[cy1:cy2, cx1:cx2]
                cls_results = self._chair_classifier(chair_crop, imgsz=224, verbose=False)
                if cls_results and cls_results[0].probs is not None:
                    probs = cls_results[0].probs
                    predicted_class = cls_results[0].names[probs.top1]
                    confidence = probs.top1conf.item()
                    if predicted_class == 'occupied' and confidence > 0.70:
                        occupied_chair_indices.add(i)


        # 4. Draw chairs with occupancy status
        for i, chair_box in enumerate(chairs):
            cx1, cy1, cx2, cy2 = chair_box
            is_occupied = (i in occupied_chair_indices)
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            label = "Occupied" if is_occupied else "Empty"
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (cx1, cy1 - th - 10), (cx1 + tw, cy1), color, -1)
            cv2.putText(frame, label, (cx1, cy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return frame

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
            
            # Always draw a rectangle around the detected face (red by default)
            box_color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 3)
            
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
                    # Redraw rectangle with green color
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), box_color, 3)
                    last = self._last_access_log.get(best_match, 0)
                    if now - last > 5.0:
                        photo_path = f"static/events/{int(now)}_{best_match}.jpg"
                        cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                        database.log_access(best_match, "granted", photo_path)
                        self._last_access_log[best_match] = now
                else:
                    status_text = "ACCESS DENIED"
                    status_color = (0, 0, 200)
                    last = self._last_access_log.get("Unknown", 0)
                    if now - last > 5.0:
                        photo_path = f"static/events/{int(now)}_unknown.jpg"
                        cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                        database.log_access("Unknown person", "denied", photo_path)
                        self._last_access_log["Unknown"] = now

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
        
        # Default lines if not set in UI
        if not self._people_lines or len(self._people_lines) != 2:
            gap = 20  # ~0.5 cm gap
            mid_x = width // 2
            line1 = [(mid_x - gap, 0), (mid_x - gap, height)]
            line2 = [(mid_x + gap, 0), (mid_x + gap, height)]
        else:
            pts = self._people_lines
            line1 = [
                (int(pts[0][0] * width), int(pts[0][1] * height)),
                (int(pts[0][2] * width), int(pts[0][3] * height))
            ]
            line2 = [
                (int(pts[1][0] * width), int(pts[1][1] * height)),
                (int(pts[1][2] * width), int(pts[1][3] * height))
            ]

        results = self._yolo.track(frame, persist=True, classes=[PERSON_CLASS], verbose=False, conf=0.20, iou=0.4, tracker="custom_botsort.yaml")
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        # Draw lines
        cv2.line(frame, line1[0], line1[1], (0, 255, 0), 2)
        cv2.line(frame, line2[0], line2[1], (0, 255, 255), 2)
        cv2.putText(frame, "L1 (OUTSIDE)", (line1[0][0] + 10, line1[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "L2 (INSIDE)", (line2[0][0] + 10, line2[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        now = time.time()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, raw_track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # --- FALLBACK RE-LINKING ---
                if raw_track_id not in self._people_id_mapping and raw_track_id not in self._people_track_history:
                    best_old_id = None
                    best_dist = 500.0  # Allow large jumps for fast movement
                    
                    for old_id, info in self._people_track_history.items():
                        time_since_lost = now - info['time']
                        if 0 < time_since_lost < 1.5:
                            dist = math.hypot(cx - info['cx'], cy - info['cy'])
                            if dist < best_dist:
                                best_dist = dist
                                best_old_id = old_id
                                
                    if best_old_id is not None:
                        self._people_id_mapping[raw_track_id] = best_old_id
                
                track_id = self._people_id_mapping.get(raw_track_id, raw_track_id)
                
                # Centroid smoothing to prevent flickering/fluctuating dots
                if track_id in self._people_tracks and self._people_tracks[track_id]["history"]:
                    prev_cx, prev_cy, _ = self._people_tracks[track_id]["history"][-1]
                    alpha = 0.5  # 50% current, 50% previous
                    cx = int(alpha * cx + (1 - alpha) * prev_cx)
                    cy = int(alpha * cy + (1 - alpha) * prev_cy)

                self._people_track_history[track_id] = {'time': now, 'cx': cx, 'cy': cy}
                
                if track_id not in self._people_tracks:
                    self._people_tracks[track_id] = {
                        "history": [(cx, cy, now)],
                        "crossings": [],
                        "last_cross_time": 0,
                        "last_counted_time": 0,
                        "last_update": now,
                        "first_seen": now
                    }
                else:
                    t = self._people_tracks[track_id]
                    t["last_update"] = now
                    t["history"].append((cx, cy, now))
                    if len(t["history"]) > 30:
                        t["history"].pop(0)
                        
                    prev_pt = (t["history"][-2][0], t["history"][-2][1]) if len(t["history"]) > 1 else (cx, cy)
                    curr_pt = (cx, cy)
                    
                    if now - t.get("last_counted_time", 0) > 2.0:
                        mid1_x = (line1[0][0] + line1[1][0]) / 2
                        mid2_x = (line2[0][0] + line2[1][0]) / 2
                        
                        prev_x, curr_x = prev_pt[0], curr_pt[0]
                        
                        crossed_lines = []
                        if min(prev_x, curr_x) <= mid1_x <= max(prev_x, curr_x) and prev_x != curr_x:
                            crossed_lines.append(1)
                        if min(prev_x, curr_x) <= mid2_x <= max(prev_x, curr_x) and prev_x != curr_x:
                            crossed_lines.append(2)
                            
                        if len(crossed_lines) == 2:
                            dist1 = abs(prev_x - mid1_x)
                            dist2 = abs(prev_x - mid2_x)
                            if dist1 < dist2:
                                crossed_lines = [1, 2]
                            else:
                                crossed_lines = [2, 1]
                                
                        for crossed in crossed_lines:
                            if t.get("last_cross_time", 0) and now - t["last_cross_time"] > 15.0:
                                t["crossings"] = []
                            if not t["crossings"] or t["crossings"][-1] != crossed:
                                t["crossings"].append(crossed)
                                t["last_cross_time"] = now
                                
                        if len(t["crossings"]) >= 2:
                            seq = t["crossings"][-2:]
                            if seq == [1, 2] or seq == [2, 1]:
                                oldest_pt = t["history"][0]
                                displacement = math.hypot(cx - oldest_pt[0], cy - oldest_pt[1])
                                min_dist = 5.0  # Relaxed to flat 5.0 pixels to capture movement during stream lag
                                
                                if displacement >= min_dist:
                                    if seq == [1, 2]:
                                        self._people_entry_count += 1
                                        t["last_counted_time"] = now
                                        t["crossings"] = []
                                        database.log_event("people", "entry", f"Track {track_id}", "info")
                                        photo_path = f"static/events/{int(now)}_people_entry.jpg"
                                        cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                                        database.log_people("entry", self._people_entry_count, self._people_exit_count, photo_path)
                                    elif seq == [2, 1]:
                                        self._people_exit_count += 1
                                        t["last_counted_time"] = now
                                        t["crossings"] = []
                                        database.log_event("people", "exit", f"Track {track_id}", "info")
                                        photo_path = f"static/events/{int(now)}_people_exit.jpg"
                                        cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                                        database.log_people("exit", self._people_entry_count, self._people_exit_count, photo_path)
                                else:
                                    t["crossings"] = []  # Reject due to jitter
                                    
                    # Stationary check
                    if now - t["first_seen"] > 10.0:
                        oldest_pt = t["history"][0]
                        displacement = math.hypot(cx - oldest_pt[0], cy - oldest_pt[1])
                        lx1, lx2 = line1[0][0], line2[0][0]
                        min_x, max_x = min(lx1, lx2) - 50, max(lx1, lx2) + 50
                        if min_x < cx < max_x and displacement < (x2 - x1):
                            self._stationary_warning_until = now + 5.0
                
                color = (0, 165, 255) # Thin bounding box (orange)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Cleanup old tracks
        to_delete = []
        for tid, t in self._people_tracks.items():
            if now - t["last_update"] > 5.0: # 5 seconds cooldown
                to_delete.append(tid)
        for tid in to_delete:
            del self._people_tracks[tid]
            
        if now - self._last_people_photo > 5.0:
            self._last_people_photo = now
            photo_path = f"static/events/{int(now)}_people_periodic.jpg"
            cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
            database.log_people("periodic", self._people_entry_count, self._people_exit_count, photo_path)

        # Calculate mid points to determine which side is "inside"
        mid1_x = (line1[0][0] + line1[1][0]) / 2
        mid2_x = (line2[0][0] + line2[1][0]) / 2
        
        # Count live tracks
        active_inside = 0
        total_person = 0
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            total_person = len(boxes)
            for box in boxes:
                cx = (box[0] + box[2]) / 2
                if mid2_x >= mid1_x and cx >= mid2_x:
                    active_inside += 1
                elif mid2_x < mid1_x and cx <= mid2_x:
                    active_inside += 1
                    
        cv2.rectangle(frame, (10, 10), (320, 150), (0, 0, 0), -1)
        cv2.putText(frame, f"IN: {self._people_entry_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self._people_exit_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Person: {total_person}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Inside: {active_inside}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        if getattr(self, "_stationary_warning_until", 0) > now:
            warn_text = "WARNING: Lines may be placed in a seating/work area!"
            cv2.rectangle(frame, (10, height - 60), (width - 10, height - 20), (0, 0, 255), -1)
            cv2.putText(frame, warn_text, (20, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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

            # --- 1-to-1 Laptop Assignment ---
            # Prevents multiple people from claiming the same laptop
            laptop_assignments = {}
            for l_idx, (lb, _lconf, _lname) in enumerate(laptop_boxes):
                lcx, lcy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
                best_tid = None
                min_dist = float('inf')
                for tid, p_box, p_conf in person_dets:
                    px1, py1, px2, py2 = p_box
                    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                    
                    dist = math.hypot(pcx - lcx, pcy - lcy)
                    margin = 50
                    is_overlapping = (px1 - margin <= lcx <= px2 + margin) and (py1 - margin <= lcy <= py2 + margin)
                    
                    # Must be touching or within expanded distance
                    if is_overlapping or dist < (px2 - px1) * 2.5:
                        if dist < min_dist:
                            min_dist = dist
                            best_tid = tid
                            
                if best_tid is not None:
                    laptop_assignments[l_idx] = best_tid

            seen_ids = set()
            for tid, p_box, p_conf in person_dets:
                seen_ids.add(tid)
                x1, y1, x2, y2 = p_box
                px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
                px2, py2 = min(width, x2 + 20), min(height, y2 + 20)
                roi = frame[py1:py2, px1:px2]

                # --- YuNet Face Detection + Head Pose Estimation ---
                face_box_abs = None
                head_score, yaw_ratio, pitch_ratio, face_rel_y = 0.0, 0.0, 0.0, 0.5
                if roi.size > 0:
                    try:
                        rh, rw = roi.shape[:2]
                        if rw > 30 and rh > 30:
                            self._worker_face_det.setInputSize((rw, rh))
                            _, detected_faces = self._worker_face_det.detect(roi)
                            if detected_faces is not None and len(detected_faces) > 0:
                                best_face = max(detected_faces, key=lambda f: f[2] * f[3])
                                fx, fy_f, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                                face_box_abs = [px1 + fx, py1 + fy_f, px1 + fx + fw, py1 + fy_f + fh]
                                
                                landmarks = best_face[4:14].reshape(5, 2)
                                head_score, yaw_ratio, pitch_ratio, face_rel_y = estimate_head_pose(
                                    landmarks, fy_f, fh, 0, rh, camera_angle="elevated"
                                )
                    except Exception:
                        pass

                # Standing check
                is_standing = False
                p_w = x2 - x1
                p_h = y2 - y1
                if p_w > 0 and (p_h / float(p_w)) > 2.2: # Increased threshold because upper body sitting can be tall
                    is_standing = True

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

                # Nearest-laptop association (Strict 1-to-1)
                has_laptop = any(laptop_assignments.get(l_idx) == tid for l_idx in range(len(laptop_boxes)))

                ai_working = False
                if hasattr(self, '_worker_classifier') and self._worker_classifier is not None and roi.size > 0:
                    try:
                        cls_res = self._worker_classifier(roi, verbose=False)[0]
                        top1_name = cls_res.names[cls_res.probs.top1]
                        ai_working = (top1_name == 'working')
                    except Exception:
                        pass

                st = self._worker_states.get(tid)
                if st is None:
                    self._worker_states[tid] = {
                        "work_s": 0.0, "idle_s": 0.0, "phone_s": 0.0, "talk_s": 0.0,
                        "working_s": 0.0, "not_working_s": 0.0,
                        "looking_away_s": 8.1,  # Default to > 8s so new detections start as NOT WORKING until proven otherwise
                        "status": "Idle",
                        "streak": 0, "gender": "Unknown", "box": p_box, "conf": p_conf,
                        "lost_frames": 0, "raw_working": False, "phone_near": phone_near,
                        "has_laptop": has_laptop, "is_standing": is_standing,
                        "score": 0, "head_metrics": (0.0, 0.0, 0.0, 0.5)
                    }
                    st = self._worker_states[tid]
                else:
                    st["box"], st["conf"], st["lost_frames"] = p_box, p_conf, 0
                    st["phone_near"], st["has_laptop"] = phone_near, has_laptop
                    st["is_standing"] = is_standing
                
                # Strict Working Logic (60/40 Split)
                is_working_now = False
                score = 0
                
                # Base condition: Must be sitting with a laptop and no phone
                if not is_standing and has_laptop and not phone_near:
                    score += 60  # Base 60% just for sitting in front of laptop
                    
                    # Remaining 40% from other AI/posture conditions
                    if ai_working: 
                        score += 20
                    score += head_score * 20
                    
                    # Any score >= 60 means they are working (so just sitting + laptop is enough)
                    if score >= 60:
                        is_working_now = True

                if is_working_now:
                    st["looking_away_s"] = 0.0
                else:
                    st["looking_away_s"] += dt

                # Hard Constraints override grace period
                if is_standing or not has_laptop or phone_near:
                    st["raw_working"] = False
                else:
                    st["raw_working"] = (st["looking_away_s"] <= 8.0)

                st["score"] = score
                st["head_metrics"] = (head_score, yaw_ratio, pitch_ratio, face_rel_y)

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
                    
                if s.get("is_standing"):
                    target = "Standing"
                elif not s["has_laptop"]:
                    target = "No Laptop"
                elif s["phone_near"]:
                    target = "Using Phone"
                elif tid in talking_ids:
                    target = "Talking"
                elif s["raw_working"]:
                    target = "Working"
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

            # Debug HUD overlay removed by user request

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
        
        # Load custom ROIs
        queue_rois = []
        if self.queue_roi and isinstance(self.queue_roi, list) and len(self.queue_roi) > 0:
            if isinstance(self.queue_roi[0][0], list) or isinstance(self.queue_roi[0][0], tuple):
                # Multiple queues
                for roi in self.queue_roi:
                    if len(roi) == 4:
                        pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in roi], np.int32)
                        queue_rois.append(pts)
            elif len(self.queue_roi) == 4:
                # Single queue (backwards compatibility)
                pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in self.queue_roi], np.int32)
                queue_rois.append(pts)
        
        if not queue_rois:
            pts = np.array([
                [int(width * 0.1), int(height * 0.4)],
                [int(width * 0.9), int(height * 0.4)],
                [int(width * 0.9), int(height * 0.9)],
                [int(width * 0.1), int(height * 0.9)]
            ], np.int32)
            queue_rois.append(pts)

        # Draw ROIs
        overlay = frame.copy()
        for i, pts in enumerate(queue_rois):
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.fillPoly(overlay, [pts], (255, 255, 0))
            cv2.putText(frame, f"LINE {i+1}", (pts[0][0], max(30, pts[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Run YOLO with a lower confidence threshold (0.15) to detect heavily occluded people in the queue zone
        results = self._yolo(frame, classes=[PERSON_CLASS], conf=0.15, verbose=False)
        
        # Track counts per line
        line_counts = [0] * len(queue_rois)
        total_people = 0
        
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
                
                # Check if person's center is inside any custom polygon ROI
                in_line_idx = -1
                for i, pts in enumerate(queue_rois):
                    if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                        in_line_idx = i
                        break
                        
                color = (0, 255, 0) if in_line_idx != -1 else (0, 0, 255)
                if in_line_idx != -1:
                    line_counts[in_line_idx] += 1
                    total_people += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)

        # Draw counts for all lines
        cv2.rectangle(frame, (10, 10), (400, 40 + 35 * len(queue_rois)), (0, 0, 0), -1)
        for i, count in enumerate(line_counts):
            count_color = (0, 255, 0) if count <= 7 else (0, 0, 255)
            cv2.putText(frame, f"Line {i+1}: {count} person", (20, 45 + 35 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, count_color, 3)

        if time.time() - self._queue_last_alarm > 2.0:
            if any(c > 7 for c in line_counts):
                os.system("afplay /System/Library/Sounds/Ping.aiff &")
                
            if time.time() - self._last_queue_photo > 5.0:
                self._last_queue_photo = time.time()
                
                # Format the log string: "line 1=2 , line 2=4"
                log_parts = [f"line {i+1}={c}" for i, c in enumerate(line_counts)]
                log_str = " , ".join(log_parts)
                
                photo_path = f"static/events/{int(time.time())}_queue_{total_people}.jpg"
                cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), frame)
                database.log_queue(log_str, photo_path)
            
        if any(c >= 5 for c in line_counts) and time.time() - self._queue_last_alarm > 10.0:
            alert_str = " , ".join([f"line {i+1}={c}" for i, c in enumerate(line_counts) if c >= 5])
            database.log_event("queue", "long_queue", alert_str, "alert")
            self._queue_last_alarm = time.time()

        return frame

    def _refine_box(self, frame, b):
        x1, y1, x2, y2 = map(int, b)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return b
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return b
        valid = [c for c in contours if cv2.contourArea(c) > 1000]
        if not valid: return b
        largest = max(valid, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(largest)
        if rw * rh < (x2-x1) * (y2-y1) * 0.10: return b
        pad = 10
        return [float(max(x1, x1 + rx - pad)), float(max(y1, y1 + ry - pad)), 
                float(min(x2, x1 + rx + rw + pad)), float(min(y2, y1 + ry + rh + pad))]

    def _process_box(self, frame):
        # conf=0.05 to ensure the custom prototype model detects the box
        results = self._custom_box_model.track(frame, persist=True, tracker="custom_botsort.yaml", conf=0.05, verbose=False)
        
        with self._status_lock:
            if not self._box_line:
                height, width = frame.shape[:2]
                mid_x = width // 2
                line_pts = [
                    [(mid_x - 100) / width, 0.0, (mid_x - 100) / width, 1.0],
                    [(mid_x + 100) / width, 0.0, (mid_x + 100) / width, 1.0]
                ]
            else:
                line_pts = list(self._box_line)
            
        out_frame, events = self._box_counter.process_frame(frame.copy(), results, line_pts)
        
        if events:
            for i, event in enumerate(events):
                event_type = event["type"]
                track_id = event["track_id"]
                cls_name = event.get("class", "unknown")
                
                if i == 0:
                    photo_path = f"static/events/{int(time.time())}_box_{event_type}_{track_id}.jpg"
                    cv2.imwrite(os.path.join(config.BASE_DIR, photo_path), out_frame)
                    database.log_box(
                        self._box_counter.counts['cardboard box']['in'],
                        self._box_counter.counts['cardboard box']['out'],
                        photo_path
                    )
            
        return out_frame

    def _process_room(self, frame):
        self._ensure_models_loaded()
        
        # Use the standard loaded YOLO model with ByteTrack to eliminate flickering
        results = self._yolo.track(frame, classes=[PERSON_CLASS], persist=True, tracker="botsort.yaml", verbose=False)
        
        people_count = 0
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu()
            confidences = results[0].boxes.conf.cpu()
            
            for box, conf in zip(boxes, confidences):
                if conf < 0.25: # Lowered threshold to catch occluded people
                    continue
                    
                people_count += 1
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate chest position (approx 30% down from top of bounding box)
                chest_x = int((x1 + x2) / 2)
                chest_y = int(y1 + (y2 - y1) * 0.3)
                
                # Calculate head position (approx 5% down from top)
                head_x = chest_x
                head_y = int(y1 + (y2 - y1) * 0.05)
                
                # Draw chest point
                cv2.circle(frame, (chest_x, chest_y), 8, (0, 255, 0), -1)
                cv2.circle(frame, (chest_x, chest_y), 4, (255, 255, 255), -1)
                
                # Draw small white point on head
                cv2.circle(frame, (head_x, head_y), 3, (255, 255, 255), -1)

        # Display total count
        cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Active Persons in Room: {people_count}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        return frame
# Single shared instance used by the Flask app
worker = CameraWorker()