"""
Worker Activity Tracker - Standalone Script
============================================
Detects people working at laptops/screens using pretrained models:
  - YOLOv8 for person, laptop, phone detection
  - YuNet (ONNX) for face detection + 5 landmarks (head pose estimation)

GREEN box = WORKING (looking at laptop/screen, face upright, oriented toward screen)
  -> Working timer (top-left of bounding box)
RED box = NOT WORKING (looking away, sleeping, head down, on phone, talking)
  -> Not-Working timer (top-right of bounding box)
Both timers always visible on each person's box.

OpenCV 5.0+ compatible (no CascadeClassifier, no Caffe models).

Usage:
  python worker_tracker.py --video 0                    # webcam
  python worker_tracker.py --video path/to/video.mp4    # video file
  python worker_tracker.py --video path/to/image.jpg    # single image
"""

import os
import sys
import csv
import time
import math
import argparse
import urllib.request

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# YOLO model search paths
YOLO_MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "yolov8s.pt"),
    os.path.join(PROJECT_ROOT, "yolov8n.pt"),
    "yolov8s.pt",
    "yolov8n.pt",
]

# YuNet face detection model (ONNX - works with OpenCV 5+)
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def _download_file(url, filepath):
    """Download file from URL if not already present."""
    if not os.path.exists(filepath):
        print(f"  Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"  Download complete.")


def load_models():
    """Load all pretrained models needed for worker tracking."""
    # 1. YOLO model
    yolo_path = None
    for candidate in YOLO_MODEL_CANDIDATES:
        if os.path.exists(candidate):
            yolo_path = candidate
            break
    if yolo_path is None:
        yolo_path = "yolov8n.pt"  # ultralytics will auto-download
    print(f"[WorkerTracker] Loading YOLO model ({yolo_path})...")
    yolo = YOLO(yolo_path)

    # 2. YuNet face detector (ONNX - OpenCV 5+ compatible)
    #    Provides face bounding box + 5 landmarks for head pose estimation
    print("[WorkerTracker] Loading YuNet face detector...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    yunet_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    _download_file(YUNET_URL, yunet_path)
    face_detector = cv2.FaceDetectorYN.create(
        yunet_path, "", (320, 320),
        score_threshold=0.5, nms_threshold=0.3, top_k=5000
    )

    worker_classifier_path = os.path.join(PROJECT_ROOT, "worker_classifier.pt")
    worker_classifier = None
    if os.path.exists(worker_classifier_path):
        print(f"[WorkerTracker] Loading Worker Classifier ({worker_classifier_path})...")
        worker_classifier = YOLO(worker_classifier_path)

    return yolo, face_detector, worker_classifier


# ---------------------------------------------------------------------------
# NMS helper
# ---------------------------------------------------------------------------
def nms_boxes(boxes, scores, iou_threshold=0.4):
    """Non-Maximum Suppression to eliminate duplicate/overlapping boxes."""
    if not boxes:
        return []
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    x1 = boxes_np[:, 0]; y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]; y2 = boxes_np[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores_np.argsort()[::-1]
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


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------
def fmt_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Head Pose Estimation from YuNet's 5 landmarks
# ---------------------------------------------------------------------------
def estimate_head_pose(landmarks, fy, fh, py1, py2, camera_angle="frontal"):
    """
    Estimate head pose and return a head_score (0.0 to 1.0) along with raw metrics.
    """
    right_eye, left_eye = landmarks[0], landmarks[1]
    nose = landmarks[2]
    right_mouth, left_mouth = landmarks[3], landmarks[4]

    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    if eye_dist < 1.0:
        return 0.0, 0.0, 0.0, 0.5  # Unreliable landmarks

    # Eye center
    eye_cx = (right_eye[0] + left_eye[0]) / 2.0
    eye_cy = (right_eye[1] + left_eye[1]) / 2.0
    # Mouth center
    mouth_cy = (right_mouth[1] + left_mouth[1]) / 2.0

    # YAW
    yaw_ratio = (nose[0] - eye_cx) / eye_dist
    # PITCH
    pitch_ratio = (mouth_cy - eye_cy) / eye_dist
    # POSTURE
    face_center_y = fy + fh / 2.0
    person_h = py2 - py1
    face_rel_y = face_center_y / person_h if person_h > 0 else 0.5

    # Calibrate thresholds based on camera angle
    if camera_angle == "elevated":
        # Elevated: Face appears lower, pitch is heavily downward
        yaw_max = 0.8
        pitch_min = -0.5
        face_y_max = 0.85
    elif camera_angle == "side":
        yaw_max = 1.2
        pitch_min = -0.2
        face_y_max = 0.75
    else:  # frontal
        yaw_max = 0.5
        pitch_min = -0.1
        face_y_max = 0.65

    # Calculate score based on how close to 'perfect' the pose is
    score = 1.0
    if abs(yaw_ratio) > yaw_max:
        score -= 0.5
    if pitch_ratio < pitch_min:
        score -= 0.8  # Heavy penalty for extreme head down (sleeping)
    if face_rel_y > face_y_max:
        score -= 0.5

    head_score = max(0.0, min(1.0, score))
    return head_score, yaw_ratio, pitch_ratio, face_rel_y


# ---------------------------------------------------------------------------
# Video initialization
# ---------------------------------------------------------------------------
def initialize_video(video_source):
    """Open video source. Returns frame/cap/metadata."""
    is_image = isinstance(video_source, str) and video_source.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp")
    )
    if is_image:
        frame = cv2.imread(video_source)
        if frame is None:
            print(f"Error: Could not read image '{video_source}'")
            return None, None, 0, 0, 0, True
        return frame, None, frame.shape[1], frame.shape[0], 30.0, True

    source = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return None, None, 0, 0, 0, False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
    return None, cap, w, h, fps, False


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
def log_worker_event(csv_path, tid, s, new_status, now):
    """Append a status-change event to the CSV log."""
    try:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "worker_id", "status",
                    "working_seconds", "not_working_seconds",
                    "productivity_pct",
                ])
            w_s = s.get("working_s", 0.0)
            nw_s = s.get("not_working_s", 0.0)
            total_s = w_s + nw_s
            pct = (w_s / total_s * 100) if total_s > 0 else 0.0
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                tid, new_status,
                f"{w_s:.1f}", f"{nw_s:.1f}", f"{pct:.1f}",
            ])
    except Exception as e:
        print(f"[WorkerTracker] CSV log error: {e}")


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------
def run(video_source="0", output_path="worker_tracker_output.mp4", camera_angle="frontal", grace_period=8.0, debug=False):
    """Main worker tracking loop."""

    # Load models
    yolo, face_detector, worker_classifier = load_models()

    # Open video
    frame, cap, width, height, fps, is_image = initialize_video(video_source)
    if width == 0:
        print("[WorkerTracker] Failed to open video source.")
        return

    # Video writer
    out = None
    final_output_path = None
    if output_path:
        os.makedirs("outputs", exist_ok=True)
        final_output_path = os.path.join("outputs", os.path.basename(output_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))

    # CSV log path
    csv_path = os.path.join(PROJECT_ROOT, "worker_logs.csv")

    # Tracking state
    worker_states = {}  # tid -> state dict
    pair_close_since = {}  # frozenset({id1, id2}) -> timestamp
    last_laptop_boxes = []
    last_phone_boxes = []
    last_tick = None
    frame_count = 0

    print("[WorkerTracker] Starting capture loop. Press 'q' to exit.")

    try:
        while True:
            now = time.time()
            dt = 0.0 if last_tick is None else max(0.0, now - last_tick)
            last_tick = now
            frame_count += 1

            # Read frame
            if not is_image:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WorkerTracker] Video feed ended.")
                    break
            else:
                frame = frame.copy()

            height, width = frame.shape[:2]

            # ---- Detection (every 2nd frame for performance) ----
            if frame_count % 2 == 1:
                # YOLO classes: person=0, laptop=62, tv=63, mouse=64, keyboard=66, cell_phone=67
                results = yolo(frame, classes=[0, 62, 63, 64, 66, 67], conf=0.05, verbose=False)
                r = results[0]

                person_candidates, person_scores = [], []
                laptop_candidates, laptop_scores, laptop_names = [], [], []
                phone_candidates, phone_scores = [], []

                if r.boxes is not None:
                    for box, conf, cls in zip(
                        r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu()
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        cls_name = yolo.names[int(cls)]
                        c = float(conf)

                        if cls_name == "person" and c >= 0.20:
                            person_candidates.append([x1, y1, x2, y2])
                            person_scores.append(c)
                        elif cls_name in ("laptop", "tv", "mouse", "keyboard") and c >= 0.10:
                            laptop_candidates.append([x1, y1, x2, y2])
                            laptop_scores.append(c)
                            laptop_names.append(cls_name)
                        elif cls_name == "cell phone" and c >= 0.05:
                            phone_candidates.append([x1, y1, x2, y2])
                            phone_scores.append(c)

                # NMS
                keep_p = nms_boxes(person_candidates, person_scores, 0.4)
                person_raw = [(person_candidates[i], person_scores[i]) for i in keep_p]

                keep_l = nms_boxes(laptop_candidates, laptop_scores, 0.3)
                laptop_boxes = [(laptop_candidates[i], laptop_scores[i], laptop_names[i]) for i in keep_l]

                keep_ph = nms_boxes(phone_candidates, phone_scores, 0.3)
                phone_boxes = [(phone_candidates[i], phone_scores[i]) for i in keep_ph]

                # --- Centroid-based tracking ---
                person_dets = []
                next_id = 0
                if worker_states:
                    next_id = max(worker_states.keys()) + 1

                used_tids = set()
                for p_box, p_conf in person_raw:
                    x1, y1, x2, y2 = p_box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    best_tid, min_dist = None, 200
                    for tid, st in worker_states.items():
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

                # --- Activity classification per person ---
                seen_ids = set()
                for tid, p_box, p_conf in person_dets:
                    seen_ids.add(tid)
                    x1, y1, x2, y2 = p_box
                    px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
                    px2, py2 = min(width, x2 + 20), min(height, y2 + 20)
                    roi = frame[py1:py2, px1:px2]

                    # --- YuNet Face Detection + Head Pose ---
                    face_box_abs = None
                    head_score, yaw_ratio, pitch_ratio, face_rel_y = 0.0, 0.0, 0.0, 0.5
                    if roi.size > 0:
                        try:
                            rh, rw = roi.shape[:2]
                            if rw > 30 and rh > 30:
                                face_detector.setInputSize((rw, rh))
                                _, detected_faces = face_detector.detect(roi)
                                if detected_faces is not None and len(detected_faces) > 0:
                                    # Pick largest face
                                    best_face = max(detected_faces, key=lambda f: f[2] * f[3])
                                    fx = int(best_face[0])
                                    fy_f = int(best_face[1])
                                    fw = int(best_face[2])
                                    fh = int(best_face[3])
                                    face_box_abs = [px1 + fx, py1 + fy_f, px1 + fx + fw, py1 + fy_f + fh]

                                    # Head pose from 5 landmarks
                                    landmarks = best_face[4:14].reshape(5, 2)
                                    head_score, yaw_ratio, pitch_ratio, face_rel_y = estimate_head_pose(
                                        landmarks, fy_f, fh, 0, rh, camera_angle=camera_angle
                                    )
                        except Exception:
                            pass

                    # Phone near face/body
                    phone_near = False
                    for pb, _pconf in phone_boxes:
                        ref_box = face_box_abs if face_box_abs is not None else p_box
                        rx1, ry1, rx2, ry2 = ref_box
                        margin = 40 if face_box_abs is not None else 15
                        if not (
                            pb[2] < rx1 - margin
                            or pb[0] > rx2 + margin
                            or pb[3] < ry1 - margin
                            or pb[1] > ry2 + margin
                        ):
                            phone_near = True
                            break

                    # Standing check
                    is_standing = False
                    p_w = x2 - x1
                    p_h = y2 - y1
                    if p_w > 0 and (p_h / float(p_w)) > 1.5:
                        is_standing = True

                    # Laptop proximity
                    has_laptop = False
                    if laptop_boxes:
                        pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
                        for lb, _lconf, _lname in laptop_boxes:
                            lcx, lcy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
                            if math.hypot(pcx - lcx, pcy - lcy) < (x2 - x1) * 3.5:
                                has_laptop = True
                                break

                    # Custom Classifier
                    ai_working = False
                    if worker_classifier is not None and roi.size > 0:
                        try:
                            cls_res = worker_classifier(roi, verbose=False)[0]
                            top1_name = cls_res.names[cls_res.probs.top1]
                            ai_working = (top1_name == 'working')
                        except Exception:
                            ai_working = False

                    # Update or create state
                    st = worker_states.get(tid)
                    if st is None:
                        st = {
                            "working_s": 0.0, "not_working_s": 0.0,
                            "work_s": 0.0, "idle_s": 0.0, "phone_s": 0.0, "talk_s": 0.0,
                            "looking_away_s": 0.0,
                            "status": "Idle",
                            "streak": 0,
                            "box": p_box, "conf": p_conf,
                            "lost_frames": 0, "raw_working": False,
                            "phone_near": phone_near, "has_laptop": has_laptop,
                            "is_standing": is_standing,
                            "score": 0, "head_metrics": (0.0, 0.0, 0.0, 0.5)
                        }
                        worker_states[tid] = st
                    else:
                        st["box"], st["conf"], st["lost_frames"] = p_box, p_conf, 0
                        st["phone_near"] = phone_near
                        st["has_laptop"] = has_laptop
                        st["is_standing"] = is_standing
                        
                    # Strict Working Logic
                    is_working_now = False
                    score = 0
                    if not is_standing and has_laptop and not phone_near:
                        # Evaluate posture
                        if ai_working: score += 50
                        score += head_score * 50
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
                        st["raw_working"] = (st["looking_away_s"] <= grace_period)

                    st["score"] = score
                    st["head_metrics"] = (head_score, yaw_ratio, pitch_ratio, face_rel_y)

                # --- Talking detection (two non-working people close for >5s) ---
                visible = [
                    (tid, s)
                    for tid, s in worker_states.items()
                    if s.get("lost_frames", 0) == 0
                ]
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
                            first_seen = pair_close_since.setdefault(pair, now)
                            if now - first_seen > 5.0:
                                talking_ids.add(tid_a)
                                talking_ids.add(tid_b)
                        else:
                            pair_close_since.pop(pair, None)

                # --- Status resolution with hysteresis ---
                for tid, s in worker_states.items():
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
                        log_worker_event(csv_path, tid, s, target, now)
                        s["status"] = target

                # --- Cleanup lost tracks ---
                for tid in list(worker_states.keys()):
                    s = worker_states[tid]
                    if tid not in seen_ids:
                        s["lost_frames"] = s.get("lost_frames", 0) + 1
                        if s["lost_frames"] > 15:
                            del worker_states[tid]

                last_laptop_boxes = laptop_boxes
                last_phone_boxes = phone_boxes

            # ---- Timer accumulation (every frame) ----
            for tid, s in worker_states.items():
                if s.get("lost_frames", 0) > 0:
                    continue
                status = s.get("status", "Idle")
                # Binary working / not-working (mutually exclusive)
                if status == "Working":
                    s["working_s"] = s.get("working_s", 0.0) + dt
                else:
                    s["not_working_s"] = s.get("not_working_s", 0.0) + dt

            # ================ DRAWING ================
            # Draw laptop boxes (subtle)
            for lb, lconf, lname in last_laptop_boxes:
                lx1, ly1, lx2, ly2 = lb
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 80, 0), 1)
                cv2.putText(
                    frame, lname, (lx1, ly1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1,
                )

            # Draw phone boxes (subtle)
            for pb, pconf in last_phone_boxes:
                phx1, phy1, phx2, phy2 = pb
                cv2.rectangle(frame, (phx1, phy1), (phx2, phy2), (180, 105, 255), 1)
                cv2.putText(
                    frame, "phone", (phx1, phy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 105, 255), 1,
                )

            COLOR_GREEN = (0, 255, 0)
            COLOR_RED = (0, 0, 255)

            working_count = 0
            not_working_count = 0

            for tid, s in worker_states.items():
                if s.get("lost_frames", 0) > 0:
                    continue

                bx1, by1, bx2, by2 = s["box"]
                status = s.get("status", "Idle")
                is_working = status == "Working"
                color = COLOR_GREEN if is_working else COLOR_RED
                status_label = "WORKING" if is_working else "NOT WORKING"

                if is_working:
                    working_count += 1
                else:
                    not_working_count += 1

                # Bounding box with padding
                px1, py1 = max(0, bx1 - 12), max(0, by1 - 12)
                px2, py2 = min(width, bx2 + 12), min(height, by2 + 12)
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
                
                if debug:
                    score = s.get("score", 0)
                    h_score, y_r, p_r, f_y = s.get("head_metrics", (0.0, 0.0, 0.0, 0.5))
                    debug_texts = [
                        f"Score: {score:.1f}",
                        f"Head: {h_score:.2f}",
                        f"Yaw: {y_r:.2f}",
                        f"Pitch: {p_r:.2f}",
                        f"FaceY: {f_y:.2f}"
                    ]
                    dy = py1 + 15
                    for d_txt in debug_texts:
                        cv2.putText(frame, d_txt, (px2 + 5, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        dy += 15

                # --- Dual timer labels ---
                w_time = fmt_time(s.get("working_s", 0.0))
                nw_time = fmt_time(s.get("not_working_s", 0.0))

                w_label = f"W: {w_time}"
                nw_label = f"NW: {nw_time}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thick = 1

                (w_tw, w_th), _ = cv2.getTextSize(w_label, font, font_scale, thick)
                (nw_tw, nw_th), _ = cv2.getTextSize(nw_label, font, font_scale, thick)

                label_h = max(w_th, nw_th) + 10
                label_y_top = max(0, py1 - label_h - 2)

                # Working timer - top-left (dark green background)
                w_bg_x1 = px1
                w_bg_x2 = px1 + w_tw + 8
                overlay = frame.copy()
                cv2.rectangle(overlay, (w_bg_x1, label_y_top), (w_bg_x2, label_y_top + label_h), (0, 80, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.rectangle(frame, (w_bg_x1, label_y_top), (w_bg_x2, label_y_top + label_h), COLOR_GREEN, 1)
                cv2.putText(
                    frame, w_label, (w_bg_x1 + 4, label_y_top + label_h - 4),
                    font, font_scale, (0, 255, 0), thick, cv2.LINE_AA,
                )

                # Not-Working timer - top-right (dark red background)
                nw_bg_x2 = px2
                nw_bg_x1 = px2 - nw_tw - 8
                overlay = frame.copy()
                cv2.rectangle(overlay, (nw_bg_x1, label_y_top), (nw_bg_x2, label_y_top + label_h), (0, 0, 80), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.rectangle(frame, (nw_bg_x1, label_y_top), (nw_bg_x2, label_y_top + label_h), COLOR_RED, 1)
                cv2.putText(
                    frame, nw_label, (nw_bg_x1 + 4, label_y_top + label_h - 4),
                    font, font_scale, (0, 0, 255), thick, cv2.LINE_AA,
                )

                # --- Bottom label: Worker ID + Status ---
                conf_pct = s.get("conf", 0) * 100
                bottom_label = f"#{tid} | {status_label} {conf_pct:.0f}%"
                (bt_w, bt_h), _ = cv2.getTextSize(bottom_label, font, font_scale, thick)

                bt_y = min(height - 2, py2 + bt_h + 8)
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, (px1, bt_y - bt_h - 4), (px1 + bt_w + 8, bt_y + 2), (15, 15, 15), -1,
                )
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                cv2.rectangle(frame, (px1, bt_y - bt_h - 4), (px1 + bt_w + 8, bt_y + 2), color, 1)
                cv2.putText(
                    frame, bottom_label, (px1 + 4, bt_y - 2),
                    font, font_scale, (255, 255, 255), thick, cv2.LINE_AA,
                )

            # --- Top HUD bar ---
            total = working_count + not_working_count
            productivity = (working_count / total * 100) if total > 0 else 0.0
            hud_h = 50
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, hud_h), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (0, 0), (width, hud_h), (60, 60, 60), 1)

            summary_text = (
                f"Workers: {total}   Working: {working_count}   "
                f"Not Working: {not_working_count}   Productivity: {productivity:.0f}%"
            )
            cv2.putText(
                frame, summary_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Per-worker timer summary in HUD
            hud_x = 10
            for tid, s in sorted(worker_states.items()):
                if s.get("lost_frames", 0) > 0:
                    continue
                w_str = fmt_time(s.get("working_s", 0.0))
                nw_str = fmt_time(s.get("not_working_s", 0.0))
                worker_summary = f"#{tid} W:{w_str} NW:{nw_str}"
                cv2.putText(
                    frame, worker_summary, (hud_x, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
                )
                (sw, _), _ = cv2.getTextSize(worker_summary, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                hud_x += sw + 20

            # ---- Write output / display ----
            if out is not None:
                out.write(frame)

            display_frame = frame
            if width > 1000:
                display_frame = cv2.resize(frame, (1000, int(1000 * height / width)))

            try:
                cv2.imshow("Worker Activity Tracker", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass

            if is_image:
                img_out = (
                    final_output_path.replace(".mp4", ".jpg")
                    if final_output_path
                    else "worker_tracker_output.jpg"
                )
                cv2.imwrite(img_out, frame)
                print(f"[WorkerTracker] Processed image saved to: {img_out}")
                break

    except KeyboardInterrupt:
        print("[WorkerTracker] Interrupted by user.")
    finally:
        # ---- Print Summary ----
        print("\n" + "=" * 60)
        print("             WORKER TRACKING SUMMARY")
        print("=" * 60)
        for tid, s in sorted(worker_states.items()):
            w_sec = s.get("working_s", 0.0)
            nw_sec = s.get("not_working_s", 0.0)
            total_sec = w_sec + nw_sec
            w_pct = (w_sec / total_sec * 100) if total_sec > 0 else 0.0
            nw_pct = 100.0 - w_pct if total_sec > 0 else 0.0
            print(f"Worker #{tid}:")
            print(f"  - WORKING:     {fmt_time(w_sec)} ({w_pct:.1f}%)")
            print(f"  - NOT WORKING: {fmt_time(nw_sec)} ({nw_pct:.1f}%)")
            print("-" * 40)
        print("=" * 60 + "\n")

        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[WorkerTracker] Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Activity Tracker")
    parser.add_argument("--video", type=str, default="0", help="Path to input video file (.mp4), image (.jpg/.png), or camera index (0)")
    parser.add_argument("--output", type=str, default="worker_tracker_output.mp4", help="Path to save the output video")
    parser.add_argument("--camera_angle", type=str, default="frontal", choices=["frontal", "elevated", "side"], help="Camera angle for pose calibration")
    parser.add_argument("--grace_period", type=float, default=8.0, help="Seconds a worker can look away before marked not working")
    parser.add_argument("--debug", action="store_true", help="Show scoring metrics overlay")
    args = parser.parse_args()
    run(args.video, args.output, args.camera_angle, args.grace_period, args.debug)