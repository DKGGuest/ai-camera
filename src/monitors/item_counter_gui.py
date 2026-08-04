import cv2
import numpy as np
import time
import sys
import os
import argparse
import math
import threading

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.models import load_yolo_model
from src.core.video import initialize_video_capture


class ThreadedCapture:
    """Reads frames in a background thread so inference never waits on I/O."""
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
            if not ret:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class ItemCounterGUI:
    def __init__(self, video_source='0', model_path='yolov8n.pt', track_buffer=120,
                 target_class='box,sugar_sack', window_name="Item Counting App",
                 imgsz=480, tracker='bytetrack.yaml', skip_frames=0):
        self.video_source = video_source
        self.window_name = window_name
        self.target_classes = [c.strip() for c in target_class.split(',') if c.strip()]
        self.imgsz = imgsz
        self.skip_frames = skip_frames  # 0 = run detection every frame, 1 = every 2nd frame, etc.

        # State
        self.counts = {cls: {"in": 0, "out": 0} for cls in self.target_classes}
        self.status = "Status: Initializing..."
        self.is_running = True

        # Tracking setup
        print(f"Loading YOLO modecardboard boxl: {model_path} targeting '{self.target_classes}'")
        self.model = load_yolo_model(model_path)

        self.tracker_yaml = tracker

        # Crossing state
        self._tracks = {}
        self.buffer_px = 30
        self._frame_counter = 0
        self._last_boxes = []

    def run(self):
        print(f"Connecting to video source '{self.video_source}'...")
        self.status = "Status: Connecting..."

        frame, cap, orig_width, orig_height, fps, is_image = initialize_video_capture(self.video_source)
        if orig_width == 0 or is_image:
            print("Error: Item Counter requires a valid video stream.")
            return

        cap = ThreadedCapture(cap)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 540)

        self.status = "Status: Running"

        self.buffer_px = int(orig_width * 0.15)
        line_x = orig_width // 2
        left_bound = line_x - self.buffer_px
        right_bound = line_x + self.buffer_px

        line1 = [(left_bound, 0), (left_bound, orig_height)]
        line2 = [(right_bound, 0), (right_bound, orig_height)]

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        while self.is_running:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.status = "Status: Error (Stream Disconnected). Attempting reconnect..."
                    error_frame = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
                    cv2.putText(error_frame, self.status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(self.window_name, error_frame)
                    cv2.waitKey(1000)
                    time.sleep(0.5)
                    continue

                self.status = "Status: Running"
                now = time.time()

                self._frame_counter += 1
                run_detection = (self.skip_frames <= 0) or (self._frame_counter % (self.skip_frames + 1) == 0)

                if run_detection:
                    results = self.model.track(
                        frame, persist=True, verbose=False, conf=0.35,
                        tracker=self.tracker_yaml, imgsz=self.imgsz
                    )
                    self._last_boxes = []

                    for r in results:
                        boxes = r.boxes
                        if boxes is None or boxes.id is None:
                            continue

                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            cls_name = self.model.names[cls_id]
                            conf = float(box.conf[0])

                            if cls_name not in self.target_classes:
                                continue

                            track_id = int(box.id[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            self._last_boxes.append((track_id, cls_name, conf, x1, y1, x2, y2))

                            if track_id not in self._tracks:
                                self._tracks[track_id] = {
                                    "history": [(cx, cy, now)],
                                    "crossings": [],
                                    "last_cross_time": 0,
                                    "last_counted_time": 0,
                                    "last_update": now,
                                    "first_seen": now
                                }
                            else:
                                t = self._tracks[track_id]
                                t["last_update"] = now
                                t["history"].append((cx, cy, now))
                                if len(t["history"]) > 30:
                                    t["history"].pop(0)

                                prev_pt = (t["history"][-2][0], t["history"][-2][1]) if len(t["history"]) > 1 else (cx, cy)
                                curr_pt = (cx, cy)

                                if now - t.get("last_counted_time", 0) > 2.0:
                                    crossed_lines = []
                                    if intersect(prev_pt, curr_pt, line1[0], line1[1]):
                                        crossed_lines.append(1)
                                    if intersect(prev_pt, curr_pt, line2[0], line2[1]):
                                        crossed_lines.append(2)

                                    if len(crossed_lines) == 2:
                                        dist1 = abs(prev_pt[0] - line1[0][0])
                                        dist2 = abs(prev_pt[0] - line2[0][0])
                                        if dist1 < dist2:
                                            crossed_lines = [1, 2]
                                        else:
                                            crossed_lines = [2, 1]

                                    for crossed in crossed_lines:
                                        if t.get("last_cross_time", 0) and now - t["last_cross_time"] > 5.0:
                                            t["crossings"] = []
                                        if not t["crossings"] or t["crossings"][-1] != crossed:
                                            t["crossings"].append(crossed)
                                            t["last_cross_time"] = now

                                    if len(t["crossings"]) >= 2:
                                        seq = t["crossings"][-2:]
                                        if seq == [1, 2] or seq == [2, 1]:
                                            oldest_pt = t["history"][0]
                                            displacement = math.hypot(cx - oldest_pt[0], cy - oldest_pt[1])
                                            min_dist = 0.3 * (x2 - x1)

                                            if displacement >= min_dist:
                                                if seq == [1, 2]:
                                                    self.counts[cls_name]["in"] += 1
                                                    t["last_counted_time"] = now
                                                    t["crossings"] = []
                                                elif seq == [2, 1]:
                                                    self.counts[cls_name]["out"] += 1
                                                    t["last_counted_time"] = now
                                                    t["crossings"] = []
                                            else:
                                                t["crossings"] = []

                                if now - t["first_seen"] > 10.0:
                                    oldest_pt = t["history"][0]
                                    displacement = math.hypot(cx - oldest_pt[0], cy - oldest_pt[1])
                                    lx1, lx2 = line1[0][0], line2[0][0]
                                    min_x, max_x = min(lx1, lx2) - 50, max(lx1, lx2) + 50
                                    if min_x < cx < max_x and displacement < (x2 - x1):
                                        self._stationary_warning_until = now + 5.0

                    to_delete = [tid for tid, t in self._tracks.items() if now - t["last_update"] > 5.0]
                    for tid in to_delete:
                        del self._tracks[tid]

                for track_id, cls_name, conf, x1, y1, x2, y2 in self._last_boxes:
                    label = f"#{track_id} {cls_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.line(frame, (line_x, 0), (line_x, orig_height), (255, 0, 0), 2)
                cv2.line(frame, (left_bound, 0), (left_bound, orig_height), (0, 255, 0), 2)
                cv2.line(frame, (right_bound, 0), (right_bound, orig_height), (0, 255, 255), 2)
                cv2.putText(frame, "L1 (OUTSIDE)", (left_bound + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "L2 (INSIDE)", (right_bound + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                y_offset = 60
                for cls in self.target_classes:
                    count_text = f"{cls} in: {self.counts[cls]['in']} | out: {self.counts[cls]['out']}"
                    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (10, y_offset - th - 10), (10 + tw + 10, y_offset + 5), (0, 0, 0), -1)
                    cv2.putText(frame, count_text, (15, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    y_offset += 40

                if getattr(self, "_stationary_warning_until", 0) > now:
                    warn_text = "WARNING: Lines may be placed in a seating/work area!"
                    cv2.rectangle(frame, (10, orig_height - 60), (orig_width - 10, orig_height - 20), (0, 0, 255), -1)
                    cv2.putText(frame, warn_text, (20, orig_height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                elif key == ord('r'):
                    for cls in self.target_classes:
                        self.counts[cls]["in"] = 0
                        self.counts[cls]["out"] = 0
                    self.status = "Status: Counters Reset"

            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(1)

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"App closed. Final Counts -> {self.counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Item Counter GUI")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--track_buffer', type=int, default=120, help='Frames to keep a lost track alive')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to model: .pt file or OpenVINO folder')
    parser.add_argument('--target_class', type=str, default='box,sugar_sack', help='Comma-separated class names to track')
    parser.add_argument('--imgsz', type=int, default=480, help='Inference resolution')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help="'bytetrack.yaml' or 'botsort.yaml'")
    parser.add_argument('--skip_frames', type=int, default=0, help='Run detection every Nth frame')
    args = parser.parse_args()

    app = ItemCounterGUI(
        video_source=args.video,
        model_path=args.model,
        track_buffer=args.track_buffer,
        target_class=args.target_class,
        imgsz=args.imgsz,
        tracker=args.tracker,
        skip_frames=args.skip_frames
    )
    app.run()


