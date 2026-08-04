import cv2
import numpy as np
import time
import sys
import os
import argparse
import math

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.models import load_yolo_model
from src.core.video import initialize_video_capture

class PeopleCounterGUI:
    def __init__(self, video_source='0', model_path='yolov8n.pt', track_buffer=120, target_class='person', window_name="People Counting App"):
        self.video_source = video_source
        self.window_name = window_name
        self.target_class = target_class
        self.canvas_width = 800
        
        # UI Dimensions
        self.banner_height = 60
        self.panel_height = 120
        self.status_height = 40
        
        # State
        self.entries_in = 0
        self.exits_out = 0
        self.status = "Status: Initializing..."
        self.is_running = True
        
        # Tracking setup
        print(f"Loading YOLO model: {model_path} targeting '{self.target_class}'")
        self.model = load_yolo_model(model_path)
        
        # Tracker config
        self.tracker_yaml = "custom_gui_botsort.yaml"
        with open(self.tracker_yaml, "w") as f:
            f.write(f"""tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: {track_buffer}
match_thresh: 0.8
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: True
model: osnet_x0_25_msmt17.pt
fuse_score: True
""")
            
        # Crossing state
        self._people_tracks = {}
        self.buffer_px = 30 # pixels around the vertical line

    def create_ui(self, video_frame):
        """Composes the full UI frame with the video feed and UI panels."""
        h, w = video_frame.shape[:2]
        scale = self.canvas_width / w
        video_height = int(h * scale)
        video_resized = cv2.resize(video_frame, (self.canvas_width, video_height))
        
        canvas_height = self.banner_height + video_height + self.panel_height + self.status_height
        canvas = np.zeros((canvas_height, self.canvas_width, 3), dtype=np.uint8)
        
        # 1. Draw Top Banner (Green)
        cv2.rectangle(canvas, (0, 0), (self.canvas_width, self.banner_height), (0, 150, 0), -1)
        text = "PEOPLE COUNTING"
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
        text_x = (self.canvas_width - text_size[0]) // 2
        text_y = (self.banner_height + text_size[1]) // 2
        cv2.putText(canvas, text, (text_x, text_y), font, 1.2, (255, 255, 255), 2)
        
        # 2. Place Video Feed
        y_offset = self.banner_height
        canvas[y_offset:y_offset+video_height, 0:self.canvas_width] = video_resized
        
        # 3. Draw Counter Panels
        panel_y = y_offset + video_height
        half_w = self.canvas_width // 2
        
        # LEFT panel (Red background, "OUT")
        cv2.rectangle(canvas, (0, panel_y), (half_w, panel_y + self.panel_height), (0, 0, 180), -1)
        # RIGHT panel (Green background, "IN")
        cv2.rectangle(canvas, (half_w, panel_y), (self.canvas_width, panel_y + self.panel_height), (0, 150, 0), -1)
        
        # Add text to LEFT panel
        cv2.putText(canvas, "OUT", (20, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_text = f"{self.exits_out:04d}"
        cv2.putText(canvas, out_text, (20, panel_y + 100), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 4)
        
        # Add text to RIGHT panel
        cv2.putText(canvas, "IN", (half_w + 20, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        in_text = f"{self.entries_in:04d}"
        cv2.putText(canvas, in_text, (half_w + 20, panel_y + 100), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 4)
        
        # 4. Draw Status Line
        status_y = panel_y + self.panel_height
        cv2.rectangle(canvas, (0, status_y), (self.canvas_width, canvas_height), (50, 50, 50), -1)
        
        # Check if error status
        status_color = (100, 100, 255) if "Error" in self.status else (220, 220, 220)
        cv2.putText(canvas, self.status, (15, status_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 1)
        cv2.putText(canvas, "Press 'r' to reset | 'q' to quit", (self.canvas_width - 320, status_y + 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
        return canvas, scale, y_offset

    def run(self):
        print(f"Connecting to video source '{self.video_source}'...")
        self.status = "Status: Connecting..."
        
        frame, cap, orig_width, orig_height, fps, is_image = initialize_video_capture(self.video_source)
        if orig_width == 0 or is_image:
            print("Error: People Counter requires a valid video stream.")
            return

        self.status = "Status: Running"
        
        # Setup vertical counting lines
        self.buffer_px = int(orig_width * 0.15)
        line_x = orig_width // 2
        left_bound = line_x - self.buffer_px
        right_bound = line_x + self.buffer_px
        
        line1 = [(left_bound, 0), (left_bound, orig_height)]
        line2 = [(right_bound, 0), (right_bound, orig_height)]
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        while self.is_running:
            try:
                ret, frame = cap.read()
                if not ret:
                    self.status = "Status: Error (Stream Disconnected). Attempting reconnect..."
                    # Show error state
                    ui_frame, _, _ = self.create_ui(np.zeros((orig_height, orig_width, 3), dtype=np.uint8))
                    cv2.imshow(self.window_name, ui_frame)
                    cv2.waitKey(1000)
                    # Attempt reconnect
                    _, cap, _, _, _, _ = initialize_video_capture(self.video_source)
                    continue
                
                self.status = "Status: Running"
                now = time.time()
                
                # YOLOv8 Tracking
                results = self.model.track(frame, persist=True, verbose=False, conf=0.35, tracker=self.tracker_yaml)
                
                for r in results:
                    boxes = r.boxes
                    if boxes is None or boxes.id is None:
                        continue
                        
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id]
                        
                        if cls_name != self.target_class:
                            continue
                            
                        track_id = int(box.id[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        # -- Counting Logic --
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
                                                self.entries_in += 1
                                                t["last_counted_time"] = now
                                                t["crossings"] = []
                                            elif seq == [2, 1]:
                                                self.exits_out += 1
                                                t["last_counted_time"] = now
                                                t["crossings"] = []
                                        else:
                                            t["crossings"] = [] # Reject due to jitter
                                            
                            # Stationary check
                            if now - t["first_seen"] > 10.0:
                                oldest_pt = t["history"][0]
                                displacement = math.hypot(cx - oldest_pt[0], cy - oldest_pt[1])
                                lx1, lx2 = line1[0][0], line2[0][0]
                                min_x, max_x = min(lx1, lx2) - 50, max(lx1, lx2) + 50
                                if min_x < cx < max_x and displacement < (x2 - x1):
                                    self._stationary_warning_until = now + 5.0
                        
                        # Draw minimal tracking overlay on the original frame
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID: {track_id}", (cx + 10, cy - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    
                # Cleanup old tracks
                to_delete = []
                for tid, t in self._people_tracks.items():
                    if now - t["last_update"] > 5.0:
                        to_delete.append(tid)
                for tid in to_delete:
                    del self._people_tracks[tid]
                
                # Draw vertical counting lines on the original frame
                cv2.line(frame, (line_x, 0), (line_x, orig_height), (255, 0, 0), 2) # Blue center line
                cv2.line(frame, (left_bound, 0), (left_bound, orig_height), (0, 255, 0), 2) # L1 Outside (Green)
                cv2.line(frame, (right_bound, 0), (right_bound, orig_height), (0, 255, 255), 2) # L2 Inside (Yellow)
                cv2.putText(frame, "L1 (OUTSIDE)", (left_bound + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "L2 (INSIDE)", (right_bound + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if getattr(self, "_stationary_warning_until", 0) > now:
                    warn_text = "WARNING: Lines may be placed in a seating/work area!"
                    cv2.rectangle(frame, (10, orig_height - 60), (orig_width - 10, orig_height - 20), (0, 0, 255), -1)
                    cv2.putText(frame, warn_text, (20, orig_height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Compose final UI
                ui_frame, _, _ = self.create_ui(frame)
                cv2.imshow(self.window_name, ui_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                elif key == ord('r'):
                    self.entries_in = 0
                    self.exits_out = 0
                    self.status = "Status: Counters Reset"
                    
            except Exception as e:
                self.status = f"Status: Error - {str(e)}"
                time.sleep(1)

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"App closed. Final Count -> IN: {self.entries_in} | OUT: {self.exits_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone People Counter GUI")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--track_buffer', type=int, default=120, help='Frames to keep a lost track alive')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to custom YOLO model (e.g., best.pt)')
    parser.add_argument('--target_class', type=str, default='person', help='Class name to track (e.g., box)')
    args = parser.parse_args()
    
    app = PeopleCounterGUI(
        video_source=args.video, 
        model_path=args.model, 
        track_buffer=args.track_buffer,
        target_class=args.target_class
    )
    app.run()
