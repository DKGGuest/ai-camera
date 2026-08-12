import cv2
import time
import math
from collections import defaultdict

class BoxCounter:
    def __init__(self):
        self.counts = defaultdict(lambda: {"in": 0, "out": 0})
        self.tracks = {}
        self.current_day = time.localtime().tm_yday
        self.target_classes = ["cardboard box"]
        self.id_mapping = {}
        self.track_history = {}


        
    def process_frame(self, frame, results, line_points):
        h, w = frame.shape[:2]
        events = []
        now = time.time()
        today = time.localtime(now).tm_yday
        
        if today != self.current_day:
            self.counts = defaultdict(lambda: {"in": 0, "out": 0})
            self.current_day = today
            
        # Parse line_points (normalized -> absolute)
        if line_points and len(line_points) == 2 and len(line_points[0]) == 4:
            line1 = [
                (int(line_points[0][0] * w), int(line_points[0][1] * h)),
                (int(line_points[0][2] * w), int(line_points[0][3] * h))
            ]
            line2 = [
                (int(line_points[1][0] * w), int(line_points[1][1] * h)),
                (int(line_points[1][2] * w), int(line_points[1][3] * h))
            ]
        else:
            base_x1, base_x2 = w // 2, w // 2
            base_y1, base_y2 = 0, h
            buffer_px = int(w * 0.15)
            line1 = [(base_x1 - buffer_px, base_y1), (base_x2 - buffer_px, base_y2)]
            line2 = [(base_x1 + buffer_px, base_y1), (base_x2 + buffer_px, base_y2)]
        
        cv2.line(frame, line1[0], line1[1], (0, 255, 0), 2)
        cv2.line(frame, line2[0], line2[1], (0, 255, 255), 2)
        cv2.putText(frame, "L1 (OUTSIDE)", (line1[0][0] + 10, max(30, line1[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "L2 (INSIDE)", (line2[0][0] + 10, max(30, line2[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            names = results[0].names
            
            # ids might be None if the tracker hasn't assigned them yet
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
            else:
                ids = [None] * len(boxes)
            
            for box, raw_track_id, cls_id, conf in zip(boxes, ids, classes, confs):
                cls_name = names[cls_id]
                if cls_name not in self.target_classes:
                    continue

                if conf <= 0.01:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # --- FALLBACK BOX RE-LINKING ---
                if raw_track_id is not None:
                    if raw_track_id not in self.id_mapping and raw_track_id not in self.track_history:
                        best_old_id = None
                        best_dist = 400.0  # Allow large jumps for fast movement / rotations
                        
                        for old_id, info in self.track_history.items():
                            time_since_lost = now - info['time']
                            if 0 < time_since_lost < 2.0:
                                dist = math.hypot(cx - info['cx'], cy - info['cy'])
                                if dist < best_dist:
                                    best_dist = dist
                                    best_old_id = old_id
                                    
                        if best_old_id is not None:
                            self.id_mapping[raw_track_id] = best_old_id
                    
                    track_id = self.id_mapping.get(raw_track_id, raw_track_id)
                else:
                    # Centroid fallback for completely untracked boxes (when tracker fails to assign any ID)
                    best_old_id = None
                    best_dist = 150.0
                    for old_id, info in self.track_history.items():
                        time_since_lost = now - info['time']
                        if 0 < time_since_lost < 1.0:
                            dist = math.hypot(cx - info['cx'], cy - info['cy'])
                            if dist < best_dist:
                                best_dist = dist
                                best_old_id = old_id
                    if best_old_id is not None:
                        track_id = best_old_id
                    else:
                        # Generate a temporary negative ID to not conflict with tracker IDs
                        track_id = -int(time.time() * 1000) % 1000000

                self.track_history[track_id] = {'time': now, 'cx': cx, 'cy': cy}

                if conf <= 0.30:
                    display_name = "untracked box"
                    color = (0, 165, 255)  # Orange for untracked
                else:
                    display_name = "cardboard box"
                    color = (0, 0, 255)  # Red for high confidence cardboard boxes
                
                # Draw the bounding box and label regardless of whether it's tracked or not
                id_text = f"#{track_id}" if track_id is not None else "#UNTRACKED"
                label = f"{id_text} {display_name} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Only perform counting logic if we actually have a tracking ID
                if track_id is not None:
                    if track_id not in self.tracks:
                        self.tracks[track_id] = {
                            "history": [(cx, cy, now)],
                            "crossings": [],
                            "last_cross_time": 0,
                            "last_counted_time": 0,
                            "last_update": now,
                            "first_seen": now
                        }
                    else:
                        t = self.tracks[track_id]
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
                                            events.append({"type": "in", "track_id": track_id, "class": cls_name})
                                            t["last_counted_time"] = now
                                            t["crossings"] = []
                                        elif seq == [2, 1]:
                                            self.counts[cls_name]["out"] += 1
                                            events.append({"type": "out", "track_id": track_id, "class": cls_name})
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

        to_delete = []
        for tid, t in self.tracks.items():
            if now - t["last_update"] > 5.0:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
            
        y_offset = 60
        for cls in self.target_classes:
            count_text = f"{cls} loading: {self.counts[cls]['in']} | unloading: {self.counts[cls]['out']}"
            (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (10, y_offset - th - 10), (10 + tw + 10, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(frame, count_text, (15, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 40

        if getattr(self, "_stationary_warning_until", 0) > now:
            warn_text = "WARNING: Lines may be placed in a seating/work area!"
            cv2.rectangle(frame, (10, h - 60), (w - 10, h - 20), (0, 0, 255), -1)
            cv2.putText(frame, warn_text, (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return frame, events
