import cv2
import numpy as np
import time

class BoxCounter:
    def __init__(self):
        self.counts = {
            "box": {"in": 0, "out": 0},
            "bag": {"in": 0, "out": 0}
        }
        
        # Track history: dict of track_id -> dict with history, class_name, counted
        self.tracks = {}
        
    def process_frame(self, frame, results, line_points):
        h, w = frame.shape[:2]
        events = []
        now = time.time()
        
        line_pt1 = (int(line_points[0][0] * w), int(line_points[0][1] * h)) if line_points and len(line_points) == 2 else None
        line_pt2 = (int(line_points[1][0] * w), int(line_points[1][1] * h)) if line_points and len(line_points) == 2 else None
        
        if line_pt1 and line_pt2:
            # Draw distinct line (CYAN)
            cv2.line(frame, line_pt1, line_pt2, (255, 255, 0), 2)
            cv2.putText(frame, "IN / OUT LINE", (line_pt1[0], line_pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
            
        def get_side(pt, line):
            x, y = pt
            (x1, y1), (x2, y2) = line
            # standard line equation: (y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1
            return (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            names = results[0].names
            
            used_ids = set(ids)
            
            for box, track_id, cls_id, conf in zip(boxes, ids, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Class 0: box (Green), Class 1: bag (Orange)
                # Fallback: if using standard COCO, map 'person' or 'backpack' to bag just for testing
                cls_name = names[cls_id]
                is_bag = (cls_name == 'bag' or cls_id == 1 or cls_name == 'backpack' or cls_name == 'suitcase')
                is_box = not is_bag
                
                cat_name = "bag" if is_bag else "box"
                color = (0, 165, 255) if is_bag else (0, 255, 0)
                
                if track_id not in self.tracks:
                    self.tracks[track_id] = {
                        "history": (cx, cy),
                        "counted": False,
                        "class_name": cat_name,
                        "last_update": now,
                        "side": get_side((cx,cy), (line_pt1, line_pt2)) if line_pt1 else 0
                    }
                else:
                    t = self.tracks[track_id]
                    t["last_update"] = now
                    prev_pt = t["history"]
                    curr_pt = (cx, cy)
                    t["history"] = curr_pt
                    
                    if not t["counted"] and line_pt1 and line_pt2:
                        # Check intersection
                        if intersect(prev_pt, curr_pt, line_pt1, line_pt2):
                            # It crossed! Direction depends on old side vs new side
                            curr_side = get_side(curr_pt, (line_pt1, line_pt2))
                            prev_side = t["side"]
                            t["side"] = curr_side
                            
                            if prev_side > 0 and curr_side < 0:
                                self.counts[cat_name]["in"] += 1
                                events.append({"type": f"{cat_name}_in", "track_id": track_id})
                                t["counted"] = True
                            elif prev_side < 0 and curr_side > 0:
                                self.counts[cat_name]["out"] += 1
                                events.append({"type": f"{cat_name}_out", "track_id": track_id})
                                t["counted"] = True

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                
                label = f"{cat_name.upper()} {track_id} ({conf:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Cleanup old tracks
            to_delete = []
            for tid, t in self.tracks.items():
                if now - t["last_update"] > 5.0:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
                
        # Draw HUD
        cv2.rectangle(frame, (10, 10), (280, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Boxes IN: {self.counts['box']['in']}   OUT: {self.counts['box']['out']}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, events
