import cv2
import time

class BoxCounter:
    def __init__(self):
        self.counts = {"in": 0, "out": 0}
        self.tracks = {}
        self.current_day = time.localtime().tm_yday
        
        
    def process_frame(self, frame, results, line_points):
        h, w = frame.shape[:2]
        events = []
        now = time.time()
        today = time.localtime(now).tm_yday
        
        if today != self.current_day:
            self.counts = {"in": 0, "out": 0}
            self.current_day = today
            
        line_pt1 = (int(line_points[0][0] * w), int(line_points[0][1] * h)) if line_points and len(line_points) == 2 else None
        line_pt2 = (int(line_points[1][0] * w), int(line_points[1][1] * h)) if line_points and len(line_points) == 2 else None
        
        if line_pt1 and line_pt2:
            cv2.line(frame, line_pt1, line_pt2, (255, 255, 0), 2)
            cv2.putText(frame, "IN / OUT LINE", (line_pt1[0], line_pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
            
        def get_side(pt, line):
            x, y = pt
            (x1, y1), (x2, y2) = line
            return (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            names = results[0].names
            
            for box, track_id, cls_id, conf in zip(boxes, ids, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                cls_name = names[cls_id]
                
                if track_id not in self.tracks:
                    self.tracks[track_id] = {
                        "history": (cx, cy),
                        "counted": False,
                        "class_name": cls_name,
                        "last_update": now,
                        "side": get_side((cx,cy), (line_pt1, line_pt2)) if line_pt1 else 0,
                        "last_cross_time": 0
                    }
                else:
                    t = self.tracks[track_id]
                    t["last_update"] = now
                    prev_pt = t["history"]
                    curr_pt = (cx, cy)
                    t["history"] = curr_pt
                    
                    if line_pt1 and line_pt2:
                        if intersect(prev_pt, curr_pt, line_pt1, line_pt2):
                            if now - t["last_cross_time"] > 2.0:
                                curr_side = get_side(curr_pt, (line_pt1, line_pt2))
                                prev_side = t["side"]
                                t["side"] = curr_side
                                
                                if prev_side > 0 and curr_side < 0:
                                    self.counts["in"] += 1
                                    events.append({"type": "in", "track_id": track_id})
                                    t["last_cross_time"] = now
                                elif prev_side < 0 and curr_side > 0:
                                    self.counts["out"] += 1
                                    events.append({"type": "out", "track_id": track_id})
                                    t["last_cross_time"] = now

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                label = f"#{track_id} {cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        to_delete = []
        for tid, t in self.tracks.items():
            if now - t["last_update"] > 5.0:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
            
        cv2.putText(frame, f"in: {self.counts['in']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"out: {self.counts['out']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame, events
