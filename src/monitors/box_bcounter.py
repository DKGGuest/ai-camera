import cv2
import numpy as np

class BoxCounter:
    def __init__(self):
        # We will track boxes (or suitcases as proxy) using a simple centroid tracker
        self.objects = {} # dict id -> (cx, cy, last_seen, state)
        self.next_object_id = 1
        self.max_disappeared = 5
        self.disappeared = {}
        
        self.loaded_count = 0
        self.unloaded_count = 0
        
    def _register(self, centroid):
        self.objects[self.next_object_id] = {'centroid': centroid, 'state': None}
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def _get_zone(self, centroid, line_pt1, line_pt2):
        if line_pt1 is None or line_pt2 is None:
            return None
            
        x, y = centroid
        x1, y1 = line_pt1
        x2, y2 = line_pt2
        
        # Calculate cross product to determine side of the line
        # > 0 is one side, < 0 is the other
        position = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        return "loaded_side" if position > 0 else "unloaded_side"
        
    def process_frame(self, frame, boxes, line_points):
        """
        boxes: list of dicts with 'bbox': [x1,y1,x2,y2] from YOLO
        line_points: [(x1,y1), (x2,y2)] relative coordinates (0-1)
        """
        h, w = frame.shape[:2]
        
        line_pt1 = (int(line_points[0][0] * w), int(line_points[0][1] * h)) if line_points and len(line_points) == 2 else None
        line_pt2 = (int(line_points[1][0] * w), int(line_points[1][1] * h)) if line_points and len(line_points) == 2 else None
        
        if line_pt1 and line_pt2:
            cv2.line(frame, line_pt1, line_pt2, (255, 0, 0), 2)
            cv2.putText(frame, "Loaded", (line_pt1[0], line_pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.putText(frame, "Unloaded", (line_pt2[0], line_pt2[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        input_centroids = np.zeros((len(boxes), 2), dtype="int")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box['bbox'])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids[i] = (cx, cy)
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 165, 0), -1)

        event_occurred = False
        event_type = None
        
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return frame, False, None # No event
            
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self._register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            event_occurred = False
            event_type = None
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 50:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Check line crossing
                if line_pt1 and line_pt2:
                    current_zone = self._get_zone(input_centroids[col], line_pt1, line_pt2)
                    prev_zone = self.objects[object_id]['state']
                    
                    if prev_zone is not None and current_zone != prev_zone:
                        if prev_zone == "unloaded_side" and current_zone == "loaded_side":
                            self.loaded_count += 1
                            event_occurred = True
                            event_type = "loaded"
                        elif prev_zone == "loaded_side" and current_zone == "unloaded_side":
                            self.unloaded_count += 1
                            event_occurred = True
                            event_type = "unloaded"
                            
                    self.objects[object_id]['state'] = current_zone
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
                    
            for col in unused_cols:
                self._register(input_centroids[col])
                
        # Draw counts on frame
        cv2.putText(frame, f"Loaded: {self.loaded_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Unloaded: {self.unloaded_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame, event_occurred, event_type
