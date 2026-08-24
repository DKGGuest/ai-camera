import math

class WorkerTracker:
    def __init__(self):
        self.workers = {} # map of worker_id to state
        self.next_id = 1
        
    def update(self, person_detections):
        """
        Simple centroid-based tracker for fallback if ultralytics built-in tracking is not used.
        Ideally we would use YOLO's .track() mode.
        If using .track(), this class maps YOLO's tracker IDs to our system worker states.
        """
        assigned_ids = set()
        active_workers = {}
        
        # In a real BoT-SORT pipeline, person_detections would already have an 'id' field
        # Here we just assume they might not and use a simple centroid distance matcher if 'id' is missing
        
        for det in person_detections:
            box = det["box"]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            best_id = None
            if "id" in det and det["id"] is not None:
                best_id = det["id"]
            else:
                # Fallback to centroid matching
                min_dist = 200
                for wid, state in self.workers.items():
                    if wid in assigned_ids:
                        continue
                    old_cx, old_cy = state["centroid"]
                    dist = math.hypot(cx - old_cx, cy - old_cy)
                    if dist < min_dist:
                        min_dist = dist
                        best_id = wid
                        
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                
            assigned_ids.add(best_id)
            
            # Update state
            active_workers[best_id] = {
                "id": best_id,
                "box": box,
                "centroid": (cx, cy),
                "conf": det.get("conf", 1.0)
            }
            
        self.workers = active_workers
        return list(active_workers.values())
