import os
from utils import log_event

class Counter:
    def __init__(self, log_path="logs.csv"):
        self.loading_count = 0
        self.unloading_count = 0
        self.crossed_ids = set()
        
        # State tracking: dictionary mapping tracking_id to previous side of the line
        self.object_states = {}
        self.log_path = log_path

    def reset(self):
        self.loading_count = 0
        self.unloading_count = 0
        self.crossed_ids.clear()
        self.object_states.clear()

    def process_object(self, tracking_id, cls_name, centroid, line_mgr, width, height):
        if tracking_id in self.crossed_ids:
            return  # Already counted, skip to prevent duplicates

        pt1, pt2 = line_mgr.get_absolute_points(width, height)
        if not pt1 or not pt2:
            return

        current_side_val = line_mgr.get_side(centroid, pt1, pt2)
        
        # Determine side strictly (ignore 0 which is exactly on the line)
        if current_side_val == 0:
            return
            
        current_side = "A" if current_side_val > 0 else "B"

        if tracking_id in self.object_states:
            prev_side = self.object_states[tracking_id]
            if prev_side != current_side:
                # Crossed the line!
                if prev_side == "B" and current_side == "A":
                    self.loading_count += 1
                    direction = "Truck -> Factory"
                    action = "Loading"
                else:
                    self.unloading_count += 1
                    direction = "Factory -> Truck"
                    action = "Unloading"

                self.crossed_ids.add(tracking_id)
                log_event(self.log_path, tracking_id, cls_name, direction, action)
        
        # Update state
        if tracking_id not in self.crossed_ids:
            self.object_states[tracking_id] = current_side
            
    def cleanup_states(self, active_tracking_ids):
        # Remove old states to prevent memory leaks for disappeared objects
        stale_ids = [tid for tid in self.object_states.keys() if tid not in active_tracking_ids and tid not in self.crossed_ids]
        for tid in stale_ids:
            del self.object_states[tid]
