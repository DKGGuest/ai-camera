import json

class WorkstationManager:
    def __init__(self, config_data):
        self.workstations = config_data.get("workstations", [])
        
    def get_workstation_for_worker(self, worker_box):
        """
        Given a worker bounding box, returns the associated workstation ID.
        Uses simple Intersection over Area (IoA) or centroid inclusion.
        """
        wx1, wy1, wx2, wy2 = worker_box
        wcx, wcy = (wx1 + wx2) / 2, (wy1 + wy2) / 2
        
        for ws in self.workstations:
            zone = ws["zone"] # [x1, y1, x2, y2]
            zx1, zy1, zx2, zy2 = zone
            
            # Check if centroid is within zone
            if zx1 <= wcx <= zx2 and zy1 <= wcy <= zy2:
                return ws["id"]
                
            # Or check overlap
            overlap_x1 = max(wx1, zx1)
            overlap_y1 = max(wy1, zy1)
            overlap_x2 = min(wx2, zx2)
            overlap_y2 = min(wy2, zy2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                # there is overlap, return this workstation
                # In a more advanced implementation, calculate IoU and pick max
                return ws["id"]
                
        return None
