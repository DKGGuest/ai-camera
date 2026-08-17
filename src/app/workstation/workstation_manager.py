def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union for two bounding boxes [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_intersection_ratio(target_box, zone_box):
    """Calculate what percentage of target_box is inside zone_box"""
    xA = max(target_box[0], zone_box[0])
    yA = max(target_box[1], zone_box[1])
    xB = min(target_box[2], zone_box[2])
    yB = min(target_box[3], zone_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    targetArea = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
    return interArea / float(targetArea)

class WorkstationManager:
    def __init__(self, workstation_config):
        self.workstations = workstation_config
        
    def associate(self, persons, laptops):
        """
        Associate detected persons and laptops with workstations.
        Returns a dictionary mapping workstation ID to its current status.
        """
        assignments = {
            ws_id: {"person": None, "laptop": None}
            for ws_id in self.workstations.keys()
        }
        
        # Associate persons
        for person in persons:
            best_ws = None
            best_ratio = 0
            for ws_id, ws_data in self.workstations.items():
                ratio = calculate_intersection_ratio(person["bbox"], ws_data["person_zone"])
                if ratio > 0.3 and ratio > best_ratio: # At least 30% inside
                    best_ratio = ratio
                    best_ws = ws_id
            
            if best_ws:
                assignments[best_ws]["person"] = person
                
        # Associate laptops
        for laptop in laptops:
            best_ws = None
            best_ratio = 0
            for ws_id, ws_data in self.workstations.items():
                ratio = calculate_intersection_ratio(laptop["bbox"], ws_data["laptop_zone"])
                if ratio > 0.3 and ratio > best_ratio:
                    best_ratio = ratio
                    best_ws = ws_id
                    
            if best_ws:
                # If multiple laptops in zone, take the one with highest IOU/confidence (simplified here)
                assignments[best_ws]["laptop"] = laptop
                
        return assignments
