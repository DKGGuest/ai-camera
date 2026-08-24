import math

class InteractionAnalyzer:
    def __init__(self):
        pass
        
    def calculate_interaction(self, pose_data, laptop_boxes, keyboard_boxes):
        """
        Determines if a worker's hands (wrists) are interacting with a laptop or keyboard.
        """
        kpts = pose_data.get("keypoints", {})
        l_wrist = kpts.get("l_wrist", [0, 0])
        r_wrist = kpts.get("r_wrist", [0, 0])
        
        # Active Interaction = wrists are visible and inside or very close to laptop/keyboard bounding boxes
        
        is_interacting = False
        
        target_zones = laptop_boxes + keyboard_boxes
        
        for zone in target_zones:
            zx1, zy1, zx2, zy2 = zone["box"]
            # add a margin of 50 pixels
            margin = 50
            zx1 -= margin
            zy1 -= margin
            zx2 += margin
            zy2 += margin
            
            # Check left wrist
            if l_wrist[0] > 0 and l_wrist[1] > 0:
                if zx1 <= l_wrist[0] <= zx2 and zy1 <= l_wrist[1] <= zy2:
                    is_interacting = True
                    break
                    
            # Check right wrist
            if r_wrist[0] > 0 and r_wrist[1] > 0:
                if zx1 <= r_wrist[0] <= zx2 and zy1 <= r_wrist[1] <= zy2:
                    is_interacting = True
                    break
                    
        return "ACTIVE" if is_interacting else "NONE"
