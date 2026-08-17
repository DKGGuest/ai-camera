import os
import math
from ultralytics import YOLO

class PoseEstimator:
    """
    Estimates pose and posture for workers.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose model not found at {model_path}")
            
        self.model = YOLO(model_path)
        self.device = device
        
        # YOLOv8 Pose keypoint indices (COCO format)
        self.KP = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }

    def estimate(self, frame):
        """
        Run pose estimation on the frame.
        Returns a list of dictionaries with bounding boxes and keypoints.
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            verbose=False,
            classes=[0] # Only detect persons
        )
        
        persons = []
        if not results or len(results[0].boxes) == 0 or results[0].keypoints is None:
            return persons
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        keypoints = results[0].keypoints.data.cpu().numpy() # [N, 17, 3] (x, y, conf)
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            kpts = keypoints[i]
            
            posture = self._analyze_posture(kpts, y1, y2)
            
            persons.append({
                "bbox": [x1, y1, x2, y2],
                "keypoints": kpts,
                "posture": posture
            })
            
        return persons

    def _analyze_posture(self, kpts, y1, y2):
        """
        Simple heuristic to determine Sitting vs Standing based on keypoints.
        A more advanced implementation would train a classifier on the keypoints.
        """
        # Get Y-coordinates of shoulders, hips, knees
        l_shoulder_y = kpts[self.KP['left_shoulder']][1] if kpts[self.KP['left_shoulder']][2] > 0.5 else None
        r_shoulder_y = kpts[self.KP['right_shoulder']][1] if kpts[self.KP['right_shoulder']][2] > 0.5 else None
        
        l_hip_y = kpts[self.KP['left_hip']][1] if kpts[self.KP['left_hip']][2] > 0.5 else None
        r_hip_y = kpts[self.KP['right_hip']][1] if kpts[self.KP['right_hip']][2] > 0.5 else None
        
        l_knee_y = kpts[self.KP['left_knee']][1] if kpts[self.KP['left_knee']][2] > 0.5 else None
        r_knee_y = kpts[self.KP['right_knee']][1] if kpts[self.KP['right_knee']][2] > 0.5 else None

        # Calculate average Y positions if visible
        shoulder_y = None
        if l_shoulder_y and r_shoulder_y: shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
        elif l_shoulder_y: shoulder_y = l_shoulder_y
        elif r_shoulder_y: shoulder_y = r_shoulder_y
        
        hip_y = None
        if l_hip_y and r_hip_y: hip_y = (l_hip_y + r_hip_y) / 2
        elif l_hip_y: hip_y = l_hip_y
        elif r_hip_y: hip_y = r_hip_y

        knee_y = None
        if l_knee_y and r_knee_y: knee_y = (l_knee_y + r_knee_y) / 2
        elif l_knee_y: knee_y = l_knee_y
        elif r_knee_y: knee_y = r_knee_y

        # Basic heuristic: if distance from hip to knee is small horizontally but large vertically, they might be standing.
        # Conversely, if hips and knees are close vertically, sitting.
        # Alternatively, rely on torso height relative to bounding box height.
        bbox_h = y2 - y1
        if shoulder_y and hip_y:
            torso_h = hip_y - shoulder_y
            if torso_h / bbox_h > 0.6:
                return "SITTING" # Bbox is mostly torso, legs not fully visible
            else:
                return "STANDING"
                
        # Fallback if keypoints are occluded
        return "UNKNOWN"
