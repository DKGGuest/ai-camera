import os
import cv2
import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", conf_thresh=0.5):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        
    def estimate(self, frame):
        """
        Estimates pose for persons in the frame.
        Returns a list of dictionaries with keypoints and bounding boxes.
        """
        results = self.model(frame, conf=self.conf_thresh, verbose=False)
        r = results[0]
        
        poses = []
        if r.keypoints is not None and r.boxes is not None:
            # zip boxes and keypoints together
            for box, kpts in zip(r.boxes.xyxy.cpu(), r.keypoints.xy.cpu()):
                x1, y1, x2, y2 = map(int, box)
                # kpts is a tensor of shape (17, 2)
                # COCO keypoints: 0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear
                # 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist
                # 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle
                
                keypoints_dict = {
                    "nose": kpts[0].tolist(),
                    "l_shoulder": kpts[5].tolist(),
                    "r_shoulder": kpts[6].tolist(),
                    "l_wrist": kpts[9].tolist(),
                    "r_wrist": kpts[10].tolist(),
                    "l_hip": kpts[11].tolist(),
                    "r_hip": kpts[12].tolist(),
                    "l_knee": kpts[13].tolist(),
                    "r_knee": kpts[14].tolist()
                }
                
                poses.append({
                    "box": [x1, y1, x2, y2],
                    "keypoints": keypoints_dict
                })
        return poses
        
    def determine_posture(self, pose_data):
        """
        Heuristic to determine posture (SITTING or STANDING).
        """
        box = pose_data["box"]
        kpts = pose_data["keypoints"]
        
        h = box[3] - box[1]
        w = box[2] - box[0]
        
        # Aspect ratio check
        if w > 0 and (h / w) > 1.8:
            return "STANDING"
            
        # Keypoint based check (if hips and knees are visible)
        l_hip = kpts["l_hip"]
        l_knee = kpts["l_knee"]
        
        # If knee is significantly below hip, and height is tall relative to width
        if l_hip[1] > 0 and l_knee[1] > 0:
            dy = l_knee[1] - l_hip[1]
            if dy > h * 0.25 and (h / w) > 1.5:
                return "STANDING"
                
        return "SITTING"
