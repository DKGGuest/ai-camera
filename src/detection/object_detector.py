import os
import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thresh=0.25, nms_thresh=0.45):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # COCO mapping: 0: person, 63: laptop, 67: cell phone, 64: mouse, 66: keyboard, 62: tv
        self.target_classes = [0, 62, 63, 64, 66, 67]
        
    def detect(self, frame):
        """
        Detects relevant objects in a frame.
        Returns a dictionary of detections grouped by category.
        """
        results = self.model(frame, classes=self.target_classes, conf=self.conf_thresh, iou=self.nms_thresh, verbose=False)
        r = results[0]
        
        detections = {
            "persons": [],
            "laptops": [],
            "phones": [],
            "keyboards": [],
            "mice": [],
            "monitors": []
        }
        
        if r.boxes is not None:
            for box, conf, cls in zip(r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu()):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls)
                c = float(conf)
                
                det = {"box": [x1, y1, x2, y2], "conf": c}
                
                if cls_id == 0:
                    detections["persons"].append(det)
                elif cls_id == 63: # laptop
                    detections["laptops"].append(det)
                elif cls_id == 62: # tv/monitor
                    detections["monitors"].append(det)
                elif cls_id == 67: # cell phone
                    detections["phones"].append(det)
                elif cls_id == 64: # mouse
                    detections["mice"].append(det)
                elif cls_id == 66: # keyboard
                    detections["keyboards"].append(det)
                    
        return detections
