import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load YOLO model from {model_path}: {e}")
            self.model = None
            
        # Target classes: For now, if using standard COCO, we map to placeholder classes like
        # 24: backpack (proxy for sack), 28: suitcase (proxy for box)
        # When custom trained, these IDs will likely be 0 (cardboard_box) and 1 (sugar_sack).
        self.target_classes = [24, 28, 0, 1] 
        
        # Name mapping
        self.class_names = {
            24: "sugar_sack (proxy)",
            28: "cardboard_box (proxy)",
            0: "cardboard_box",
            1: "sugar_sack"
        }

    def process_frame(self, frame, conf_threshold, iou_threshold):
        if self.model is None:
            return []

        # Run inference with BoT-SORT tracker
        # model.track gives persistent IDs
        results = self.model.track(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            tracker="botsort.yaml", # or bytetrack.yaml
            persist=True,
            classes=self.target_classes,
            verbose=False
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.int().cpu().numpy()

                for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, class_ids):
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    # Calculate centroid
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    
                    detections.append({
                        "bbox": box,
                        "track_id": track_id,
                        "conf": conf,
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "centroid": (cx, cy)
                    })
        return detections
