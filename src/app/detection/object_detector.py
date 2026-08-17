import os
from ultralytics import YOLO

class ObjectDetector:
    """
    Handles detection of person, laptop, phone, etc.
    Using a single YOLO model for efficiency.
    """
    def __init__(self, model_path: str, conf_thresholds: dict, device: str = "cuda"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detection model not found at {model_path}")
            
        self.model = YOLO(model_path)
        self.device = device
        self.conf_thresholds = conf_thresholds
        
        # COCO IDs
        self.classes = {
            'person': 0,
            'laptop': 63,
            'phone': 67
        }
        
    def detect(self, frame):
        """
        Run inference on a single frame.
        Returns a dict of detections categorized by class.
        """
        # Run inference for our target classes only
        target_class_ids = list(self.classes.values())
        results = self.model.predict(
            source=frame,
            classes=target_class_ids,
            device=self.device,
            verbose=False
        )
        
        detections = {
            'person': [],
            'laptop': [],
            'phone': []
        }
        
        if not results or len(results[0].boxes) == 0:
            return detections
            
        boxes = results[0].boxes
        for box, cls_id, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.int().cpu().numpy(), boxes.conf.cpu().numpy()):
            
            x1, y1, x2, y2 = map(int, box)
            
            # Map cls_id back to name
            cls_name = None
            for name, cid in self.classes.items():
                if cid == cls_id:
                    cls_name = name
                    break
                    
            if cls_name and conf >= self.conf_thresholds.get(f"{cls_name}_conf", 0.5):
                detections[cls_name].append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf)
                })
                
        return detections
