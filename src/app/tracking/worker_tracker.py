import os
from ultralytics import YOLO

class WorkerTracker:
    """
    Handles detection and tracking of workers (persons) and objects (laptops, phones).
    Uses BoT-SORT for persistent ID tracking across frames.
    """
    def __init__(self, model_path: str, tracker_yaml: str, conf_thresholds: dict, device: str = "cuda"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tracking model not found at {model_path}")
            
        self.model = YOLO(model_path)
        self.device = device
        self.conf_thresholds = conf_thresholds
        
        # BoT-SORT or ByteTrack config path (usually just 'botsort.yaml' built into ultralytics)
        self.tracker_yaml = tracker_yaml
        
        # COCO IDs
        self.classes = {
            'person': 0,
            'laptop': 63,
            'phone': 67
        }
        
    def track(self, frame):
        """
        Run tracking on a single frame.
        Returns a dictionary of tracked objects.
        """
        target_class_ids = list(self.classes.values())
        
        # Run tracking
        results = self.model.track(
            source=frame,
            classes=target_class_ids,
            persist=True,
            tracker=self.tracker_yaml,
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
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            
            # Tracking ID might be None if the tracker just initialized or lost the object
            track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None and len(boxes.id) > i else None
            
            x1, y1, x2, y2 = map(int, box)
            
            # Map cls_id back to name
            cls_name = None
            for name, cid in self.classes.items():
                if cid == cls_id:
                    cls_name = name
                    break
                    
            if cls_name and conf >= self.conf_thresholds.get(f"{cls_name}_conf", 0.5):
                detections[cls_name].append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })
                
        return detections
