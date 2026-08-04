import cv2
from ultralytics import YOLOWorld

def test_yolo_world():
    model = YOLOWorld('yolov8s-world.pt')
    
    # Set the target classes for YOLO-World
    model.set_classes(["cardboard box", "sugar sack"])
    
    img_path = r"imagesbox\WhatsApp Image 2026-08-03 at 5.13.38 PM.jpeg"
    
    results = model.predict(img_path, conf=0.05)
    
    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy.cpu(), r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().tolist()):
            print(f"Detected: {model.names[cls_id]} with confidence {conf:.2f}")

if __name__ == "__main__":
    test_yolo_world()
