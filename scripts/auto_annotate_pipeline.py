import os
import glob
import cv2
import shutil
from ultralytics import YOLO

# COCO Classes: 0: person, 63: laptop, 67: cell phone
TARGET_CLASSES = [0, 63, 67]

def auto_annotate(image_dir, model_path='yolov8s.pt'):
    print(f"Loading model {model_path} for auto-annotation...")
    model = YOLO(model_path)
    
    # We will look for images in the image_dir and its subdirectories
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_paths)} images to annotate in {image_dir}")
    
    annotated_count = 0
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Predict with standard confidence
        results = model.predict(img, conf=0.25, verbose=False)
        
        label_path = os.path.splitext(img_path)[0] + '.txt'
        
        boxes_to_write = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box, cls_id in zip(results[0].boxes.xywhn.cpu(), results[0].boxes.cls.int().cpu().tolist()):
                if cls_id in TARGET_CLASSES:
                    bx, by, bw, bh = box
                    boxes_to_write.append(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
                    
        if boxes_to_write:
            with open(label_path, 'w') as f:
                f.writelines(boxes_to_write)
            annotated_count += 1
            
    print(f"Successfully auto-annotated {annotated_count} images.")

if __name__ == '__main__':
    # Assuming run from root directory
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(workspace_dir, 'worker_images')
    auto_annotate(data_dir)
