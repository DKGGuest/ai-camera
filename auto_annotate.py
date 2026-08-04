import os
import glob
import cv2
from ultralytics import YOLO

def auto_annotate():
    model = YOLO('best.pt')
    image_paths = glob.glob(r'imagesbox\*.jpeg')
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Predict with very low confidence
        results = model.predict(img, conf=0.01, verbose=False)
        
        best_box = None
        max_area = 0
        best_cls = 0
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box, cls_id, conf in zip(results[0].boxes.xywhn.cpu(), results[0].boxes.cls.int().cpu().tolist(), results[0].boxes.conf.cpu().tolist()):
                bx, by, bw, bh = box
                area = bw * bh
                if area > max_area:
                    max_area = area
                    best_box = box
                    best_cls = cls_id
                    
        label_path = img_path.replace('.jpeg', '.txt')
        if best_box is not None and max_area > 0.01:
            with open(label_path, 'w') as f:
                f.write(f"0 {best_box[0]:.6f} {best_box[1]:.6f} {best_box[2]:.6f} {best_box[3]:.6f}\n")
            print(f"Annotated {os.path.basename(img_path)} - area: {max_area:.2f}")
        else:
            print(f"Failed to find box in {os.path.basename(img_path)}")

if __name__ == '__main__':
    auto_annotate()
