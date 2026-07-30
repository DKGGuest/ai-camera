import os
import shutil
import cv2
from ultralytics import YOLO

# Paths
IMAGE_DIR = r"C:\Users\user\Desktop\video_webapp\cardboard_images\cardboard_images"
OUTPUT_DIR = r"C:\Users\user\Desktop\New folder\webapp 21341235\box_dataset"

def setup_directories():
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"), exist_ok=True)

def auto_annotate():
    print("Setting up directories...")
    setup_directories()
    
    # We will use YOLOv8n to try and pre-annotate. 
    # This might not catch everything perfectly, but it gives a starting point.
    print("Loading base YOLO model for auto-annotation...")
    model = YOLO("yolov8n.pt") 
    
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(images)} images.")
    
    train_split = 0.8
    train_count = int(len(images) * train_split)
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        split = "train" if i < train_count else "val"
        
        # Copy image
        out_img_path = os.path.join(OUTPUT_DIR, "images", split, img_name)
        shutil.copy(img_path, out_img_path)
        
        # Run inference
        results = model(img_path, verbose=False)
        r = results[0]
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(OUTPUT_DIR, "labels", split, label_name)
        
        with open(label_path, "w") as f:
            if r.boxes is not None:
                for box in r.boxes:
                    # We will force class 0 (box) for any bounding box found 
                    # since we only care about the boxes in these specific images.
                    # YOLO format: class x_center y_center width height (normalized)
                    x, y, w, h = box.xywhn[0].cpu().numpy()
                    f.write(f"0 {x} {y} {w} {h}\n")
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} images...")
            
    # Create dataset.yaml
    yaml_content = f"""
path: {OUTPUT_DIR}
train: images/train
val: images/val

names:
  0: box
"""
    with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
        f.write(yaml_content)
        
    print(f"Dataset generated at: {OUTPUT_DIR}")
    print("IMPORTANT: You must manually verify the labels in the labels folder before training.")
    print("Use a tool like Label Studio or Roboflow if you want a visual editor.")

if __name__ == "__main__":
    auto_annotate()
