import os
import shutil
import cv2
from ultralytics import YOLO
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models_loader import load_custom_box_model

IMAGE_DIR = r"C:\Users\user\Desktop\New folder\webapp 21341235\imagesbox"
DATASET_DIR = os.path.join(config.BASE_DIR, "datasets", "cardboard_box_dataset")
IMAGES_TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
IMAGES_VAL_DIR = os.path.join(DATASET_DIR, "images", "val")
LABELS_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")
LABELS_VAL_DIR = os.path.join(DATASET_DIR, "labels", "val")
YAML_FILE = os.path.join(DATASET_DIR, "data.yaml")

def setup_directories():
    # Re-create directories to ensure a clean state
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    for d in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
        os.makedirs(d, exist_ok=True)

def generate_dataset():
    print("Loading model for auto-annotation...")
    model = load_custom_box_model()
    
    print("Processing images in:", IMAGE_DIR)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle for a random split
    import random
    random.shuffle(image_files)
    
    # Simple split
    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    def process_files(files, img_dest, lbl_dest):
        for img_file in files:
            img_path = os.path.join(IMAGE_DIR, img_file)
            txt_file_name = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(IMAGE_DIR, txt_file_name)
            
            dest_img_path = os.path.join(img_dest, img_file)
            dest_txt_path = os.path.join(lbl_dest, txt_file_name)
            
            # If a manual annotation exists, just copy the image and the text file
            if os.path.exists(txt_path):
                shutil.copy(img_path, dest_img_path)
                shutil.copy(txt_path, dest_txt_path)
                continue
                
            # Otherwise, use auto-annotation
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Predict with TTA (Test-Time Augmentation) and high resolution to find small boxes
            results = model.predict(img, conf=0.02, augment=True, imgsz=1280, verbose=False)
            
            h, w = img.shape[:2]
            labels = []
            if results and results[0].boxes is not None:
                for box, cls_id in zip(results[0].boxes.xywhn.cpu(), results[0].boxes.cls.cpu()):
                    cls_name = results[0].names[int(cls_id)]
                    # We only care about box-like classes
                    if cls_name in ["box", "cardboard box", "sugar_sack", "person"]:
                        x_center, y_center, width, height = box
                        labels.append(f"0 {float(x_center)} {float(y_center)} {float(width)} {float(height)}")
            
            if labels:
                # Copy image
                shutil.copy(img_path, dest_img_path)
                # Write label
                with open(dest_txt_path, "w") as f:
                    f.write("\n".join(labels))

    print(f"Processing {len(train_files)} train images...")
    process_files(train_files, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR)
    
    print(f"Processing {len(val_files)} val images...")
    process_files(val_files, IMAGES_VAL_DIR, LABELS_VAL_DIR)
    
    # Write yaml
    yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: cardboard box
"""
    with open(YAML_FILE, "w") as f:
        f.write(yaml_content.strip())
        
    print(f"Dataset created at {DATASET_DIR}")

def train_model():
    print("Starting training...")
    # Load a stronger base model for better accuracy
    model = YOLO("yolov8n.pt") # Use the faster nano model
    
    # Train with more epochs and heavy augmentation to learn all angles and reduce flickering
    results = model.train(
        data=YAML_FILE,
        epochs=20, # Reduced to 20 for speed
        imgsz=320, # Reduced image size for 4x speedup
        batch=4,
        scale=0.7,
        degrees=45.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        project=os.path.join(config.BASE_DIR, "runs"),
        name="cardboard_box_finetune"
    )
    
    print("Training completed.")
    # Copy best model to our custom path
    best_model_path = os.path.join(config.BASE_DIR, "runs", "cardboard_box_finetune", "weights", "best.pt")
    if os.path.exists(best_model_path):
        dest_model_path = os.path.join(config.BASE_DIR, "models", "custom_box_model.pt")
        os.makedirs(os.path.dirname(dest_model_path), exist_ok=True)
        shutil.copy(best_model_path, dest_model_path)
        print(f"New model saved to {dest_model_path}")
    else:
        print("Error: Could not find best.pt")

if __name__ == "__main__":
    setup_directories()
    generate_dataset()
    train_model()
