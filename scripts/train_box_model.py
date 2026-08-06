import os
import shutil
import cv2
from ultralytics import YOLO
import sys
import random

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

CUSTOM_MODEL_PATH = os.path.join(config.BASE_DIR, "models", "custom_box_model.pt")

def setup_directories():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    for d in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
        os.makedirs(d, exist_ok=True)

def generate_dataset():
    print("Loading model for auto-annotation...")
    model = load_custom_box_model()
    
    print("Processing images in:", IMAGE_DIR)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    random.shuffle(image_files)
    
    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    def process_files(files, img_dest, lbl_dest):
        count = 0
        for img_file in files:
            img_path = os.path.join(IMAGE_DIR, img_file)
            txt_file_name = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(IMAGE_DIR, txt_file_name)
            
            dest_img_path = os.path.join(img_dest, img_file)
            dest_txt_path = os.path.join(lbl_dest, txt_file_name)
            
            # If a manual annotation exists, just copy image and label
            if os.path.exists(txt_path):
                shutil.copy(img_path, dest_img_path)
                shutil.copy(txt_path, dest_txt_path)
                count += 1
                continue
                
            # Otherwise, auto-annotate: assume the entire image is a cardboard box
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Try model prediction first
            results = model.predict(img, conf=0.02, augment=True, imgsz=640, verbose=False)
            
            labels = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box, cls_id in zip(results[0].boxes.xywhn.cpu(), results[0].boxes.cls.cpu()):
                    x_center, y_center, bw, bh = box
                    labels.append(f"0 {float(x_center)} {float(y_center)} {float(bw)} {float(bh)}")
            
            # If no detection, assume the whole image is a cardboard box (with margin)
            if not labels:
                labels.append("0 0.5 0.5 0.9 0.9")
            
            shutil.copy(img_path, dest_img_path)
            with open(dest_txt_path, "w") as f:
                f.write("\n".join(labels))
            count += 1
        return count

    train_count = process_files(train_files, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR)
    print(f"Processed {train_count} train images")
    
    val_count = process_files(val_files, IMAGES_VAL_DIR, LABELS_VAL_DIR)
    print(f"Processed {val_count} val images")
    
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
    print("=" * 60)
    print("  TRAINING CARDBOARD BOX MODEL - 10 EXTRA EPOCHS")
    print("=" * 60)
    
    # Resume from the existing custom model to retain previous learning
    if os.path.exists(CUSTOM_MODEL_PATH):
        print(f"Resuming from existing model: {CUSTOM_MODEL_PATH}")
        model = YOLO(CUSTOM_MODEL_PATH)
    else:
        print("No existing custom model, starting from yolov8n.pt")
        model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=YAML_FILE,
        epochs=10,
        imgsz=640,        # Higher resolution for better angle detection
        batch=4,
        scale=0.7,
        degrees=90.0,     # Full rotation augmentation for all angles
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,        # Mix images together for robustness
        copy_paste=0.2,   # Copy-paste augmentation
        project=os.path.join(config.BASE_DIR, "runs"),
        name="cardboard_box_retrain"
    )
    
    print("Training completed!")
    
    # Copy best model
    best_model_path = os.path.join(config.BASE_DIR, "runs", "cardboard_box_retrain", "weights", "best.pt")
    if os.path.exists(best_model_path):
        os.makedirs(os.path.dirname(CUSTOM_MODEL_PATH), exist_ok=True)
        shutil.copy(best_model_path, CUSTOM_MODEL_PATH)
        print(f"Updated model saved to {CUSTOM_MODEL_PATH}")
    else:
        # Try last.pt as fallback
        last_model_path = os.path.join(config.BASE_DIR, "runs", "cardboard_box_retrain", "weights", "last.pt")
        if os.path.exists(last_model_path):
            os.makedirs(os.path.dirname(CUSTOM_MODEL_PATH), exist_ok=True)
            shutil.copy(last_model_path, CUSTOM_MODEL_PATH)
            print(f"Updated model (last.pt) saved to {CUSTOM_MODEL_PATH}")
        else:
            print("ERROR: Could not find best.pt or last.pt!")

if __name__ == "__main__":
    setup_directories()
    generate_dataset()
    train_model()
    print("\nDone! Restart your app.py to use the new model.")
