import os
import shutil
from ultralytics import YOLO

# Paths
BASE_DIR = r"C:\Users\user\Desktop\New folder\webapp 21341235"
DATASET_YAML = os.path.join(BASE_DIR, "box_dataset", "dataset.yaml")
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

def train_model():
    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset not found at {DATASET_YAML}")
        print("Please run scripts/prepare_dataset.py first!")
        return

    print("Starting YOLOv8 training on the custom box dataset...")
    
    # Load a pretrained model (recommended for training)
    model = YOLO("yolov8n.pt")
    
    # Train the model
    # Note: If you have an NVIDIA GPU, this will automatically use it.
    # Otherwise, it will run on CPU (which will be slower).
    results = model.train(
        data=DATASET_YAML,
        epochs=30, # 30 epochs is a good starting point for a small dataset
        imgsz=640,
        batch=16,
        project=os.path.join(BASE_DIR, "runs"),
        name="custom_box_training"
    )
    
    # The trained weights are saved in runs/custom_box_training/weights/best.pt
    # We will copy it to the root directory for easy access
    trained_weights_path = os.path.join(BASE_DIR, "runs", "custom_box_training", "weights", "best.pt")
    
    if os.path.exists(trained_weights_path):
        shutil.copy(trained_weights_path, OUTPUT_MODEL_PATH)
        print(f"Training complete! Custom model saved to {OUTPUT_MODEL_PATH}")
    else:
        print("Training may have failed. Could not find best.pt.")

if __name__ == "__main__":
    train_model()
