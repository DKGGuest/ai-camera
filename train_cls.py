import os
import shutil
import random
from ultralytics import YOLO
from pathlib import Path

def setup_dataset():
    source_dir = "worker_images"
    dataset_dir = "datasets/worker_cls"
    
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        
    # Create dataset directories
    for split in ['train', 'val']:
        for cls in ['working', 'non-working', 'laptop_images', 'phone_images']:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)
            
    # Process each class
    for cls in ['working', 'non-working', 'laptop_images', 'phone_images']:
        cls_dir = os.path.join(source_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        # 80/20 split
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train
        for img in train_images:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(dataset_dir, 'train', cls, img)
            shutil.copy2(src, dst)
            
        # Copy to val
        for img in val_images:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(dataset_dir, 'val', cls, img)
            shutil.copy2(src, dst)
            
    return os.path.abspath(dataset_dir)

def train_model(dataset_path):
    print("Loading base YOLOv8 classification model...")
    # Using 'worker_classifier.pt' as a starting point to continue training
    model = YOLO('worker_classifier.pt') 
    
    print("Starting deep training for 100 epochs on custom dataset...")
    results = model.train(
        data=dataset_path,
        epochs=100,
        imgsz=224,
        batch=16,
        name='worker_cls_model'
    )
    
    # After training, copy best model to main directory
    run_dir = model.trainer.save_dir
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    
    if os.path.exists(best_weights):
        print(f"Copying best model from {best_weights} to worker_classifier.pt")
        shutil.copy2(best_weights, "worker_classifier.pt")
        print("Model successfully updated!")

if __name__ == '__main__':
    dataset_path = setup_dataset()
    train_model(dataset_path)
