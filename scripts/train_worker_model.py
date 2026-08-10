import os
import shutil
import random
from ultralytics import YOLO

def setup_dataset(src_dir, dest_dir, split_ratio=0.8):
    classes = ['working', 'non-working']
    
    # Create dataset directories
    for split in ['train', 'val']:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
            
    # Copy and split images
    for cls in classes:
        cls_src = os.path.join(src_dir, cls)
        if not os.path.exists(cls_src):
            print(f"Warning: Source directory not found: {cls_src}")
            continue
            
        images = [f for f in os.listdir(cls_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        for img in train_images:
            shutil.copy2(os.path.join(cls_src, img), os.path.join(dest_dir, 'train', cls, img))
            
        for img in val_images:
            shutil.copy2(os.path.join(cls_src, img), os.path.join(dest_dir, 'val', cls, img))
            
        print(f"Class '{cls}': {len(train_images)} train, {len(val_images)} val")

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_images = os.path.join(project_root, 'worker_images')
    dataset_dir = os.path.join(project_root, 'datasets', 'worker_cls')
    
    # clean up previous dataset if exists to prevent duplicates
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        
    print("Setting up dataset...")
    setup_dataset(src_images, dataset_dir)
    
    print("Training YOLO classification model...")
    model = YOLO('yolov8s-cls.pt')
    
    results = model.train(
        data=dataset_dir,
        epochs=100,
        imgsz=224,
        project=project_root,
        name='worker_cls_train',
        exist_ok=True # overwrite previous training run if any
    )
    
    # After training, ultralytics saves the best model in <project>/<name>/weights/best.pt
    best_model_path = os.path.join(project_root, 'worker_cls_train', 'weights', 'best.pt')
    final_model_path = os.path.join(project_root, 'worker_classifier.pt')
    
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, final_model_path)
        print(f"Training complete. Model saved to: {final_model_path}")
    else:
        print("Training completed but could not find best.pt!")

if __name__ == '__main__':
    main()
