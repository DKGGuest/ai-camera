import os
import shutil
from ultralytics import YOLO

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(base_dir, 'datasets', 'worker_cls')
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset not found at {dataset_dir}. Please run crop_dataset.py first.")
        return
        
    print(f"Training worker classifier using dataset at {dataset_dir}...")
    
    # Load base classification model
    model = YOLO('yolov8n-cls.pt')
    
    # Train the model
    results = model.train(
        data=dataset_dir,
        epochs=20,
        imgsz=224,
        batch=16,
        project=os.path.join(base_dir, 'runs'),
        name='worker_cls',
        exist_ok=True
    )
    
    print("Training complete.")
    
    # Copy best model to root directory
    best_model_path = os.path.join(base_dir, 'runs', 'worker_cls', 'weights', 'best.pt')
    target_path = os.path.join(base_dir, 'worker_classifier.pt')
    
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, target_path)
        print(f"Copied newly trained model to {target_path}")
    else:
        print("Warning: Could not find trained weights.")

if __name__ == '__main__':
    main()
