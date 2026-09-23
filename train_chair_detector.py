from ultralytics import YOLO

def train_chair_model():
    print("Loading base YOLOv8 model for object detection...")
    # Using yolov8s.pt (small) or yolov8m.pt (medium) is recommended for good accuracy
    model = YOLO('yolov8s.pt') 
    
    print("Starting training on chair detection dataset...")
    # Change this path to where you extracted the Roboflow YOLOv8 dataset
    dataset_yaml = 'chair_detection_dataset/data.yaml'
    
    results = model.train(
        data=dataset_yaml, 
        epochs=50,             # Number of epochs
        imgsz=640,             # Image size
        batch=16,              # Batch size
        name='custom_chair_detector',
        project='runs/detect'
    )
    
    print("Training complete! Your custom model is saved in 'runs/detect/custom_chair_detector/weights/best.pt'")

if __name__ == '__main__':
    train_chair_model()
