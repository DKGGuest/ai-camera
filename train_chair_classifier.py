from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-cls.pt')  # load a pretrained classification model
    dataset_path = 'chair_occupancy_dataset'
    
    print(f"Training on dataset: {dataset_path} for 50 epochs...")
    results = model.train(data=dataset_path, epochs=50, imgsz=224, name='chair_classifier', project='runs/classify', device=0)

if __name__ == '__main__':
    main()
