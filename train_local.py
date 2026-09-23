from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    print("Starting local training for 30 epochs...")
    model.train(
        data='chair_detection_dataset/data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        project='runs/detect',
        name='custom_chair_30ep'
    )
    
if __name__ == '__main__':
    main()
