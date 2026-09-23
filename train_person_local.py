from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    print("Starting local training for person sitting/standing model (30 epochs)...")
    model.train(
        data='person _sitting_or_standing_dataset/data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        project='runs/detect',
        name='custom_person_30ep'
    )
    
if __name__ == '__main__':
    main()
