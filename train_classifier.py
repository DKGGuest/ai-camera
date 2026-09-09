from ultralytics import YOLO

def train():
    model = YOLO("yolov8n-cls.pt")
    
    results = model.train(
        data=r"c:\Users\user\Desktop\New folder\webapp new\webapp new\chair_occupancy_dataset",
        epochs=100,
        imgsz=224,
        project=r"c:\Users\user\Desktop\New folder\webapp new\webapp new\runs\classify",
        name="chair_classifier"
    )

if __name__ == "__main__":
    train()
