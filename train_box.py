from ultralytics import YOLO

def train_custom_model():
    print("Loading base YOLOv8 model...")
    # Load a pre-trained YOLOv8 nano model as the starting point
    model = YOLO('yolov8n.pt') 
    
    print("Starting training on custom box dataset...")
    # Train the model. 
    # Make sure 'dataset/data.yaml' is the file you exported from your labeling tool (Roboflow/MakeSense).
    results = model.train(
        data='box_dataset/dataset.yaml', 
        epochs=20,             # Set to 20 to improve accuracy with the new images
        imgsz=640,             
        batch=8,               
        name='omada_box_model' 
    )
    
    print("Training complete! Your custom model is saved in 'runs/detect/omada_box_model/weights/best.pt'")

if __name__ == '__main__':
    train_custom_model()
