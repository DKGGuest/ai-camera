import os
import shutil
import random
from ultralytics import YOLO

def setup_dataset(source_dir, output_dir):
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(images_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)
    
    # Get all images
    all_files = os.listdir(source_dir)
    images = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Match images with labels
    valid_pairs = []
    for img in images:
        base_name = os.path.splitext(img)[0]
        txt_name = base_name + '.txt'
        if txt_name in all_files:
            valid_pairs.append((img, txt_name))
            
    print(f"Found {len(valid_pairs)} labeled images.")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * 0.8)
    
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    def copy_pairs(pairs, split):
        for img, txt in pairs:
            shutil.copy(os.path.join(source_dir, img), os.path.join(images_dir, split, img))
            shutil.copy(os.path.join(source_dir, txt), os.path.join(labels_dir, split, txt))
            
    copy_pairs(train_pairs, 'train')
    copy_pairs(val_pairs, 'val')
    print("Dataset setup complete.")
    
    # Create dataset.yaml
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
names:
  0: cardboard box
"""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    return yaml_path

if __name__ == '__main__':
    source_dir = r"c:\Users\user\Desktop\New folder\webapp 21341235\imagesbox"
    output_dir = r"c:\Users\user\Desktop\New folder\webapp 21341235\box_dataset"
    
    yaml_path = setup_dataset(source_dir, output_dir)
    
    print("Starting YOLO training...")
    model = YOLO('yolov8n.pt')  # Start with a pretrained nano model
    model.train(data=yaml_path, epochs=100, imgsz=640, project='runs', name='box_model')
    print("Training complete. Model saved in runs/box_model/weights/best.pt")
