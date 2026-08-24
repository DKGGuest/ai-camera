import os
import glob
import cv2
import random

# Mapping of raw folders to classification classes
MAPPING = {
    'working': 'working',
    'laptop_images': 'working',
    'non-working': 'not_working',
    'phone_images': 'not_working'
}

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(base_dir, 'worker_images')
    dataset_dir = os.path.join(base_dir, 'datasets', 'worker_cls')
    
    # Create dataset structure
    for split in ['train', 'val']:
        for cls in ['working', 'not_working']:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)
            
    crop_count = 0
    
    for raw_folder, cls_name in MAPPING.items():
        folder_path = os.path.join(images_dir, raw_folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist.")
            continue
            
        print(f"Processing {folder_path} -> {cls_name}")
        
        # Find all images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
            
        for img_path in image_paths:
            # Check for label file
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if not os.path.exists(txt_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            # Extract persons (class 0)
            person_idx = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == '0':
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    # Convert to absolute coords
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    
                    # Clamp to image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Crop
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        split = 'val' if random.random() < 0.2 else 'train' # 80/20 split
                        base_name = os.path.basename(img_path)
                        crop_name = f"{os.path.splitext(base_name)[0]}_p{person_idx}.jpg"
                        out_path = os.path.join(dataset_dir, split, cls_name, crop_name)
                        cv2.imwrite(out_path, crop)
                        crop_count += 1
                        person_idx += 1
                        
    print(f"Successfully generated {crop_count} cropped images in {dataset_dir}")

if __name__ == '__main__':
    main()
