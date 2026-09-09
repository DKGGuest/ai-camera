import os
import shutil
import random

def prepare_dataset():
    src_occupied = r"c:\Users\user\Desktop\New folder\webapp new\webapp new\chair_images\occupied_chairs"
    src_empty1 = r"c:\Users\user\Desktop\New folder\webapp new\webapp new\chair_images\empty_chairs"
    src_empty2 = r"c:\Users\user\Desktop\New folder\webapp new\webapp new\chair_images\chair"
    
    dataset_dir = r"c:\Users\user\Desktop\New folder\webapp new\webapp new\chair_occupancy_dataset"
    
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    # Create structure
    for split in ['train', 'val']:
        for cls in ['empty', 'occupied']:
            os.makedirs(os.path.join(dataset_dir, split, cls), exist_ok=True)
            
    # Function to distribute files
    def process_folder(src_folder, target_cls, prefix=""):
        if not os.path.exists(src_folder): return
        
        files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)
        
        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        for i, f in enumerate(train_files):
            new_name = f"{prefix}_train_{i}_{f}"
            shutil.copy(os.path.join(src_folder, f), os.path.join(dataset_dir, 'train', target_cls, new_name))
            
        for i, f in enumerate(val_files):
            new_name = f"{prefix}_val_{i}_{f}"
            shutil.copy(os.path.join(src_folder, f), os.path.join(dataset_dir, 'val', target_cls, new_name))
            
        print(f"Processed {len(files)} files from {src_folder} into {target_cls}")
            
    process_folder(src_occupied, 'occupied', 'occ')
    process_folder(src_empty1, 'empty', 'emp1')
    process_folder(src_empty2, 'empty', 'emp2')

if __name__ == '__main__':
    prepare_dataset()
    print("Dataset prepared successfully!")
