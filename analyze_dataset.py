import os
import glob
from collections import defaultdict

dataset_path = r"c:\Users\user\Desktop\New folder\webapp 21341235\worker_images"

classes = defaultdict(int)
total_images = 0
total_annotations = 0
resolutions = set()

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            total_images += 1
        elif file.endswith('.txt') and file != 'classes.txt':
            total_annotations += 1
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        cls_id = parts[0]
                        classes[cls_id] += 1

print(f"Total Images: {total_images}")
print(f"Total Annotation Files: {total_annotations}")
print("Class Distribution:")
for k, v in classes.items():
    print(f"Class {k}: {v} instances")

