import cv2
import os

video_path = "WhatsApp Video 2026-07-30 at 12.01.13 PM.mp4"
output_dir = "imagesbox"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

frame_interval = int(fps * 0.5)  # 2 frames per second
count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if count % frame_interval == 0:
        filename = os.path.join(output_dir, f"video_frame_{saved}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1
        
    count += 1

cap.release()
print(f"Extracted {saved} frames to {output_dir}")
