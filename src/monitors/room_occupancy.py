import cv2
import numpy as np
import time

from src.core.video import initialize_video_capture, initialize_video_writer
from src.core.models import load_yolo_model

def run(video_source='0', output_path='room_output.mp4'):
    print("Loading YOLOv8 model for room occupancy counting...")
    model = load_yolo_model('yolov8s.pt') 
    PERSON_CLASS = 0

    frame, cap, width, height, fps, is_image = initialize_video_capture(video_source)
    if width == 0:
        return

    out, final_output_path = initialize_video_writer(output_path, width, height, fps)

    if not is_image:
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading video stream")
            return
    else:
        first_frame = frame.copy()

    if not is_image:
        frame = first_frame
        
    print(f"Processing '{video_source}'...")
    
    while True:
        loop_start = time.time()
        
        # Use the model with ByteTrack to eliminate flickering
        results = model.track(frame, classes=[0], persist=True, tracker="botsort.yaml", verbose=False)
        
        people_count = 0
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu()
            confidences = results[0].boxes.conf.cpu()
            
            for box, conf in zip(boxes, confidences):
                if conf < 0.25: # Lowered threshold to catch occluded people
                    continue
                    
                people_count += 1
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate chest position (approx 30% down from top of bounding box)
                chest_x = int((x1 + x2) / 2)
                chest_y = int(y1 + (y2 - y1) * 0.3)
                
                # Calculate head position (approx 5% down from top)
                head_x = chest_x
                head_y = int(y1 + (y2 - y1) * 0.05)
                
                # Draw chest point
                cv2.circle(frame, (chest_x, chest_y), 8, (0, 255, 0), -1)
                cv2.circle(frame, (chest_x, chest_y), 4, (255, 255, 255), -1)
                
                # Draw small white point on head
                cv2.circle(frame, (head_x, head_y), 3, (255, 255, 255), -1)

        # Display total count
        cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Active Persons in Room: {people_count}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        display_frame = frame
        if width > 1000:
            display_frame = cv2.resize(frame, (1000, int(1000 * height / width)))
            
        cv2.imshow('Room Occupancy Monitor', display_frame)

        if is_image:
            image_out_path = final_output_path.replace('.mp4', '.jpg') if final_output_path else 'room_output.jpg'
            cv2.imwrite(image_out_path, frame)
            print("Press any key in the image window to close it...")
            cv2.waitKey(0)
            break
        else:
            if out is not None:
                out.write(frame)
            
            loop_time = time.time() - loop_start
            wait_ms = max(1, int(1000 / fps) - int(loop_time * 1000))
            
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
                
        if not is_image:
            ret, frame = cap.read()
            if not ret:
                break
                
    if not is_image and cap is not None:
        cap.release()
    if out is not None:
        out.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print(f"Processing complete! Output saved to '{final_output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Room Occupancy: Count total active people in the room")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='room_output.mp4', help='Path to save output video')
    args = parser.parse_args()
    run(args.video, args.output)
