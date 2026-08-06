import cv2
import numpy as np
import time
import sys
import os
import database

from src.core.video import initialize_video_capture, initialize_video_writer
from src.core.models import load_yolo_model
from src.utils.drawing import is_inside_polygon, get_interactive_roi

def run(video_source='0', output_path='queue_output.mp4'):
    print("Loading YOLOv8 model for precise people counting...")
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

    roi_polygon = get_interactive_roi(first_frame)

    if not is_image:
        frame = first_frame
        
    print(f"Processing '{video_source}'...")
    last_alarm_time = 0
    last_people_in_line = 0
    events_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'events')
    os.makedirs(events_dir, exist_ok=True)
    
    while True:
        loop_start = time.time()
        
        results = model(frame, classes=[PERSON_CLASS], verbose=False)
        
        cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 255, 0), thickness=2)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_polygon], (255, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "QUEUE ZONE", (int(width * 0.3), int(height * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        people_in_line = 0
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu()
            confidences = results[0].boxes.conf.cpu()
            
            for box, conf in zip(boxes, confidences):
                if conf < 0.4:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                feet_x = int((x1 + x2) / 2)
                feet_y = y2
                
                in_line = is_inside_polygon((feet_x, feet_y), roi_polygon)
                
                if in_line:
                    people_in_line += 1
                    color = (0, 255, 0)
                    label = "In Line"
                else:
                    color = (0, 0, 255)
                    label = "Ignoring"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (feet_x, feet_y), 5, color, -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        count_color = (0, 255, 0) if people_in_line <= 7 else (0, 0, 255)
        cv2.putText(frame, f"People in Line: {people_in_line}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, count_color, 3)
        
        if people_in_line > 7:
            cv2.putText(frame, "ALARM: QUEUE TOO LONG!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if time.time() - last_alarm_time > 2.0:
                import os as _os; _os.system("afplay /System/Library/Sounds/Ping.aiff &")
                last_alarm_time = time.time()
                
        if people_in_line != last_people_in_line:
            current_time = time.time()
            img_filename = f"{int(current_time)}_queue_{people_in_line}.jpg"
            img_path_full = os.path.join(events_dir, img_filename)
            cv2.imwrite(img_path_full, frame)
            
            relative_path = f"/static/events/{img_filename}"
            person_count_str = f"Line 1: {people_in_line}"
            database.log_queue(person_count_str, relative_path)
            print(f"Queue count changed. Logged: {person_count_str}")
            
        last_people_in_line = people_in_line
        
        display_frame = frame
        if width > 1000:
            display_frame = cv2.resize(frame, (1000, int(1000 * height / width)))
            
        cv2.imshow('Precise Queue Monitor', display_frame)

        if is_image:
            image_out_path = final_output_path.replace('.mp4', '.jpg') if final_output_path else 'queue_output.jpg'
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
    parser = argparse.ArgumentParser(description="Queue Monitor: Count people standing in a specific line/area")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='queue_output.mp4', help='Path to save output video')
    args = parser.parse_args()
    run(args.video, args.output)
