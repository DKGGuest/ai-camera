import cv2
import numpy as np
import time

from src.core.video import initialize_video_capture, initialize_video_writer
from src.core.models import load_yolo_model

def run(video_source='0', output_path='adaptive_output.mp4'):
    print("Loading YOLOv8 model for human detection...")
    model = load_yolo_model('yolov8s.pt') 
    PERSON_CLASS = 0

    frame, cap, width, height, fps, is_image = initialize_video_capture(video_source)
    if width == 0:
        return

    out, final_output_path = initialize_video_writer(output_path, width, height, fps)

    print(f"Processing '{video_source}'...")
    if not is_image:
        print(f"Live monitoring started. Will save final video to: {final_output_path}")
    
    # Adaptive Streaming State
    active_mode = False
    last_human_seen_time = 0
    COOLDOWN_SECONDS = 5.0 # Stay in High Quality for 5 seconds after a person leaves
    
    # Idle settings
    LOW_RES_WIDTH = 320
    LOW_RES_HEIGHT = int(320 * (height / width)) if width > 0 else 240
    
    while True:
        loop_start = time.time()
        
        if not is_image:
            ret, frame = cap.read()
            if not ret:
                break
                
        detect_frame = cv2.resize(frame, (LOW_RES_WIDTH, LOW_RES_HEIGHT))
        results = model(detect_frame, classes=[PERSON_CLASS], verbose=False)
        
        human_detected = False
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for conf in results[0].boxes.conf.cpu():
                if conf > 0.4:
                    human_detected = True
                    break
                    
        current_time = time.time()
        if human_detected:
            last_human_seen_time = current_time
            active_mode = True
        elif current_time - last_human_seen_time > COOLDOWN_SECONDS:
            active_mode = False
            
        output_frame = frame.copy()
        
        if not active_mode:
            pixelated = cv2.resize(output_frame, (LOW_RES_WIDTH, LOW_RES_HEIGHT), interpolation=cv2.INTER_NEAREST)
            output_frame = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)
            
            status_text = "IDLE - LOW BITRATE STREAM"
            status_color = (0, 0, 255)
            bitrate_text = "~150 kbps"
        else:
            status_text = "ACTIVE - FULL HD HIGH BITRATE"
            status_color = (0, 255, 0)
            bitrate_text = "~6000 kbps"
            
            full_results = model(output_frame, classes=[PERSON_CLASS], verbose=False)
            if full_results[0].boxes is not None:
                for box, conf in zip(full_results[0].boxes.xyxy.cpu(), full_results[0].boxes.conf.cpu()):
                    if conf > 0.4:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(output_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.rectangle(output_frame, (10, 10), (550, 100), (0, 0, 0), -1)
        cv2.putText(output_frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(output_frame, f"Simulated Data Usage: {bitrate_text}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if is_image:
            image_out_path = final_output_path.replace('.mp4', '.jpg') if final_output_path else 'adaptive_output.jpg'
            cv2.imwrite(image_out_path, output_frame)
            print("Press any key in the image window to close it...")
            
            display_frame = output_frame
            if width > 1000:
                display_frame = cv2.resize(output_frame, (1000, int(1000 * height / width)))
            cv2.imshow('Adaptive Stream Model', display_frame)
            cv2.waitKey(0)
            break
        else:
            if out is not None:
                out.write(output_frame)
            
            display_frame = output_frame
            if width > 1000:
                display_frame = cv2.resize(output_frame, (1000, int(1000 * height / width)))
            cv2.imshow('Adaptive Stream Model', display_frame)
            
            loop_time = time.time() - loop_start
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
    parser = argparse.ArgumentParser(description="Adaptive Stream: Dynamically adjust bitrate and resolution based on human presence")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='adaptive_output.mp4', help='Path to save output video')
    args = parser.parse_args()
    run(args.video, args.output)
