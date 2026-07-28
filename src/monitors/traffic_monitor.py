import cv2
import numpy as np
from collections import defaultdict
import time

from src.core.video import initialize_video_capture, initialize_video_writer
from src.core.models import load_yolo_model

def run(video_source='0', output_path='traffic_output.mp4'):
    print("Loading YOLOv8 model...")
    model = load_yolo_model('yolov8s.pt') 
    vehicle_classes = [2, 3, 5, 7]

    frame, cap, width, height, fps, is_image = initialize_video_capture(video_source)
    if width == 0 or is_image:
        print("Traffic monitor requires a video stream.")
        return

    out, final_output_path = initialize_video_writer(output_path, width, height, fps)

    track_history = defaultdict(list)
    upward_count = 0
    counted_ids = set()
    lane_changers = set()
    
    COUNTING_LINE_Y = int(height * 0.5)
    LANE_LINES_X = [int(width * 0.5)]
    HISTORY_LENGTH = 10 

    print(f"Processing video '{video_source}'...")
    
    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(frame, persist=True, classes=vehicle_classes, verbose=False)
        
        cv2.line(frame, (0, COUNTING_LINE_Y), (width, COUNTING_LINE_Y), (0, 255, 255), 2)
        cv2.putText(frame, "Counting Line", (10, COUNTING_LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        for lx in LANE_LINES_X:
            cv2.line(frame, (lx, COUNTING_LINE_Y), (lx, height), (255, 255, 0), 2)
            cv2.putText(frame, "Lane", (lx + 5, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                history = track_history[track_id]
                history.append((cx, cy))
                
                if len(history) > HISTORY_LENGTH:
                    history.pop(0)
                    
                is_lane_changing = track_id in lane_changers
                
                if len(history) >= 2:
                    prev_cx, prev_cy = history[-2]
                    curr_cx, curr_cy = history[-1]
                    
                    if prev_cy > COUNTING_LINE_Y and curr_cy <= COUNTING_LINE_Y:
                        if track_id not in counted_ids:
                            upward_count += 1
                            counted_ids.add(track_id)
                
                if not is_lane_changing and len(history) >= 2:
                    prev_cx = history[-2][0]
                    curr_cx = history[-1][0]
                    
                    for lx in LANE_LINES_X:
                        if curr_cy > COUNTING_LINE_Y or prev_cy > COUNTING_LINE_Y:
                            if x1 <= lx <= x2:
                                is_lane_changing = True
                                lane_changers.add(track_id)
                                break
                
                color = (0, 0, 255) if is_lane_changing else (0, 255, 0)
                label = f"ID: {track_id}"
                if is_lane_changing:
                    label += " - LANE CHANGE"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                points = np.array(history).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Upward Count: {upward_count}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if out is not None:
            out.write(frame)
        
        display_frame = frame
        if width > 1000:
            display_frame = cv2.resize(frame, (1000, int(1000 * height / width)))
        cv2.imshow('Traffic Monitor', display_frame)
        
        loop_time = time.time() - loop_start
        wait_ms = max(1, int(1000 / fps) - int(loop_time * 1000))
        
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break
            
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete! Output saved to '{final_output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Traffic Monitor: Count cars upwards and detect lane changes")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='traffic_output.mp4', help='Path to save output video')
    args = parser.parse_args()
    run(args.video, args.output)
