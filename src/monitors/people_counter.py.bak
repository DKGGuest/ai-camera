import cv2

from src.core.video import initialize_video_capture, initialize_video_writer
from src.core.models import load_yolo_model

# We use a dummy Tracker for the code that wasn't fully implemented
class Tracker:
    def update(self, current_detections):
        objects = {}
        for idx, det in enumerate(current_detections):
            x1, y1, x2, y2 = det
            objects[idx] = ((x1 + x2) // 2, (y1 + y2) // 2)
        return objects

def run(video_source='0', output_path='output_counted_video.mp4'):
    print("Loading YOLOv8 model...")
    model = load_yolo_model('yolov8s.pt')
    
    frame, cap, frame_width, frame_height, fps, is_image = initialize_video_capture(video_source)
    if frame_width == 0 or is_image:
        print("People counter requires a video stream.")
        return

    out, final_output_path = initialize_video_writer(output_path, frame_width, frame_height, fps)

    line_start_x = 0  
    line_end_x = frame_width 
    counting_line_y = frame_height // 3 
    
    buffer_pixels = 40 
    upper_line_y = counting_line_y - buffer_pixels
    lower_line_y = counting_line_y + buffer_pixels

    tracker = Tracker()
    object_state = {}

    entry_count = 0
    exit_count = 0

    print("Starting video processing with tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=False, verbose=False)

        current_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if model.names[int(box.cls[0])] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_detections.append((x1, y1, x2, y2))

        objects = tracker.update(current_detections)

        current_tracker_ids = set(objects.keys())
        for obj_id in list(object_state.keys()):
            if obj_id not in current_tracker_ids:
                del object_state[obj_id]

        for object_id, centroid in objects.items():
            centroid_x, centroid_y = centroid

            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 255), -1)

            actual_centroid_zone = None
            actual_centroid_zone_text = "Unknown"

            if centroid_y < upper_line_y:
                actual_centroid_zone = 'above'
                actual_centroid_zone_text = 'Above'
            elif centroid_y > lower_line_y:
                actual_centroid_zone = 'below'
                actual_centroid_zone_text = 'Below'
            else:
                actual_centroid_zone = 'in_buffer'
                actual_centroid_zone_text = 'Buffer'

            if not (line_start_x <= centroid_x <= line_end_x):
                 cv2.putText(frame, f"ID:{object_id} (Out of Bounds)", (centroid_x - 10, centroid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                 continue 

            if object_id not in object_state: 
                if actual_centroid_zone == 'above':
                    object_state[object_id] = 'above_ready_to_enter'
                elif actual_centroid_zone == 'below':
                    object_state[object_id] = 'below_ready_to_exit'
                else: 
                    object_state[object_id] = 'pending_in_buffer'
            
            current_obj_state = object_state[object_id]

            if current_obj_state == 'pending_in_buffer':
                if actual_centroid_zone == 'above':
                    object_state[object_id] = 'above_ready_to_enter'
                elif actual_centroid_zone == 'below':
                    object_state[object_id] = 'below_ready_to_exit'

            elif current_obj_state == 'above_ready_to_enter':
                if actual_centroid_zone == 'below':
                    entry_count += 1
                    object_state[object_id] = 'crossing_down'

            elif current_obj_state == 'crossing_down':
                if actual_centroid_zone == 'above': 
                    object_state[object_id] = 'below_ready_to_exit' 

            elif current_obj_state == 'below_ready_to_exit':
                if actual_centroid_zone == 'above':
                    exit_count += 1
                    object_state[object_id] = 'crossing_up'

            elif current_obj_state == 'crossing_up':
                if actual_centroid_zone == 'below': 
                    object_state[object_id] = 'above_ready_to_enter' 

            cv2.putText(frame, f"ID:{object_id} ({actual_centroid_zone_text})", (centroid_x - 10, centroid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.line(frame, (line_start_x, counting_line_y), (line_end_x, counting_line_y), (255, 0, 0), 2) 
        cv2.line(frame, (line_start_x, upper_line_y), (line_end_x, upper_line_y), (0, 255, 255), 1) 
        cv2.line(frame, (line_start_x, lower_line_y), (line_end_x, lower_line_y), (0, 255, 255), 1) 
        
        cv2.putText(frame, f"Buffer Zone ({buffer_pixels}px)", (line_start_x + 5, counting_line_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(frame, f"Entries: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if out is not None:
            out.write(frame)

        cv2.imshow('People Counter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release() 
    cv2.destroyAllWindows()
    print(f"Processing finished. Total Entries: {entry_count}, Total Exits: {exit_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="People Counter")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='output_counted_video.mp4', help='Path to save output video')
    args = parser.parse_args()
    run(args.video, args.output)
