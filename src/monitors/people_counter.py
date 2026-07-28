import cv2
import time
from ultralytics import YOLO

from src.core.video import initialize_video_capture, initialize_video_writer
from src.core.models import load_yolo_model
from src.core.edge_client import EdgeClient

def run(video_source='0', output_path='output_counted_video.mp4', frame_callback=None):
    print("Initializing Commercial People Counter...")
    # Using Nano for speed on edge devices
    model = load_yolo_model('yolov8n.pt')
    
    # Connect this camera node to the central server!
    edge_client = EdgeClient(server_url="http://127.0.0.1:5000", camera_id="Factory_Floor_Camera_2")

    frame, cap, width, height, fps, is_image = initialize_video_capture(video_source)
    if width == 0 or is_image:
        print("Error: People Counter requires a video stream.")
        return

    out, final_output_path = initialize_video_writer(output_path, width, height, fps)

    # Horizontal counting line in the middle of the screen
    line_y = height // 2

    # Buffer zone around the line to stop bounding-box jitter from
    # flickering across line_y and causing repeated/false counts.
    buffer_px = 30
    upper_bound = line_y - buffer_px
    lower_bound = line_y + buffer_px

    tracked_zone = {}   # track_id -> 'above' or 'below' (last CONFIRMED zone)
    entry_count = 0
    exit_count = 0
    
    # To avoid spamming the server, we only send an update for an ID once every 2 seconds
    last_broadcast_time = {}

    print(f"Processing video '{video_source}'...")
    
    while True:
        loop_start = time.time()
        ret, frame_read = cap.read()
        if not ret:
            break
            
        frame = frame_read
        
        # Use YOLO's built-in ByteTrack
        results = model.track(frame, persist=True, verbose=False, conf=0.35)
        
        # Draw the counting line + buffer boundaries
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
        cv2.line(frame, (0, upper_bound), (width, upper_bound), (0, 255, 255), 1)
        cv2.line(frame, (0, lower_bound), (width, lower_bound), (0, 255, 255), 1)
        cv2.putText(frame, "ZONE A (ENTRIES)", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "ZONE B (EXITS)", (10, line_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.id is None:
                continue
                
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                # We only care about people
                if cls_name != 'person':
                    continue
                    
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # --- EDGE COMPUTING BROADCAST ---
                # Periodically tell the central server where this tracked ID is
                now = time.time()
                if track_id not in last_broadcast_time or (now - last_broadcast_time[track_id] > 2.0):
                    edge_client.event_queue.append({
                        "camera_id": edge_client.camera_id,
                        "event_type": "person_tracked",
                        "track_id": track_id,
                        "location": {"x": cx, "y": cy}
                    })
                    last_broadcast_time[track_id] = now
                
                # Determine which side of the buffer this person is
                # clearly on. Inside the buffer itself, we don't know yet
                # (this is what stops the flicker/false-count bug).
                if cy < upper_bound:
                    current_zone = 'above'
                elif cy > lower_bound:
                    current_zone = 'below'
                else:
                    current_zone = None  # still inside the buffer

                if track_id not in tracked_zone:
                    # First time we see this ID: only record a zone once
                    # it's clearly on one side (don't guess from inside buffer)
                    if current_zone is not None:
                        tracked_zone[track_id] = current_zone
                else:
                    prev_zone = tracked_zone[track_id]
                    if current_zone is not None and current_zone != prev_zone:
                        # Person has fully crossed from one confirmed zone
                        # to the other confirmed zone -> count exactly once
                        if prev_zone == 'above' and current_zone == 'below':
                            entry_count += 1
                            edge_client.event_queue.append({"camera_id": edge_client.camera_id, "event_type": "zone_entry"})
                        elif prev_zone == 'below' and current_zone == 'above':
                            exit_count += 1
                            edge_client.event_queue.append({"camera_id": edge_client.camera_id, "event_type": "zone_exit"})
                        tracked_zone[track_id] = current_zone
                    # if current_zone is None (in buffer) or unchanged,
                    # do nothing - this is what prevents double counting
                
                # Visuals
                color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id}", (x1, max(y1 - 10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw counters
        cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"ENTRIES: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(frame, f"EXITS: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if out is not None:
            out.write(frame)

        if frame_callback is not None:
            if not frame_callback(frame, {"entries": entry_count, "exits": exit_count}):
                break
        else:
            display_frame = frame
            if width > 800:
                display_frame = cv2.resize(frame, (800, int(800 * height / width)))
            cv2.imshow('Commercial People Counter', display_frame)
            
            loop_time = time.time() - loop_start
            wait_ms = max(1, int(1000 / fps) - int(loop_time * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

    if cap is not None:
        cap.release()
    if out is not None:
        out.release()
    edge_client.stop()
    cv2.destroyAllWindows()
    print(f"Processing complete! Final Count -> Entries: {entry_count} | Exits: {exit_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Commercial People Counter")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='output_counted_video.mp4', help='Path to save output video')
    args = parser.parse_args()
    run(args.video, args.output)
