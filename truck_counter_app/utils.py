import cv2
import csv
import os
import time

# Colors (BGR format for OpenCV)
COLOR_BOX = (0, 255, 0)      # Green
COLOR_SACK = (255, 0, 0)     # Blue
COLOR_LINE = (0, 255, 255)   # Yellow
COLOR_LOAD = (0, 255, 0)     # Green
COLOR_UNLOAD = (0, 0, 255)   # Red
COLOR_TEXT = (255, 255, 255) # White

def draw_bbox(frame, bbox, label, color, thickness=2, show_conf=True, conf=0.0):
    x1, y1, x2, y2 = map(int, bbox)
    # Anti-aliased rectangle not natively supported by cv2.rectangle for the box, but we use LINE_AA for text
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    text = label
    if show_conf:
        text += f" {conf:.2f}"
        
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    
def draw_line(frame, pt1, pt2, color=COLOR_LINE, thickness=2):
    if pt1 and pt2:
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
        # Draw small circles at endpoints
        cv2.circle(frame, pt1, 5, color, -1)
        cv2.circle(frame, pt2, 5, color, -1)

def draw_dashboard(frame, loading, unloading, fps, active_objects, settings):
    h, w = frame.shape[:2]
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, f"Loading: {loading}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LOAD, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Unloading: {unloading}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_UNLOAD, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Boxes/Sacks: {loading + unloading}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} | Active: {active_objects}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)


def log_event(csv_path, tracking_id, cls_name, direction, event_type):
    # event_type: "Loading" or "Unloading"
    file_exists = os.path.isfile(csv_path)
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Time", "Tracking ID", "Object Class", "Direction", "Action"])
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, tracking_id, cls_name, direction, event_type])
    except Exception as e:
        print(f"Failed to log to CSV: {e}")
