import cv2
import numpy as np

def draw_banner(frame, text, color):
    (h, w) = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), color, -1)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.5
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    text_x = (w - text_width) // 2
    text_y = 50
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

def is_inside_polygon(point, polygon):
    # cv2.pointPolygonTest returns positive if inside, negative if outside, 0 if on the edge
    return cv2.pointPolygonTest(polygon, point, False) >= 0

drawing_points = []
def _draw_polygon_callback(event, x, y, flags, param):
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))

def get_interactive_roi(frame):
    global drawing_points
    drawing_points = []
    clone = frame.copy()
    window_name = "Draw Box (Click points, press ENTER to finish)"
    
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1200:
        scale = 1200 / w
        clone = cv2.resize(clone, (1200, int(h * scale)))
        
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _draw_polygon_callback)
    
    print("Please draw the polygon by clicking on the image.")
    print("Press ENTER when you are done.")
    
    while True:
        display_frame = clone.copy()
        for pt in drawing_points:
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
        if len(drawing_points) > 1:
            cv2.polylines(display_frame, [np.array(drawing_points)], isClosed=False, color=(0, 255, 255), thickness=2)
        if len(drawing_points) > 2:
            cv2.polylines(display_frame, [np.array(drawing_points)], isClosed=True, color=(0, 255, 255), thickness=1)
            
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('q'): # Enter or q
            break
            
    cv2.destroyWindow(window_name)
    
    if len(drawing_points) < 3:
        print("Not enough points drawn. Using default box.")
        return np.array([
            [int(w * 0.25), int(h * 0.2)],
            [int(w * 0.75), int(h * 0.2)],
            [int(w * 0.85), int(h * 0.9)],
            [int(w * 0.15), int(h * 0.9)]
        ], np.int32)
        
    return np.array([(int(pt[0]/scale), int(pt[1]/scale)) for pt in drawing_points], np.int32)
