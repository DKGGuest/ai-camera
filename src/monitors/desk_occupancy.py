import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

def point_in_box(point, box, padding=0):
    """Check if a point (x, y) is inside a bounding box (x1, y1, x2, y2) with optional padding."""
    x, y = point
    x1, y1, x2, y2 = box
    return (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding)

def get_video_stream(rtsp_url):
    """Attempt to open the video stream and handle connection errors."""
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"Error: Could not connect to stream '{rtsp_url}'. Retrying in 5 seconds...")
        time.sleep(5)
        return None
    return cap

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points (p1 -> p2 -> p3), angle is at p2."""
    if p1[0] == 0 or p2[0] == 0 or p3[0] == 0:
        return 0
    
    # Vector 1: p2 -> p1
    v1 = [p1[0] - p2[0], p1[1] - p2[1]]
    # Vector 2: p2 -> p3
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    
    if mag1 * mag2 == 0:
        return 0
        
    angle_rad = math.acos(max(-1.0, min(1.0, dot_product / (mag1 * mag2))))
    return math.degrees(angle_rad)

class TrackedChair:
    def __init__(self, box):
        self.box = box
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]
        self.cx = (box[0] + box[2]) / 2.0
        self.cy = (box[1] + box[3]) / 2.0
        self.missed_frames = 0
        self.matched = True
        
    def update(self, box):
        # Update center with smoothing
        new_cx = (box[0] + box[2]) / 2.0
        new_cy = (box[1] + box[3]) / 2.0
        self.cx = self.cx * 0.7 + new_cx * 0.3
        self.cy = self.cy * 0.7 + new_cy * 0.3
        
        new_w = box[2] - box[0]
        new_h = box[3] - box[1]
        
        # Smooth width and height (allow shrinking and growing equally to prevent bloated overlapping boxes)
        self.w = self.w * 0.7 + new_w * 0.3
        self.h = self.h * 0.7 + new_h * 0.3
        
        self.box = (
            int(self.cx - self.w/2),
            int(self.cy - self.h/2),
            int(self.cx + self.w/2),
            int(self.cy + self.h/2)
        )
        self.missed_frames = 0

class TrackedPerson:
    def __init__(self, box, status):
        self.box = box
        self.cx = (box[0] + box[2]) / 2.0
        self.cy = (box[1] + box[3]) / 2.0
        self.status_history = [status]
        self.status = status
        self.missed_frames = 0
        self.matched = True
        
    def update(self, box, raw_status):
        self.cx = (box[0] + box[2]) / 2.0
        self.cy = (box[1] + box[3]) / 2.0
        self.box = box
        
        # Keep history of last 15 frames for stable output
        self.status_history.append(raw_status)
        if len(self.status_history) > 15:
            self.status_history.pop(0)
            
        # Majority vote
        sitting_count = self.status_history.count("Sitting")
        if sitting_count > len(self.status_history) / 2:
            self.status = "Sitting"
        else:
            self.status = "Standing"
            
        self.missed_frames = 0

def main():
    print("Loading YOLOv8m for chairs and YOLOv8m-pose for posture analysis...")
    model_chairs = YOLO("yolov8m.pt")
    model_pose = YOLO("yolov8m-pose.pt")
    
    rtsp_url = 'YOUR_RTSP_URL_HERE'
    
    PERSON_CONF_THRESHOLD = 0.35
    CHAIR_CONF_THRESHOLD = 0.08 # Lowered threshold to detect far chairs, relying on strict NMS
    CHAIR_PADDING = 60

    cap = get_video_stream(rtsp_url)
    while cap is None:
         cap = get_video_stream(rtsp_url)

    print("Starting video processing. Press 'q' to quit.")

    tracked_chairs = []
    tracked_people = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream lost! Attempting to reconnect...")
            cap.release()
            cap = None
            while cap is None:
                cap = get_video_stream(rtsp_url)
            continue
            
        def is_duplicate_box(boxA, boxB):
            # Proportional distance check to merge fragmented/adjacent bounding boxes for the same chair/person
            cA_x, cA_y = (boxA[0]+boxA[2])/2, (boxA[1]+boxA[3])/2
            cB_x, cB_y = (boxB[0]+boxB[2])/2, (boxB[1]+boxB[3])/2
            dx = abs(cA_x - cB_x)
            dy = abs(cA_y - cB_y)
            
            wA, hA = boxA[2] - boxA[0], boxA[3] - boxA[1]
            wB, hB = boxB[2] - boxB[0], boxB[3] - boxB[1]
            areaA = wA * hA
            areaB = wB * hB
            minArea = min(areaA, areaB)
            
            avg_w = (wA + wB) / 2.0
            avg_h = (hA + hB) / 2.0
            
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            
            if minArea <= 0: return False
            ioma = interArea / float(minArea) # Intersection over Minimum Area
            iou = interArea / float(areaA + areaB - interArea)
            
            # If the smaller box is mostly inside the larger box, it's a fragment (e.g. backrest, handle)
            if ioma > 0.4:
                return True
                
            # If centers are very close horizontally and vertically, they belong to the same chair
            if dx < avg_w * 0.25 and dy < avg_h * 0.4:
                return True
                
            # For similarly sized boxes, check standard IoU
            return iou > 0.55

        # 1. Run inference for chairs (Standard YOLO)
        # Increased IoU from 0.45 to 0.60 so adjacent/overlapping chairs in the back aren't deleted by YOLO NMS
        results_chairs = model_chairs(frame, classes=[56], conf=CHAIR_CONF_THRESHOLD, iou=0.60, imgsz=1280, verbose=False)
        raw_detected_chairs = []
        if results_chairs[0].boxes is not None:
            for box in results_chairs[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = (x2 - x1), (y2 - y1)
                
                if w > 300: # Filter unusually wide boxes (likely two merged chairs)
                    continue
                    
                raw_detected_chairs.append((x1, y1, x2, y2))
                
        # Sort by area descending so we keep the largest box (the full chair) and filter out the smaller fragments
        raw_detected_chairs.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
                
        # Filter raw detections to eliminate duplicate sub-boxes on the same chair
        filtered_detected_chairs = []
        for d_box in raw_detected_chairs:
            overlap = False
            for f_box in filtered_detected_chairs:
                if is_duplicate_box(d_box, f_box):
                    overlap = True
                    break
            if not overlap:
                filtered_detected_chairs.append(d_box)

        detected_chairs = []
        for (x1, y1, x2, y2) in filtered_detected_chairs:
            # Shrink chair bounding box area by ~50% to tighten around the chair
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2
            new_w = int(w * 0.75)
            new_h = int(h * 0.75)
            nx1 = cx - new_w // 2
            nx2 = cx + new_w // 2
            ny1 = cy - new_h // 2
            ny2 = cy + new_h // 2
            
            detected_chairs.append((nx1, ny1, nx2, ny2))
                
        # Update Tracked Chairs to prevent shrinking
        for tc in tracked_chairs:
            tc.matched = False
            
        for d_box in detected_chairs:
            dcx = (d_box[0] + d_box[2]) / 2
            dcy = (d_box[1] + d_box[3]) / 2
            
            best_tc = None
            best_dist = 80 # pixels tracking threshold
            
            for tc in tracked_chairs:
                dist = math.hypot(dcx - tc.cx, dcy - tc.cy)
                if dist < best_dist:
                    best_dist = dist
                    best_tc = tc
                    
            if best_tc is not None:
                best_tc.update(d_box)
                best_tc.matched = True
            else:
                tracked_chairs.append(TrackedChair(d_box))
                
        active_chairs = []
        chairs = [] # Array of (x1, y1, x2, y2) to pass to occupancy logic
        for tc in tracked_chairs:
            if not tc.matched:
                tc.missed_frames += 1
            
            # Keep the chair in memory even if it's completely occluded for up to 300 frames (~10s)
            if tc.missed_frames < 300: 
                active_chairs.append(tc)
                chairs.append(tc.box)
                
        tracked_chairs = active_chairs
                
        # 2. Run inference for people and poses (YOLO Pose)
        results_pose = model_pose(frame, conf=PERSON_CONF_THRESHOLD, imgsz=1280, verbose=False)
        people = [] 
        
        if results_pose[0].keypoints is not None and results_pose[0].boxes is not None:
            # Pre-filter duplicate people detections
            valid_pose_indices = []
            for i in range(len(results_pose[0].boxes)):
                boxA = list(map(int, results_pose[0].boxes[i].xyxy[0]))
                overlap = False
                for j in valid_pose_indices:
                    boxB = list(map(int, results_pose[0].boxes[j].xyxy[0]))
                    if is_duplicate_box(boxA, boxB):
                        overlap = True
                        break
                if not overlap:
                    valid_pose_indices.append(i)
                    
            for i in valid_pose_indices:
                box = results_pose[0].boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                keypoints = results_pose[0].keypoints.data[i]
                kp_conf_threshold = 0.3
                angle = 180 # Default to straight/standing if angle can't be calculated
                status = "Unknown"
                
                # Check keypoints (indices 5, 11, 13 for left side, 6, 12, 14 for right side)
                l_shoulder, l_hip, l_knee = keypoints[5], keypoints[11], keypoints[13]
                r_shoulder, r_hip, r_knee = keypoints[6], keypoints[12], keypoints[14]
                
                knees_visible = False
                
                # Calculate angle using the side that is most visible
                if l_shoulder[2] > kp_conf_threshold and l_hip[2] > kp_conf_threshold and l_knee[2] > kp_conf_threshold:
                    angle = calculate_angle(l_shoulder[:2], l_hip[:2], l_knee[:2])
                    knees_visible = True
                elif r_shoulder[2] > kp_conf_threshold and r_hip[2] > kp_conf_threshold and r_knee[2] > kp_conf_threshold:
                    angle = calculate_angle(r_shoulder[:2], r_hip[:2], r_knee[:2])
                    knees_visible = True
                        
                # Determine posture:
                is_lying_down = (x2 - x1) > (y2 - y1) * 0.8
                
                # Human-like logic: check if the person's hips are inside any tracked chair
                hips_in_chair = False
                hip_y = -1
                if l_hip[2] > 0.3 and r_hip[2] > 0.3:
                    hip_y = (l_hip[1] + r_hip[1]) / 2
                    hip_x = (l_hip[0] + r_hip[0]) / 2
                elif l_hip[2] > 0.3:
                    hip_y, hip_x = l_hip[1], l_hip[0]
                elif r_hip[2] > 0.3:
                    hip_y, hip_x = r_hip[1], r_hip[0]
                    
                if hip_y != -1:
                    for c_box in chairs:
                        cx1, cy1, cx2, cy2 = c_box
                        # Check if hip is within the chair box (with some padding)
                        if (cx1 - 20) < hip_x < (cx2 + 20) and (cy1 - 20) < hip_y < (cy2 + 20):
                            hips_in_chair = True
                            break
                
                if is_lying_down:
                    raw_status = "Sitting"
                elif hips_in_chair:
                    raw_status = "Sitting"
                elif knees_visible and 45 <= angle <= 165:
                    raw_status = "Sitting"
                elif not knees_visible and (y2 - y1) < (x2 - x1) * 2.2:
                    # Fallback: if knees are occluded (e.g., under desk) and bounding box is relatively square, assume sitting
                    raw_status = "Sitting"
                else:
                    raw_status = "Standing"
                    
                people.append({
                    'box': (x1, y1, x2, y2),
                    'raw_status': raw_status,
                    'center': (center_x, center_y),
                    'angle': angle,
                    'kp': keypoints,
                    'is_lying': is_lying_down
                })

        # Apply temporal smoothing to people to prevent flickering
        for tp in tracked_people:
            tp.matched = False
            
        for p in people:
            px, py = p['center']
            best_tp = None
            best_dist = 100
            
            for tp in tracked_people:
                dist = math.hypot(px - tp.cx, py - tp.cy)
                if dist < best_dist:
                    best_dist = dist
                    best_tp = tp
                    
            if best_tp is not None:
                best_tp.update(p['box'], p['raw_status'])
                best_tp.matched = True
                p['status'] = best_tp.status # Assign the smoothed status back
            else:
                new_tp = TrackedPerson(p['box'], p['raw_status'])
                tracked_people.append(new_tp)
                p['status'] = new_tp.status
                
        # Clean up lost people
        active_people = []
        for tp in tracked_people:
            if not tp.matched:
                tp.missed_frames += 1
            if tp.missed_frames < 30:
                active_people.append(tp)
        tracked_people = active_people

        # 3. Determine occupancy: Sitting or Standing people can occupy a desk
        occupied_chair_indices = set()
        
        # Draw people and collect them for matching
        candidate_people = []
        for p_idx, person in enumerate(people):
            # Define polygon points using all visible keypoints (head to toe)
            poly_points = []
            kp = person['kp']
            for kp_idx in range(17):
                if kp[kp_idx][2] > 0.3:
                    poly_points.append([int(kp[kp_idx][0]), int(kp[kp_idx][1])])
            
            # Draw the posture polygon instead of bounding boxes or points
            box_color = (255, 100, 100) if person['status'] == "Sitting" else (100, 255, 255)
            
            if len(poly_points) >= 3:
                pts = np.array(poly_points, np.int32).reshape((-1, 1, 2))
                hull = cv2.convexHull(pts)
                # cv2.polylines(frame, [hull], isClosed=True, color=box_color, thickness=3)
            
            label = f"{person['status']} ({int(person['angle'])} deg)"
            if person['is_lying']: label += " [Lying]"
                
            cv2.putText(frame, label, 
                        (person['box'][0], person['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            if person['status'] == "Sitting":
                candidate_people.append((p_idx, person))
                
        # Calculate matching based on intersection area (major portion of person in chair)
        matches = []
        for p_idx, person in candidate_people:
            px1, py1, px2, py2 = person['box']
            person_area = (px2 - px1) * (py2 - py1)
            
            for c_idx, chair_box in enumerate(chairs):
                cx1, cy1, cx2, cy2 = chair_box
                chair_area = (cx2 - cx1) * (cy2 - cy1)
                
                # Check bounding box intersection
                ix1 = max(px1, cx1)
                iy1 = max(py1, cy1)
                ix2 = min(px2, cx2)
                iy2 = min(py2, cy2)
                
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    
                    # Consider it a match candidate if overlap is substantial
                    if inter_area > 0.15 * min(person_area, chair_area):
                        iou = inter_area / float(person_area + chair_area - inter_area)
                        
                        # Distance between centers
                        pcx = (px1 + px2) / 2
                        pcy = (py1 + py2) / 2
                        ccx = (cx1 + cx2) / 2
                        ccy = (cy1 + cy2) / 2
                        dist = math.hypot(pcx - ccx, pcy - ccy)
                        
                        matches.append({
                            'p_idx': p_idx,
                            'c_idx': c_idx,
                            'inter_area': inter_area,
                            'iou': iou,
                            'dist': dist
                        })
                    
        # Sort matches by HIGHEST IoU, then lowest distance
        matches.sort(key=lambda x: (x['iou'], -x['dist']), reverse=True)
        
        matched_people = set()
        
        for match in matches:
            p_idx = match['p_idx']
            c_idx = match['c_idx']
            
            if p_idx not in matched_people and c_idx not in occupied_chair_indices:
                matched_people.add(p_idx)
                occupied_chair_indices.add(c_idx)
                # If a chair is occupied, reset its missed frames so it doesn't disappear
                tracked_chairs[c_idx].missed_frames = 0
            
        # 4. Draw chairs based on occupancy
        for i, chair_box in enumerate(chairs):
            cx1, cy1, cx2, cy2 = chair_box
            is_occupied = (i in occupied_chair_indices)
            
            if is_occupied:
                color = (0, 0, 255)  # Red for Occupied
                label = "Occupied"
            else:
                color = (0, 255, 0)  # Green for Empty
                label = "Empty"
                
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 3)
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (cx1, cy1 - text_height - 10), (cx1 + text_width, cy1), color, -1)
            cv2.putText(frame, label, (cx1, cy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        cv2.imshow("Desk Occupancy Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
