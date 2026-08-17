import os
import cv2
import yaml
import sys
import time

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.app.tracking.worker_tracker import WorkerTracker
from src.app.pose.pose_estimator import PoseEstimator
from src.app.workstation.workstation_manager import WorkstationManager
from src.app.activity.activity_scorer import ActivityScorer
from src.app.activity.state_machine import StateMachine
from src.app.database.database import DatabaseLogger

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def draw_debug_ui(frame, assignments, final_states):
    """Draw bounding boxes, statuses, and reasons on the frame"""
    colors = {
        "WORKING": (0, 255, 0), # Green
        "UNCERTAIN": (0, 255, 255), # Yellow
        "NOT_WORKING": (0, 0, 255) # Red
    }
    
    for ws_id, data in assignments.items():
        person = data.get("person")
        laptop = data.get("laptop")
        state_info = final_states.get(ws_id)
        
        if not state_info:
            continue
            
        state = state_info["state"]
        color = colors.get(state, (255, 255, 255))
        score = state_info["score"]
        
        # Draw Person
        if person:
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Worker {ws_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{state} ({score:.0f}%)", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw Pose
            kpts = person.get("keypoints", [])
            for kpt in kpts:
                kx, ky, kconf = kpt
                if kconf > 0.5:
                    cv2.circle(frame, (int(kx), int(ky)), 3, (255, 0, 0), -1)
                    
        # Draw Laptop
        if laptop:
            lx1, ly1, lx2, ly2 = laptop["bbox"]
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
            
        # Draw status reasons on top left (if this was a real UI, we'd render a dashboard)
        y_offset = 30
        cv2.putText(frame, f"WS: {ws_id} | {state} | {score:.0f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 25
        for reason in state_info["reasons"]:
            cv2.putText(frame, reason, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

def main():
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    config = load_config(config_path)
    
    # Initialize DB (make sure dir exists)
    db_path = os.path.join(base_dir, config['logging']['db_path'])
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_logger = DatabaseLogger(db_path)
    
    # Init Models
    tracker = WorkerTracker(
        model_path=os.path.join(base_dir, config['models']['detection']),
        tracker_yaml=config['models']['tracking'],
        conf_thresholds={
            "person_conf": config['models']['person_conf'],
            "laptop_conf": config['models']['laptop_conf'],
            "phone_conf": config['models']['phone_conf']
        },
        device=config['models']['device']
    )
    
    pose_estimator = PoseEstimator(
        model_path=os.path.join(base_dir, config['models']['pose']),
        device=config['models']['device']
    )
    
    # Init Logic
    ws_manager = WorkstationManager(config['workstation'])
    activity_scorer = ActivityScorer(config['scoring'])
    # Need to pass thresholds into scorer from config directly or update ActivityScorer init
    activity_scorer.thresholds = config['thresholds'] 
    
    state_machine = StateMachine(config['temporal'])
    
    # Video Source
    stream_url = config['camera']['stream_url']
    if stream_url == "0": stream_url = 0
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"Failed to open video source {stream_url}")
        return
        
    print("Starting Worker Monitoring Pipeline...")
    
    frame_count = 0
    process_every_n_frames = int(config['camera']['fps'] / config['temporal']['fps_processing'])
    if process_every_n_frames < 1: process_every_n_frames = 1
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % process_every_n_frames != 0:
                continue # Skip frames to maintain target processing FPS
                
            # 1. Detection and Tracking
            tracked_objects = tracker.track(frame)
            
            # 2. Pose Estimation
            # Note: We run pose estimation separately here for modularity.
            # In a real deployed edge scenario, we'd fuse these models to save GPU compute.
            persons_with_pose = pose_estimator.estimate(frame)
            
            # Map pose to tracked persons based on bounding box overlap (IoU)
            from src.app.workstation.workstation_manager import calculate_iou
            for tracked_p in tracked_objects['person']:
                best_iou = 0
                best_pose = None
                for pose_p in persons_with_pose:
                    iou = calculate_iou(tracked_p['bbox'], pose_p['bbox'])
                    if iou > 0.5 and iou > best_iou:
                        best_iou = iou
                        best_pose = pose_p
                
                if best_pose:
                    tracked_p['keypoints'] = best_pose['keypoints']
                    tracked_p['posture'] = best_pose['posture']
                    
            # 3. Workstation Association
            assignments = ws_manager.associate(tracked_objects['person'], tracked_objects['laptop'])
            
            # 4. Activity Scoring & State Machine Update
            final_states = {}
            for ws_id, data in assignments.items():
                # Score
                score_result = activity_scorer.score_workstation(data)
                
                # Update temporal state
                final_state = state_machine.update(
                    ws_id=ws_id,
                    raw_state=score_result['raw_state'],
                    score=score_result['score'],
                    reasons=score_result['reasons'],
                    db_logger=db_logger
                )
                final_states[ws_id] = final_state
                
            # 5. UI/Debug Output
            draw_debug_ui(frame, assignments, final_states)
            
            # Show (Only for local testing, remove for headless server deployment)
            cv2.imshow("Worker Monitor Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline shut down gracefully.")

if __name__ == '__main__':
    main()
