import os
import cv2
import time
import yaml

from src.detection.object_detector import ObjectDetector
from src.pose.pose_estimator import PoseEstimator
from src.tracking.worker_tracker import WorkerTracker
from src.workstation.workstation_manager import WorkstationManager
from src.activity.interaction_analyzer import InteractionAnalyzer
from src.activity.activity_scorer import ActivityScorer
from src.activity.state_machine import StateMachine

class WorkerMonitor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize modular components
        self.detector = ObjectDetector(
            model_path=self.config["models"]["detection"],
            conf_thresh=self.config["models"]["conf_threshold"],
            nms_thresh=self.config["models"]["nms_threshold"]
        )
        self.pose_estimator = PoseEstimator(
            model_path=self.config["models"]["pose"],
            conf_thresh=self.config["models"]["conf_threshold"]
        )
        self.tracker = WorkerTracker()
        self.workstation_mgr = WorkstationManager(self.config)
        self.interaction_analyzer = InteractionAnalyzer()
        self.scorer = ActivityScorer(self.config)
        self.state_machine = StateMachine(self.config)
        
        self.worker_states = {} # master state dictionary
        self.last_tick = time.time()
        
    def process_frame(self, frame):
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now
        
        # 1. Object Detection
        detections = self.detector.detect(frame)
        
        # 2. Tracking
        tracked_workers = self.tracker.update(detections["persons"])
        
        # 3. Pose Estimation (for all persons)
        poses = self.pose_estimator.estimate(frame)
        
        # 4. Activity Analysis Loop
        results = []
        for worker in tracked_workers:
            wid = worker["id"]
            box = worker["box"]
            
            # Find matching pose
            matched_pose = None
            for p in poses:
                # simple bounding box match
                pb = p["box"]
                # calculate IoU or just overlap to associate
                if box[0] <= pb[2] and box[2] >= pb[0] and box[1] <= pb[3] and box[3] >= pb[1]:
                    matched_pose = p
                    break
                    
            if wid not in self.worker_states:
                self.worker_states[wid] = {"status": "UNCERTAIN"}
                
            state_dict = self.worker_states[wid]
            
            # Environment association
            ws_id = self.workstation_mgr.get_workstation_for_worker(box)
            state_dict["workstation"] = ws_id
            
            if matched_pose:
                posture = self.pose_estimator.determine_posture(matched_pose)
                interaction = self.interaction_analyzer.calculate_interaction(
                    matched_pose, detections["laptops"], detections["keyboards"]
                )
            else:
                posture = "UNKNOWN"
                interaction = "UNKNOWN"
                
            state_dict["posture"] = posture
            state_dict["interaction"] = interaction
            
            # Scoring
            score, reasons = self.scorer.calculate_score(state_dict, detections)
            
            # State Machine transition
            final_status, changed = self.state_machine.update_state(wid, state_dict, score, dt)
            
            results.append({
                "id": wid,
                "box": box,
                "score": score,
                "reasons": reasons,
                "status": final_status,
                "posture": posture,
                "interaction": interaction,
                "workstation": ws_id
            })
            
        return results, detections

def run_pipeline():
    # Placeholder to test initialization
    print("WorkerMonitor modular architecture loaded successfully.")
