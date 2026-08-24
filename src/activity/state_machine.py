import time

class StateMachine:
    def __init__(self, config_data):
        cfg = config_data.get("activity", {})
        self.working_conf = cfg.get("working_confirmation_seconds", 2.0)
        self.not_working_conf = cfg.get("not_working_confirmation_seconds", 1.0)
        self.score_thresh_w = cfg.get("score_thresholds", {}).get("working", 70)
        self.score_thresh_nw = cfg.get("score_thresholds", {}).get("uncertain", 40)
        
    def update_state(self, worker_id, current_state_dict, new_score, dt):
        """
        Updates the worker's status based on temporal smoothing.
        Returns the new final status and state string.
        """
        # Determine raw target state based on score
        if new_score >= self.score_thresh_w:
            target_state = "WORKING"
        elif new_score >= self.score_thresh_nw:
            target_state = "UNCERTAIN"
        else:
            target_state = "NOT_WORKING"
            
        current_status = current_state_dict.get("status", "UNCERTAIN")
        pending_status = current_state_dict.get("pending_status", target_state)
        pending_time = current_state_dict.get("pending_time", 0.0)
        
        if target_state == pending_status:
            pending_time += dt
        else:
            pending_status = target_state
            pending_time = 0.0
            
        # Check if we should transition
        transitioned = False
        if pending_status == "WORKING" and pending_time >= self.working_conf:
            if current_status != "WORKING":
                current_status = "WORKING"
                transitioned = True
        elif pending_status == "NOT_WORKING" and pending_time >= self.not_working_conf:
            if current_status != "NOT_WORKING":
                current_status = "NOT_WORKING"
                transitioned = True
        elif pending_status == "UNCERTAIN" and pending_time >= 1.0: # 1s confirm for uncertain
            if current_status != "UNCERTAIN":
                current_status = "UNCERTAIN"
                transitioned = True
                
        # Update dictionary
        current_state_dict["status"] = current_status
        current_state_dict["pending_status"] = pending_status
        current_state_dict["pending_time"] = pending_time
        
        return current_status, transitioned
