class ActivityScorer:
    def __init__(self, config_data):
        self.config = config_data.get("activity", {})
        
    def calculate_score(self, worker_state, env_detections):
        """
        Calculates an activity score based on the user's rules.
        """
        score = 0
        reasons = []
        
        # Positive signals
        score += 20
        reasons.append("Person detected (+20)")
        
        has_laptop = False
        for l in env_detections["laptops"]:
            # check proximity to worker
            score += 20
            reasons.append("Laptop detected near (+20)")
            has_laptop = True
            break
            
        if worker_state.get("workstation") is not None:
            score += 10
            reasons.append(f"Inside workstation {worker_state['workstation']} (+10)")
            
        posture = worker_state.get("posture", "UNKNOWN")
        if posture == "SITTING":
            score += 20
            reasons.append("Sitting posture (+20)")
            
        interaction = worker_state.get("interaction", "NONE")
        if interaction == "ACTIVE":
            score += 15
            reasons.append("Active interaction (+15)")
            
        # Negative signals
        if posture == "STANDING":
            score -= 40
            reasons.append("Standing (-40)")
            
        if worker_state.get("workstation") is None:
            score -= 40
            reasons.append("Outside workstation (-40)")
            
        if not has_laptop:
            score -= 30
            reasons.append("No laptop near (-30)")
            
        return max(0, min(100, score)), reasons
