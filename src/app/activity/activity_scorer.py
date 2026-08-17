class ActivityScorer:
    def __init__(self, config):
        self.positive_scores = config.get('positive', {})
        self.negative_scores = config.get('negative', {})
        self.thresholds = config.get('thresholds', {})
        
    def score_workstation(self, ws_assignment, pose_data=None):
        """
        Calculates the activity score based on the signals.
        ws_assignment: dict with 'person' and 'laptop' 
        pose_data: dict mapping person bbox/id to posture info
        """
        person = ws_assignment["person"]
        laptop = ws_assignment["laptop"]
        
        score = 0
        reasons = []
        
        if person:
            score += self.positive_scores.get('person_detected', 20)
            reasons.append("✓ Person detected")
            
            score += self.positive_scores.get('inside_workstation', 10)
            reasons.append("✓ Inside workstation")
            
            # Posture from pose_data
            posture = person.get('posture', 'UNKNOWN')
            if posture == 'SITTING':
                score += self.positive_scores.get('sitting_posture', 20)
                reasons.append("✓ Sitting")
            elif posture == 'STANDING':
                score += self.negative_scores.get('standing', -40)
                reasons.append("✗ Standing")
                
            # TODO: add hand interactions using pose keypoints
        else:
            score += self.negative_scores.get('outside_workstation', -40)
            reasons.append("✗ Outside workstation / Not detected")

        if laptop:
            score += self.positive_scores.get('laptop_detected', 20)
            reasons.append("✓ Laptop detected")
        else:
            # If laptop zone is visible but no laptop detected
            score += self.negative_scores.get('laptop_absent', -30)
            reasons.append("✗ Laptop absent")
            
        # Bound score between 0 and 100
        score = max(0, min(100, score))
        
        # Determine raw state
        if score >= self.thresholds.get('working_min', 70):
            raw_state = "WORKING"
        elif score >= self.thresholds.get('uncertain_min', 40):
            raw_state = "UNCERTAIN"
        else:
            raw_state = "NOT_WORKING"
            
        return {
            "score": score,
            "raw_state": raw_state,
            "reasons": reasons
        }
