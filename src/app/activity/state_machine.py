import time
from datetime import datetime

class StateMachine:
    def __init__(self, config):
        self.config = config
        # Map: ws_id -> { 'state': str, 'last_change_time': float, 'history': [] }
        self.states = {}
        
        self.t_working = config.get('working_confirmation_seconds', 2)
        self.t_not_working = config.get('not_working_confirmation_seconds', 1)
        self.t_uncertain = config.get('uncertain_timeout_seconds', 5)
        
    def update(self, ws_id, raw_state, score, reasons, db_logger=None):
        """
        Updates the temporal state machine for a given workstation.
        Prevents flickering by requiring sustained evidence.
        """
        now = time.time()
        
        if ws_id not in self.states:
            self.states[ws_id] = {
                'current_state': 'UNCERTAIN',
                'target_state': raw_state,
                'state_start_time': now,
                'target_start_time': now
            }
            
        sm = self.states[ws_id]
        
        # If the incoming raw state changed, reset the target timer
        if sm['target_state'] != raw_state:
            sm['target_state'] = raw_state
            sm['target_start_time'] = now
            
        elapsed = now - sm['target_start_time']
        
        # Determine if we should transition
        transition = False
        if sm['current_state'] != sm['target_state']:
            if sm['target_state'] == 'WORKING' and elapsed >= self.t_working:
                transition = True
            elif sm['target_state'] == 'NOT_WORKING' and elapsed >= self.t_not_working:
                transition = True
            elif sm['target_state'] == 'UNCERTAIN' and elapsed >= self.t_uncertain:
                transition = True
                
        # Execute transition
        if transition:
            old_state = sm['current_state']
            sm['current_state'] = sm['target_state']
            sm['state_start_time'] = now
            
            if db_logger:
                db_logger.log_transition(ws_id, old_state, sm['current_state'], score, reasons)
                
        time_in_state = now - sm['state_start_time']
                
        return {
            "state": sm['current_state'],
            "time_in_state": time_in_state,
            "raw_state": raw_state,
            "score": score,
            "reasons": reasons
        }
