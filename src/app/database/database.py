import sqlite3
from datetime import datetime

class DatabaseLogger:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create transitions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            workstation_id TEXT,
            old_state TEXT,
            new_state TEXT,
            score REAL,
            reasons TEXT
        )
        ''')
        
        # Create continuous log table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            workstation_id TEXT,
            state TEXT,
            score REAL
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_transition(self, ws_id, old_state, new_state, score, reasons):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        reasons_str = "; ".join(reasons)
        
        cursor.execute('''
        INSERT INTO transitions (timestamp, workstation_id, old_state, new_state, score, reasons)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (now, ws_id, old_state, new_state, score, reasons_str))
        
        conn.commit()
        conn.close()
        
        # Also print to stdout per prompt requirement for logging
        print(f"{now} | WS: {ws_id} | {old_state} -> {new_state} | Score: {score:.1f} | Reasons: {reasons_str}")
