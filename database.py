import sqlite3
import threading
import time

import config

_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock:
        conn = _connect()
        # General Events Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                mode TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detail TEXT,
                status TEXT
            )
            """
        )
        
        # Access Logs Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                photo_path TEXT
            )
            """
        )
        
        # Adaptive Logs Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS adaptive_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                state TEXT NOT NULL,
                duration REAL NOT NULL,
                photo_path TEXT
            )
            """
        )
        
        # People Logs Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS people_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                direction TEXT NOT NULL,
                total_entries INTEGER NOT NULL,
                total_exits INTEGER NOT NULL,
                photo_path TEXT
            )
            """
        )
        
        # Queue Logs Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queue_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                person_count TEXT NOT NULL,
                photo_path TEXT
            )
            """
        )
        
        # Worker Logs Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                worker_name TEXT NOT NULL,
                work_time_s REAL NOT NULL,
                rest_time_s REAL NOT NULL,
                photo_path TEXT
            )
            """
        )
        
        # Box Logs Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS box_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                loaded_count INTEGER NOT NULL,
                unloaded_count INTEGER NOT NULL,
                photo_path TEXT
            )
            """
        )
        
        # Bag & Box Logs Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bag_box_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                boxes_in INTEGER NOT NULL,
                boxes_out INTEGER NOT NULL,
                bags_in INTEGER NOT NULL,
                bags_out INTEGER NOT NULL,
                photo_path TEXT
            )
            """
        )
        
        conn.commit()
        conn.close()


def cleanup_old_records():
    """Delete records older than 7 days (7 * 24 * 3600 seconds)"""
    cutoff = time.time() - (7 * 24 * 3600)
    tables = ["events", "access_logs", "adaptive_logs", "people_logs", "queue_logs", "worker_logs"]
    
    with _lock:
        conn = _connect()
        for table in tables:
            try:
                conn.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
            except Exception:
                pass
        conn.commit()
        conn.close()


def log_event(mode, event_type, detail="", status="info"):
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO events (ts, mode, event_type, detail, status) VALUES (?, ?, ?, ?, ?)",
            (time.time(), mode, event_type, detail, status),
        )
        conn.commit()
        conn.close()
    
    # Run cleanup periodically
    if int(time.time()) % 100 == 0:
        cleanup_old_records()


def clear_all_events():
    with _lock:
        conn = _connect()
        conn.execute("DELETE FROM events")
        conn.commit()
        conn.close()


def get_events(limit=50):
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


# Functions to insert into specific tables
def log_access(name, status, photo_path):
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO access_logs (ts, name, status, photo_path) VALUES (?, ?, ?, ?)",
            (time.time(), name, status, photo_path)
        )
        conn.commit()
        conn.close()

def log_adaptive(state, duration, photo_path):
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO adaptive_logs (ts, state, duration, photo_path) VALUES (?, ?, ?, ?)",
            (time.time(), state, duration, photo_path)
        )
        conn.commit()
        conn.close()

def log_people(direction, total_entries, total_exits, photo_path):
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO people_logs (ts, direction, total_entries, total_exits, photo_path) VALUES (?, ?, ?, ?, ?)",
            (time.time(), direction, total_entries, total_exits, photo_path)
        )
        conn.commit()
        conn.close()

def log_queue(person_count, photo_path):
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO queue_logs (ts, person_count, photo_path) VALUES (?, ?, ?)",
            (time.time(), person_count, photo_path)
        )
        conn.commit()
        conn.close()

def log_worker(worker_name, work_time_s, rest_time_s, photo_path):
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO worker_logs (ts, worker_name, work_time_s, rest_time_s, photo_path) VALUES (?, ?, ?, ?, ?)",
            (time.time(), worker_name, work_time_s, rest_time_s, photo_path)
        )
        conn.commit()
        conn.close()

def log_box(loaded_count, unloaded_count, photo_path):
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO box_logs (ts, loaded_count, unloaded_count, photo_path) VALUES (?, ?, ?, ?)",
            (time.time(), loaded_count, unloaded_count, photo_path)
        )
        conn.commit()
        conn.close()

def log_bag_box(boxes_in, boxes_out, bags_in, bags_out, photo_path):
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO bag_box_logs (ts, boxes_in, boxes_out, bags_in, bags_out, photo_path) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), boxes_in, boxes_out, bags_in, bags_out, photo_path)
        )
        conn.commit()
        conn.close()

# Function to fetch data by model
def get_model_data(mode, limit=50):
    with _lock:
        conn = _connect()
        rows = []
        # Fallback for old schemas if needed (SQLite doesn't error if selecting * and columns exist)
        try:
            if mode == "access":
                rows = conn.execute("SELECT id, ts, name, status, photo_path FROM access_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            elif mode == "adaptive":
                rows = conn.execute("SELECT id, ts, state, duration, photo_path FROM adaptive_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            elif mode == "people":
                rows = conn.execute("SELECT id, ts, direction, total_entries, total_exits, photo_path FROM people_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            elif mode == "queue":
                rows = conn.execute("SELECT id, ts, person_count, photo_path FROM queue_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            elif mode == "worker":
                rows = conn.execute("SELECT id, ts, worker_name, work_time_s, rest_time_s, photo_path FROM worker_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            elif mode == "box":
                rows = conn.execute("SELECT id, ts, boxes_in, boxes_out, bags_in, bags_out, photo_path FROM bag_box_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError:
            # Table might not be migrated, just select *
            rows = conn.execute(f"SELECT * FROM {mode}_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
    return [dict(r) for r in rows]
