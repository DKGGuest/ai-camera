import psycopg2
import psycopg2.extras
import threading
import time
from datetime import datetime

import config

_lock = threading.Lock()


def _connect():
    conn = psycopg2.connect(config.DB_URL)
    return conn


def init_db():
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            
            # Access Logs Table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS access_logs (
                    id SERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    date TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    photo_path TEXT
                )
                """
            )
            
            # Adaptive Logs Table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptive_logs (
                    id SERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    date TEXT NOT NULL,
                    state TEXT NOT NULL,
                    duration TEXT NOT NULL,
                    photo_path TEXT
                )
                """
            )
            
            # People Logs Table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS people_logs (
                    id SERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    date TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    total_entries INTEGER NOT NULL,
                    total_exits INTEGER NOT NULL,
                    photo_path TEXT
                )
                """
            )
            
            # Queue Logs Table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS queue_logs (
                    id SERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    date TEXT NOT NULL,
                    person_count TEXT NOT NULL,
                    photo_path TEXT
                )
                """
            )
            
            # Worker Logs Table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_logs (
                    id SERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    date TEXT NOT NULL,
                    worker_name TEXT NOT NULL,
                    work_time_s DOUBLE PRECISION NOT NULL,
                    rest_time_s DOUBLE PRECISION NOT NULL,
                    photo_path TEXT
                )
                """
            )
            
            # Box Logs Table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS box_logs (
                    id SERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    date TEXT NOT NULL,
                    loaded_count INTEGER NOT NULL,
                    unloaded_count INTEGER NOT NULL,
                    photo_path TEXT
                )
                """
            )
            
            # Bag & Box Logs Table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bag_box_logs (
                    id SERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    date TEXT NOT NULL,
                    boxes_in INTEGER NOT NULL,
                    boxes_out INTEGER NOT NULL,
                    bags_in INTEGER NOT NULL,
                    bags_out INTEGER NOT NULL,
                    photo_path TEXT
                )
                """
            )
            
            # Room Logs Table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS room_logs (
                    id SERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    date TEXT NOT NULL,
                    person_count INTEGER NOT NULL,
                    photo_path TEXT
                )
                """
            )
        
        conn.commit()
        conn.close()


def cleanup_old_records():
    """Delete records older than 7 days (7 * 24 * 3600 seconds)"""
    cutoff = time.time() - (7 * 24 * 3600)
    tables = ["access_logs", "adaptive_logs", "people_logs", "queue_logs", "worker_logs", "box_logs", "bag_box_logs", "room_logs"]
    
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            for table in tables:
                try:
                    cur.execute(f"DELETE FROM {table} WHERE ts < %s", (cutoff,))
                except Exception:
                    pass
        conn.commit()
        conn.close()

def _get_date_str():
    return datetime.now().strftime("%Y-%m-%d")

# Functions to insert into specific tables
def log_access(name, status, photo_path):
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO access_logs (ts, date, name, status, photo_path) VALUES (%s, %s, %s, %s, %s)",
                (time.time(), _get_date_str(), name, status, photo_path)
            )
        conn.commit()
        conn.close()

def log_adaptive(state, duration, photo_path):
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO adaptive_logs (ts, date, state, duration, photo_path) VALUES (%s, %s, %s, %s, %s)",
                (time.time(), _get_date_str(), state, str(duration), photo_path)
            )
        conn.commit()
        conn.close()

def log_people(direction, total_entries, total_exits, photo_path):
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO people_logs (ts, date, direction, total_entries, total_exits, photo_path) VALUES (%s, %s, %s, %s, %s, %s)",
                (time.time(), _get_date_str(), direction, total_entries, total_exits, photo_path)
            )
        conn.commit()
        conn.close()

def log_queue(person_count, photo_path):
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO queue_logs (ts, date, person_count, photo_path) VALUES (%s, %s, %s, %s)",
                (time.time(), _get_date_str(), person_count, photo_path)
            )
        conn.commit()
        conn.close()

def log_worker(worker_name, work_time_s, rest_time_s, photo_path):
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO worker_logs (ts, date, worker_name, work_time_s, rest_time_s, photo_path) VALUES (%s, %s, %s, %s, %s, %s)",
                (time.time(), _get_date_str(), worker_name, work_time_s, rest_time_s, photo_path)
            )
        conn.commit()
        conn.close()

def log_box(loaded_count, unloaded_count, photo_path):
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO box_logs (ts, date, loaded_count, unloaded_count, photo_path) VALUES (%s, %s, %s, %s, %s)",
                (time.time(), _get_date_str(), loaded_count, unloaded_count, photo_path)
            )
        conn.commit()
        conn.close()

def log_bag_box(boxes_in, boxes_out, bags_in, bags_out, photo_path):
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO bag_box_logs (ts, date, boxes_in, boxes_out, bags_in, bags_out, photo_path) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (time.time(), _get_date_str(), boxes_in, boxes_out, bags_in, bags_out, photo_path)
            )
        conn.commit()
        conn.close()

def log_room(person_count, photo_path):
    with _lock:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO room_logs (ts, date, person_count, photo_path) VALUES (%s, %s, %s, %s)",
                (time.time(), _get_date_str(), person_count, photo_path)
            )
        conn.commit()
        conn.close()

def log_event(mode, event_type, detail, status):
    pass

def get_events(limit=50):
    return []

def clear_all_events():
    pass

# Function to fetch data by model
def get_model_data(mode, limit=50):
    with _lock:
        conn = _connect()
        rows = []
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                if mode == "access":
                    cur.execute("SELECT id, ts, date, name, status, photo_path FROM access_logs ORDER BY id DESC LIMIT %s", (limit,))
                elif mode == "adaptive":
                    cur.execute("SELECT id, ts, date, state, duration, photo_path FROM adaptive_logs ORDER BY id DESC LIMIT %s", (limit,))
                elif mode == "people":
                    cur.execute("SELECT id, ts, date, direction, total_entries, total_exits, photo_path FROM people_logs ORDER BY id DESC LIMIT %s", (limit,))
                elif mode == "queue":
                    cur.execute("SELECT id, ts, date, person_count, photo_path FROM queue_logs ORDER BY id DESC LIMIT %s", (limit,))
                elif mode == "worker":
                    cur.execute("SELECT id, ts, date, worker_name, work_time_s, rest_time_s, photo_path FROM worker_logs ORDER BY id DESC LIMIT %s", (limit,))
                elif mode == "box":
                    cur.execute("SELECT id, ts, date, loaded_count, unloaded_count, photo_path FROM box_logs ORDER BY id DESC LIMIT %s", (limit,))
                elif mode == "room":
                    cur.execute("SELECT id, ts, date, person_count, photo_path FROM room_logs ORDER BY id DESC LIMIT %s", (limit,))
                else:
                    return []
                rows = cur.fetchall()
        except psycopg2.Error:
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(f"SELECT * FROM {mode}_logs ORDER BY id DESC LIMIT %s", (limit,))
                    rows = cur.fetchall()
            except psycopg2.Error:
                pass
        conn.close()
    return [dict(r) for r in rows]
