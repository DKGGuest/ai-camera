import os

# ---------------------------------------------------------------------------
# Login credentials (change these!)
# ---------------------------------------------------------------------------
ADMIN_USERNAME = os.environ.get("SC_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("SC_ADMIN_PASS", "admin123")

# Flask secret key (used to sign the login session cookie)
SECRET_KEY = os.environ.get("SC_SECRET_KEY", "change-this-secret-key-please")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_workers")
DB_PATH = os.path.join(BASE_DIR, "smart_camera.db")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CAMERA_URL = os.environ.get("SC_CAMERA_URL", "http://192.168.1.50:81/stream")
FACE_SIMILARITY_THRESHOLD = 0.363
