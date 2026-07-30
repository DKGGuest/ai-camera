import json
import os

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_SETTINGS = {
    "conf_threshold": 0.4,
    "iou_threshold": 0.45,
    "tracking_threshold": 0.5,
    "fps_limit": 30,
    "line_thickness": 2,
    "bbox_thickness": 2,
    "show_labels": True,
    "show_ids": True,
    "show_confidence": True,
    "model_path": "yolov8n.pt",
    "counting_line": None # [(x1, y1), (x2, y2)] in relative coordinates (0 to 1)
}

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
            # Merge with defaults to ensure all keys exist
            settings = DEFAULT_SETTINGS.copy()
            settings.update(data)
            return settings
    except Exception as e:
        print(f"Error loading settings: {e}")
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False
