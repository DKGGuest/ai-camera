import os
import time
from functools import wraps

import cv2
from flask import (Flask, Response, jsonify, redirect, render_template,
                    request, session, url_for)

import config
import database
from camera_worker import worker

app = Flask(__name__)
app.secret_key = config.SECRET_KEY


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return redirect(url_for("dashboard") if session.get("logged_in") else url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    status = worker.get_status()
    return render_template("dashboard.html", status=status, modes=["access","vehicle","adaptive","people","worker","queue", "box"])


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(worker.mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
@login_required
def api_status():
    return jsonify(worker.get_status())


@app.route("/api/set_mode", methods=["POST"])
@login_required
def api_set_mode():
    mode = request.json.get("mode")
    try:
        worker.set_mode(mode)
        return jsonify({"ok": True, "mode": mode})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/set_camera", methods=["POST"])
@login_required
def api_set_camera():
    url = request.json.get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "Empty URL"}), 400
    worker.set_camera_url(url)
    return jsonify({"ok": True, "url": url})


@app.route("/api/events")
@login_required
def api_events():
    return jsonify(database.get_events(limit=50))


@app.route("/api/model_data")
@login_required
def api_model_data():
    mode = request.args.get("mode", "access")
    return jsonify(database.get_model_data(mode, limit=50))


@app.route("/api/clear_events", methods=["POST"])
@login_required
def api_clear_events():
    database.clear_all_events()
    return jsonify({"ok": True})


@app.route("/api/reset_people_counts", methods=["POST"])
@login_required
def api_reset_people_counts():
    worker.reset_people_counts()
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# Enroll known faces (upload photo OR capture current frame)
# --------------------------------------------------------------------------- #
@app.route("/enroll", methods=["GET"])
@login_required
def enroll_page():
    faces = sorted(os.listdir(config.KNOWN_FACES_DIR)) if os.path.exists(config.KNOWN_FACES_DIR) else []
    return render_template("enroll.html", faces=faces)


@app.route("/api/enroll/upload", methods=["POST"])
@login_required
def enroll_upload():
    name = request.form.get("name", "").strip().replace(" ", "")
    file = request.files.get("photo")
    if not name or not file:
        return jsonify({"ok": False, "error": "Name and photo are required"}), 400

    os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{name}_{int(time.time())}{ext}"
    filepath = os.path.join(config.KNOWN_FACES_DIR, filename)
    file.save(filepath)

    worker.reload_known_faces()
    database.log_event("access", "enrolled", f"{name} (uploaded photo)", "info")
    return jsonify({"ok": True, "filename": filename})


@app.route("/api/enroll/capture", methods=["POST"])
@login_required
def enroll_capture():
    name = request.json.get("name", "").strip().replace(" ", "")
    if not name:
        return jsonify({"ok": False, "error": "Name is required"}), 400

    frame = worker.get_latest_frame_bgr()
    if frame is None:
        return jsonify({"ok": False, "error": "No camera frame available yet"}), 400

    os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
    filename = f"{name}_{int(time.time())}.jpg"
    filepath = os.path.join(config.KNOWN_FACES_DIR, filename)
    cv2.imwrite(filepath, frame)

    worker.reload_known_faces()
    database.log_event("access", "enrolled", f"{name} (captured from camera)", "info")
    return jsonify({"ok": True, "filename": filename})


@app.route("/api/set_queue_roi", methods=["POST"])
@login_required
def api_set_queue_roi():
    points = request.json.get("points")
    if points is not None and len(points) != 0:
        if not isinstance(points, list) or len(points) > 5:
            return jsonify({"ok": False, "error": "Must have 1 to 5 queues"}), 400
        for roi in points:
            if not isinstance(roi, list) or len(roi) != 4:
                return jsonify({"ok": False, "error": "Each queue must have exactly 4 points"}), 400
    
    settings = worker.load_settings_dict()
    settings["queue_roi"] = points if points else None
    worker.save_settings_dict(settings)
    
    worker.reload_settings()
    return jsonify({"ok": True})


@app.route("/api/set_people_lines", methods=["POST"])
@login_required
def api_set_people_lines():
    points = request.json.get("points")
    if points is not None and len(points) != 0:
        if not isinstance(points, list) or len(points) != 2:
            return jsonify({"ok": False, "error": "Must have exactly 2 lines"}), 400
        for line in points:
            if not isinstance(line, list) or len(line) != 4:
                return jsonify({"ok": False, "error": "Each line must have exactly 4 values (x1, y1, x2, y2)"}), 400
    
    settings = worker.load_settings_dict()
    settings["people_lines"] = points if points else None
    worker.save_settings_dict(settings)
    
    worker.reload_settings()
    return jsonify({"ok": True})


@app.route("/api/set_box_line", methods=["POST"])
@login_required
def api_set_box_line():
    points = request.json.get("points")
    if points is not None and len(points) != 0:
        if not isinstance(points, list) or len(points) != 2:
            return jsonify({"ok": False, "error": "Must have exactly 2 points for a line"}), 400
    
    settings = worker.load_settings_dict()
    settings["box_line"] = points if points else None
    worker.save_settings_dict(settings)
    
    worker.reload_settings()
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    database.init_db()
    worker.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
