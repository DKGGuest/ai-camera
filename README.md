# Smart Camera Intelligence Dashboard

A comprehensive, Flask-based AI surveillance and analytics dashboard that seamlessly pulls live RTSP/HTTP streams (like ESP32-S3-CAM or CP Plus) and processes them through multiple switchable AI models in real-time. It features an interactive UI for drawing custom regions of interest (ROIs) and tracking lines directly on the camera feed.

This platform bridges the gap between raw camera streams and actionable business intelligence, storing all events locally in an SQLite database and presenting them via a live-updating web interface.

## 🌟 Key Features

- **Multi-Model AI Architecture:** Seamlessly switch between 7 specialized computer vision models on-the-fly without restarting the stream.
- **Interactive ROI Setup:** Draw custom polygon zones (for queue monitoring) or multi-point crossing lines (for people and box counting) directly on the web video feed.
- **Zero-Latency Stream Handling:** Implements a dedicated capture thread that continuously consumes RTSP network buffers, ensuring the AI processing never falls behind the live stream.
- **Live Event Logging:** Every detection, entry, exit, or productivity shift is logged to a local SQLite database (`smart_camera.db`) and pushed to the dashboard instantly.
- **Dynamic Camera Configuration:** Switch between default ESP32 streams, RTSP feeds, or any custom IP camera directly from the UI.
- **Face Enrollment System:** Upload photos or capture directly from the live feed to enroll new users into the Access Control model instantly.

---

## 🧠 Available AI Models

| Mode | Name | Description & Capabilities |
|------|------|---------------------------|
| 🧑‍💼 | **Access Control** | Uses YuNet and SFace for high-speed face detection and recognition. Logs "ACCESS GRANTED" for enrolled faces and "ACCESS DENIED" for unknown persons. |
| 🚗 | **Vehicle Detection** | Uses YOLOv8 to detect and track cars, motorcycles, buses, and trucks. Counts vehicles crossing a designated threshold. |
| 🚶 | **People Counter** | Dual-line directional counting. Users can draw 4 points to establish "Outside" and "Inside" lines. Tracks Entries, Exits, and calculates current occupancy. |
| 📦 | **Box Loading Tracker** | Specialized YOLO tracker for inventory management. Users draw 4 points (2 lines) to create a directional threshold that logs when cardboard boxes are "Loaded" or "Unloaded". |
| 🏭 | **Worker Tracker** | Advanced productivity monitor. Tracks individuals and cross-references their face orientation, proximity to laptops, and cell phone usage to determine if they are "Working", "Using Phone", "Talking", or "Idle". Provides a live productivity percentage HUD. |
| 🧍‍♂️ | **Queue Monitor** | Allows users to draw up to 5 custom polygon zones on the camera feed. Continuously monitors the number of people inside each zone and issues alerts for long wait times. |
| 🪑 | **Desk Occupancy** | Detects desks/chairs and people simultaneously. Accurately maps sitting individuals to specific chairs, artificially tracking occluded chairs. Displays green/red boxes for Empty/Occupied statuses. |
| 👥 | **Room Occupancy** | Monitors the overall active occupancy of a room. Detects people and marks their chest points, keeping a live count of total individuals present without drawing invasive bounding boxes. |
| ⚡ | **Adaptive Bitrate** | Privacy & bandwidth preservation mode. Drops stream resolution and bitrate to ultra-low when the scene is empty. Instantly switches to high-fidelity processing when a person is detected. |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, SQLite3
- **Computer Vision:** OpenCV (cv2), Ultralytics YOLOv8, Custom YOLO models
- **Frontend:** Vanilla JavaScript, HTML5, CSS3, SVG (for interactive ROIs)
- **Tracking:** ByteTrack & BoT-SORT (custom configuration)

---

## 🚀 Installation & Setup

### 1. Install Dependencies
Clone the repository and install the required Python packages. Python 3.9+ is recommended.
```bash
git clone <repository_url>
cd webapp
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
The system defaults to `admin / admin123`. You can override this using environment variables:
```bash
export SC_ADMIN_USER=admin
export SC_ADMIN_PASS=your_secure_password
export SC_SECRET_KEY=your_secret_flask_key
```
*(On Windows PowerShell, use `$env:SC_ADMIN_PASS="..."`)*

### 3. Run the Server
```bash
python app.py
```

### 4. Access the Dashboard
Open your web browser (Chrome recommended for optimal MJPEG streaming) and navigate to:
```
http://localhost:5000
```

---

## 📸 Adding Cameras

The dashboard comes with built-in presets for an ESP32-S3-CAM and a CP Plus RTSP stream. To add a new camera:
1. Select **"Add Custom Camera IP/URL..."** from the dropdown.
2. Enter your MJPEG HTTP URL (e.g., `http://192.168.1.50:81/stream`) or RTSP stream URL.
3. Click **Add**. The backend will automatically re-initialize the capture thread with the new feed.

## 📂 Database & Logs

All analytics are stored in `smart_camera.db`. 
- **Events Table:** High-level alerts and events (e.g., Access Denied, Long Queue Alert).
- **Model-Specific Tables:** Each model maintains its own table (e.g., `worker_logs`, `bag_box_logs`) for granular reporting and dashboard visualization.
- **Periodic Snapshots:** The system saves annotated snapshot frames to `static/events/` when significant actions occur (or periodically for timeline tracking).
