# Smart Camera Web Dashboard

A Chrome-based website (Flask backend) that pulls the live feed from your
**ESP32-S3-CAM**, runs a switchable AI model on it, logs every detection to a
live table, and lets you enroll faces from the browser.

Built on top of your existing `smart-camera-main-main` project (YuNet/SFace
face recognition + YOLOv8 vehicle/person detection) — this just wraps it in a
web app instead of the old `cv2.imshow` desktop windows.

## Models included (switch with one click on the dashboard)

| Mode       | What it does                                                            |
|------------|--------------------------------------------------------------------------|
| `access`   | Face recognition → ACCESS GRANTED / ACCESS DENIED, entry logged with name |
| `vehicle`  | YOLOv8 vehicle detection, tracking, and line-crossing count             |
| `adaptive` | Detects if a person is present; drops resolution/bitrate when idle       |

## 1. Install dependencies

```bash
cd webapp
pip install -r requirements.txt
```

## 2. Set your ESP32-S3-CAM stream URL

Your ESP32-S3-CAM firmware serves an MJPEG stream, usually at a URL like:

```
http://<esp32-ip-address>:81/stream
```

You can either:
- Edit `config.py` → `DEFAULT_CAMERA_URL`, or
- Set it live from the dashboard ("ESP32-S3-CAM Stream URL" box → Save) once
  the site is running — no restart needed.

## 3. Set your login (optional, defaults to admin/admin123)

```bash
export SC_ADMIN_USER=admin
export SC_ADMIN_PASS=your_secure_password
export SC_SECRET_KEY=some_random_string
```
(On Windows PowerShell use `$env:SC_ADMIN_PASS="..."` instead of `export`.)

## 4. Run it

```bash
python app.py
```

Then open **Chrome** and go to:

```
http://localhost:5000
```

Log in, and you'll land on the dashboard with:
- Live ESP32-S3-CAM video feed
- Buttons to switch between Access Control / Vehicle Detection / Adaptive Bitrate
- A live-updating table of every detection event (grant/denied, vehicle
  crossings, bitrate switches)

## 5. Enroll faces for Access Control

Go to **Enroll Faces** in the nav bar. You can either:
- Upload a photo, or
- Click **Capture Now** to grab the current frame straight from the live
  ESP32 stream.

Both save into the `known_workers/` folder (auto-created) and the running
Access Control model reloads instantly — no restart needed.

## Notes

- The first time you switch to `access` mode, it downloads two small ONNX
  models (YuNet face detector, SFace recognizer) into `models/` — needs
  internet access once.
- `yolov8n.pt` is already bundled in this folder for vehicle/person detection.
- All detection events are stored in `smart_camera.db` (SQLite) — query it
  directly with any SQLite browser if you want raw data outside the website.
- If the ESP32 stream drops, the dashboard badge turns red and the backend
  auto-retries the connection every second.
