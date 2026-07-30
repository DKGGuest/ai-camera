import sys
import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QInputDialog, QMessageBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from settings import load_settings, save_settings
from line_manager import LineManager
from counter import Counter
from detector import Detector
from utils import draw_bbox, draw_line, draw_dashboard, COLOR_BOX, COLOR_SACK

class VideoThread:
    """Simple wrapper to manage OpenCV capture in the main thread for now,
    but can be expanded to a QThread if heavy lag occurs."""
    pass

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial AI Truck Loading Counter")
        self.resize(1200, 800)
        
        self.settings = load_settings()
        self.detector = Detector(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", self.settings["model_path"]))
        self.line_mgr = LineManager()
        if self.settings.get("counting_line"):
            self.line_mgr.set_line(self.settings["counting_line"][0], self.settings["counting_line"][1])
            
        self.counter = Counter(log_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "events.csv"))
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_paused = False
        
        # Drawing state
        self.drawing_line = False
        self.temp_line_pt1 = None
        
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # Left Panel - Video Display
        self.video_label = QLabel("No Video Source")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
        self.video_label.setMinimumSize(800, 600)
        self.video_label.mousePressEvent = self.video_mouse_press
        main_layout.addWidget(self.video_label, stretch=3)
        
        # Right Panel - Controls
        control_layout = QVBoxLayout()
        
        btn_cam = QPushButton("Open Camera")
        btn_cam.clicked.connect(self.open_camera)
        control_layout.addWidget(btn_cam)
        
        btn_vid = QPushButton("Load Video File")
        btn_vid.clicked.connect(self.open_video)
        control_layout.addWidget(btn_vid)
        
        btn_rtsp = QPushButton("Open RTSP Stream")
        btn_rtsp.clicked.connect(self.open_rtsp)
        control_layout.addWidget(btn_rtsp)
        
        btn_pause = QPushButton("Pause / Resume")
        btn_pause.clicked.connect(self.toggle_pause)
        control_layout.addWidget(btn_pause)
        
        control_layout.addSpacing(20)
        
        btn_draw = QPushButton("Draw Counting Line")
        btn_draw.clicked.connect(self.start_draw_line)
        control_layout.addWidget(btn_draw)
        
        btn_clear_line = QPushButton("Clear Line")
        btn_clear_line.clicked.connect(self.clear_line)
        control_layout.addWidget(btn_clear_line)
        
        btn_reset = QPushButton("Reset Counts")
        btn_reset.clicked.connect(self.reset_counts)
        control_layout.addWidget(btn_reset)
        
        control_layout.addStretch()
        
        # Settings Group
        settings_group = QGroupBox("Settings")
        form_layout = QFormLayout()
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.settings["conf_threshold"])
        self.conf_spin.valueChanged.connect(self.update_settings)
        form_layout.addRow("Confidence:", self.conf_spin)
        
        settings_group.setLayout(form_layout)
        control_layout.addWidget(settings_group)
        
        main_layout.addLayout(control_layout, stretch=1)
        self.setLayout(main_layout)

    def open_camera(self):
        self.start_video(0)

    def open_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename:
            self.start_video(filename)
            
    def open_rtsp(self):
        url, ok = QInputDialog.getText(self, "Open RTSP", "Enter RTSP URL:")
        if ok and url:
            self.start_video(url)

    def start_video(self, source):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(source)
        self.timer.start(int(1000 / self.settings["fps_limit"]))

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def start_draw_line(self):
        self.drawing_line = True
        self.temp_line_pt1 = None
        QMessageBox.information(self, "Draw Line", "Click two points on the video to draw the counting line.")

    def clear_line(self):
        self.line_mgr.clear_line()
        self.settings["counting_line"] = None
        save_settings(self.settings)

    def reset_counts(self):
        self.counter.reset()

    def update_settings(self):
        self.settings["conf_threshold"] = self.conf_spin.value()
        save_settings(self.settings)

    def video_mouse_press(self, event):
        if not self.drawing_line or self.cap is None:
            return
            
        # Map widget click coordinates to relative 0-1 coordinates
        x = event.pos().x() / self.video_label.width()
        y = event.pos().y() / self.video_label.height()
        
        if self.temp_line_pt1 is None:
            self.temp_line_pt1 = (x, y)
        else:
            self.line_mgr.set_line(self.temp_line_pt1, (x, y))
            self.drawing_line = False
            self.temp_line_pt1 = None
            
            # Save to config
            self.settings["counting_line"] = [self.line_mgr.pt1, self.line_mgr.pt2]
            save_settings(self.settings)

    def update_frame(self):
        if self.is_paused or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
            
        # Ensure aspect ratio scaling to fit the QLabel without distortion
        # For simplicity in logic, we process the raw frame and map the line.
        h, w = frame.shape[:2]
        
        # 1. Detect and Track
        detections = self.detector.process_frame(frame, self.settings["conf_threshold"], self.settings["iou_threshold"])
        
        # 2. Process Crossings
        active_ids = []
        for det in detections:
            active_ids.append(det["track_id"])
            self.counter.process_object(det["track_id"], det["cls_name"], det["centroid"], self.line_mgr, w, h)
            
            # Draw BBox
            color = COLOR_BOX if "box" in det["cls_name"] else COLOR_SACK
            draw_bbox(frame, det["bbox"], f"ID:{det['track_id']} {det['cls_name']}", color, 
                      show_conf=self.settings["show_confidence"], conf=det["conf"])
            
            # Draw Centroid
            cv2.circle(frame, det["centroid"], 4, color, -1)

        self.counter.cleanup_states(active_ids)
        
        # 3. Draw Line & Dashboard
        pt1, pt2 = self.line_mgr.get_absolute_points(w, h)
        if pt1 and pt2:
            draw_line(frame, pt1, pt2)
            
        draw_dashboard(frame, self.counter.loading_count, self.counter.unloading_count, 
                       self.settings["fps_limit"], len(detections), self.settings)

        # 4. Render to PyQt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))
