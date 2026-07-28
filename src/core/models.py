import os
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")

def load_yolo_model(model_name='yolov8s.pt'):
    print(f"Loading YOLO model ({model_name})...")
    return YOLO(model_name)

def load_face_models():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Modern High-Accuracy Face Detection (YuNet)
    detector_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    detector_model = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    
    # Modern High-Accuracy Face Recognition (SFace)
    recognizer_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    recognizer_model = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
    
    download_file(detector_url, detector_model)
    download_file(recognizer_url, recognizer_model)
    
    detector = cv2.FaceDetectorYN.create(detector_model, "", (320, 320), 0.6, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(recognizer_model, "")
    
    return detector, recognizer

def recognizer_match(feature1, feature2):
    f1 = feature1.flatten()
    f2 = feature2.flatten()
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))

def load_known_faces(known_dir, detector, recognizer):
    known_embeddings = {}
    if not os.path.exists(known_dir):
        os.makedirs(known_dir)
        print(f"Created directory {known_dir}. Please add some images of known people.")
        return known_embeddings

    print(f"Loading known faces from '{known_dir}'...")
    for filename in os.listdir(known_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(filename)[0]
            name = base_name.split('_')[0]
            
            img_path = os.path.join(known_dir, filename)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            (h, w) = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            
            if faces is not None and len(faces) > 0:
                face = faces[0]
                aligned_face = recognizer.alignCrop(img, face)
                feature = recognizer.feature(aligned_face)
                known_embeddings[name] = feature[0]
                print(f"Loaded {name}")
            else:
                print(f"Warning: No face found in {filename}")

    return known_embeddings

def load_gender_model():
    """Backwards-compatible stub: returns (None, None).
    OpenCV 5.0 removed CascadeClassifier and Caffe importer.
    Use load_worker_face_detector() instead for face detection."""
    print("[models] Note: Gender model skipped (OpenCV 5 removed Caffe support).")
    return None, None


def load_worker_face_detector():
    """Load YuNet face detector for worker tracking.
    Returns a cv2.FaceDetectorYN instance (works with OpenCV 5.0+).
    YuNet provides face bounding box + 5 landmarks for head pose estimation."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    detector_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    detector_model = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")

    download_file(detector_url, detector_model)

    detector = cv2.FaceDetectorYN.create(
        detector_model, "", (320, 320),
        score_threshold=0.5, nms_threshold=0.3, top_k=5000
    )
    return detector
