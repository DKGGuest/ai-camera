import os
import urllib.request

import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld

import config


def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")


def load_yolo_model():
    print(f"Loading YOLO model ({config.YOLO_MODEL_PATH})...")
    return YOLO(config.YOLO_MODEL_PATH)


def load_custom_box_model():
    world_path = os.path.join(config.BASE_DIR, "yolov8s-world.pt")
    print(f"Loading YOLO-World model for boxes ({world_path})...")
    if os.path.exists(world_path):
        model = YOLOWorld(world_path)
        model.set_classes(["cardboard box", "sugar_sack", "box"])
        return model
    else:
        print("Warning: YOLO-World model not found. Falling back to default YOLO model.")
        return YOLO(config.YOLO_MODEL_PATH)


def load_worker_classifier():
    classifier_path = os.path.join(config.BASE_DIR, "worker_classifier.pt")
    print(f"Loading custom worker classifier ({classifier_path})...")
    if os.path.exists(classifier_path):
        return YOLO(classifier_path)
    else:
        print("Warning: Custom worker classifier not found.")
        return None


def load_face_models():
    models_dir = os.path.join(config.BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    detector_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    detector_model = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")

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
        return known_embeddings

    for filename in os.listdir(known_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            base_name = os.path.splitext(filename)[0]
            name = base_name.split("_")[0]

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
                print(f"Loaded known face: {name}")

    return known_embeddings
