import cv2
import easyocr
import numpy as np
import time

from src.core.video import initialize_video_capture
from src.core.models import load_yolo_model

def preprocess_plate(plate_img):
    plate_img = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return gray, thresh

def run(video_source='0', log_path='license_plates.log'):
    print("Loading AI Models... This may take a moment.")
    
    model = load_yolo_model('yolov8s.pt')
    VEHICLE_CLASSES = [2, 3, 5, 7]

    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    if plate_cascade.empty():
        print("Error: Could not load the plate cascade classifier.")
        return

    print("Initializing EasyOCR. This will download models if running for the first time...")
    reader = easyocr.Reader(['en'], gpu=True)

    print("Models loaded successfully!")

    frame, cap, width, height, fps, is_image = initialize_video_capture(video_source)
    if width == 0:
        return

    seen_plates = set()
    log_file = open(log_path, 'a')
    
    print("\nStarting ALPR system. Press 'q' to quit.\n")

    frame_count = 0
    
    while True:
        if is_image:
            if frame is None:
                break
        else:
            ret, frame_read = cap.read()
            if not ret:
                break
            frame = frame_read
            
        frame_count += 1
        
        results = model(frame, verbose=False, conf=0.4)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                    cv2.putText(frame, "Vehicle", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    vehicle_roi = frame[vy1:vy2, vx1:vx2]
                    if vehicle_roi.size == 0:
                        continue
                        
                    gray_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                    plates = plate_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=4, minSize=(30, 10))
                    
                    for (px, py, pw, ph) in plates:
                        abs_px1 = vx1 + px
                        abs_py1 = vy1 + py
                        abs_px2 = abs_px1 + pw
                        abs_py2 = abs_py1 + ph
                        
                        cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
                        
                        if is_image or frame_count % 3 == 0:
                            plate_img = frame[abs_py1:abs_py2, abs_px1:abs_px2]
                            gray_plate, thresh_plate = preprocess_plate(plate_img)
                            
                            text_results = reader.readtext(gray_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
                            
                            if len(text_results) > 0:
                                text_results.sort(key=lambda x: x[2], reverse=True)
                                best_text = text_results[0][1]
                                confidence = text_results[0][2]
                                
                                if len(best_text) >= 4 and confidence > 0.4:
                                    cv2.putText(frame, f"{best_text} ({confidence:.2f})", (abs_px1, abs_py1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                                    
                                    if best_text not in seen_plates:
                                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                        log_entry = f"[{timestamp}] Plate: {best_text} | Conf: {confidence:.2f}\n"
                                        print(log_entry.strip())
                                        log_file.write(log_entry)
                                        log_file.flush()
                                        seen_plates.add(best_text)

        cv2.putText(frame, "ALPR System Active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Logged Plates: {len(seen_plates)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display_frame = frame
        if frame.shape[1] > 1200:
            display_frame = cv2.resize(frame, (1200, int(1200 * frame.shape[0] / frame.shape[1])))
            
        cv2.imshow('Automatic License Plate Recognition', display_frame)

        if is_image:
            print("Image processed! Press any key in the image window to exit.")
            cv2.waitKey(0)
            break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not is_image and cap is not None:
        cap.release()
    log_file.close()
    cv2.destroyAllWindows()
    print(f"\nSaved all detected plates to '{log_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automatic License Plate Recognition (ALPR)")
    parser.add_argument('--video', type=str, default='0', help="Path to video file or '0' for webcam")
    parser.add_argument('--log', type=str, default='license_plates.log', help="File to save the read plates")
    args = parser.parse_args()
    run(args.video, args.log)
