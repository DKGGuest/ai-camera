import cv2
import numpy as np

from src.core.video import initialize_video_capture, initialize_video_writer
from src.core.models import load_face_models, load_known_faces, recognizer_match

def run(video_source='sample_traffic.mp4', output_path='face_output.mp4', known_dir='known_workers'):
    print("Loading High-Accuracy AI Models...")
    detector, recognizer = load_face_models()

    known_embeddings = load_known_faces(known_dir, detector, recognizer)
    if not known_embeddings:
        print("Warning: No known faces loaded. The system will only detect faces without recognizing them.")

    similarity_threshold = 0.363

    frame, cap, width, height, fps, is_image = initialize_video_capture(video_source)
    if width == 0:
        return

    out, final_output_path = initialize_video_writer(output_path, width, height, fps)

    print(f"Processing '{video_source}'...")
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

        (h, w) = frame.shape[:2]
        detector.setInputSize((w, h))
        
        _, faces = detector.detect(frame)
        
        valid_faces = []
        if faces is not None:
            for face in faces:
                box = face[0:4].astype(int)
                x, y, fw, fh = box
                
                try:
                    aligned_face = recognizer.alignCrop(frame, face)
                    feature = recognizer.feature(aligned_face)
                    valid_faces.append({'box': (x, y, x+fw, y+fh), 'emb': feature[0], 'face_data': face})
                except Exception as e:
                    pass

        matches = []
        for f_idx, face in enumerate(valid_faces):
            for known_key, known_emb in known_embeddings.items():
                score = recognizer_match(face['emb'], known_emb)
                if score > similarity_threshold:
                    matches.append((score, f_idx, known_key))

        matches.sort(key=lambda x: x[0], reverse=True)

        assigned_faces = {}
        assigned_names = set()

        for score, f_idx, known_key in matches:
            if f_idx not in assigned_faces and known_key not in assigned_names:
                assigned_faces[f_idx] = (known_key, score)
                assigned_names.add(known_key)

        for f_idx, face in enumerate(valid_faces):
            (startX, startY, endX, endY) = face['box']
            
            if f_idx in assigned_faces:
                best_match, best_score = assigned_faces[f_idx]
                color = (0, 255, 0)
                label = f"{best_match} ({best_score:.2f})"
            else:
                color = (0, 0, 255)
                label = "Unknown"
                
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, max(startY - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if out is not None:
            out.write(frame)

        display_frame = frame
        if width > 800:
            display_frame = cv2.resize(frame, (800, int(800 * height / width)))
        cv2.imshow('High-Accuracy Facial Recognition', display_frame)
        
        if is_image:
            print("Image processed! Press any key in the image window to exit.")
            cv2.waitKey(0)
            break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not is_image and cap is not None:
        cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete! Output saved to '{final_output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Facial Recognition on Video using OpenCV SFace")
    parser.add_argument('--video', type=str, default='sample_traffic.mp4', help='Path to the input video file or camera index (e.g. 0)')
    parser.add_argument('--output', type=str, default='face_output.mp4', help='Path to save the output video')
    parser.add_argument('--known_dir', type=str, default='known_workers', help='Directory containing known faces')
    args = parser.parse_args()
    run(args.video, args.output, args.known_dir)
