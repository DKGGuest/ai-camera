import cv2

from src.core.video import initialize_video_capture
from src.core.models import load_face_models, load_known_faces, recognizer_match
from src.utils.drawing import draw_banner

def run(video_source='0', known_dir='known_workers'):
    print("Loading High-Accuracy Access Control System...")
    detector, recognizer = load_face_models()
    
    known_embeddings = load_known_faces(known_dir, detector, recognizer)
    
    if not known_embeddings:
        print("Warning: No known faces enrolled. Everyone will be DENIED.")

    similarity_threshold = 0.363 

    frame, cap, width, height, fps, is_image = initialize_video_capture(video_source)
    if width == 0:
        return

    print("System Ready. Waiting for a face...")

    while True:
        if is_image:
            if frame is None:
                break
        else:
            ret, frame_read = cap.read()
            if not ret:
                break
            frame = frame_read

        (h, w) = frame.shape[:2]
        detector.setInputSize((w, h))
        
        _, faces = detector.detect(frame)

        largest_face_area = 0
        largest_face_data = None
        largest_box = None

        if faces is not None:
            for face in faces:
                box = face[0:4].astype(int)
                x, y, fw, fh = box
                area = fw * fh
                
                if area > largest_face_area and fw > 40 and fh > 40:
                    largest_face_area = area
                    largest_face_data = face
                    largest_box = box

        status_text = "SCANNING..."
        status_color = (150, 150, 150)
        
        if largest_face_data is not None:
            x, y, fw, fh = largest_box
            
            try:
                aligned_face = recognizer.alignCrop(frame, largest_face_data)
                feature = recognizer.feature(aligned_face)
                
                best_match = "Unknown"
                best_score = -1.0
                
                for known_key, known_emb in known_embeddings.items():
                    score = recognizer_match(feature[0], known_emb)
                    if score > best_score:
                        best_score = score
                        if score > similarity_threshold:
                            best_match = known_key.split('_')[0]
                            
                if best_match != "Unknown":
                    status_text = f"ACCESS GRANTED: {best_match.upper()}"
                    status_color = (0, 200, 0)
                    box_color = (0, 255, 0)
                else:
                    status_text = "ACCESS DENIED"
                    status_color = (0, 0, 200)
                    box_color = (0, 0, 255)
                    
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), box_color, 3)
                
            except Exception as e:
                pass
                
        draw_banner(frame, status_text, status_color)

        display_frame = frame
        if w > 1000:
            display_frame = cv2.resize(frame, (1000, int(1000 * h / w)))
            
        try:
            cv2.imshow('Smart Access Control', display_frame)
            if is_image:
                print("Image processed! Press any key in the image window to exit.")
                cv2.waitKey(0)
                break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except cv2.error:
            pass

    if not is_image and cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="High-Accuracy Access Control")
    parser.add_argument('--video', type=str, default='0', help='Path to the input video file or camera index (e.g. 0)')
    parser.add_argument('--known_dir', type=str, default='known_workers', help='Directory containing known faces')
    args = parser.parse_args()
    run(args.video, args.known_dir)
