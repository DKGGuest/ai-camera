import cv2
import os

def initialize_video_capture(video_source):
    """
    Initializes cv2.VideoCapture and returns the cap object along with width, height, fps.
    Also returns a boolean indicating if the source is an image.
    """
    is_image = isinstance(video_source, str) and video_source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    
    if is_image:
        frame = cv2.imread(video_source)
        if frame is None:
            print(f"Error: Could not read image '{video_source}'")
            return None, None, 0, 0, 0, True
        width = frame.shape[1]
        height = frame.shape[0]
        fps = 30.0
        return frame, None, width, height, fps, True
    
    source = int(video_source) if str(video_source).isdigit() else video_source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return None, None, 0, 0, 0, False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
        
    return None, cap, width, height, fps, False

def initialize_video_writer(output_path, width, height, fps):
    """
    Initializes cv2.VideoWriter.
    Forces output path to be in the 'outputs' directory.
    """
    if not output_path:
        return None, None
        
    os.makedirs('outputs', exist_ok=True)
    basename = os.path.basename(output_path)
    final_output_path = os.path.join('outputs', basename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
    return out, final_output_path
