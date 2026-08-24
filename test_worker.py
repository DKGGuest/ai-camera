import cv2
import os
from src.monitors.worker import WorkerMonitor

def test():
    print("Initializing WorkerMonitor...")
    monitor = WorkerMonitor("src/config/config.yaml")
    
    img_path = r"worker_images\working\WhatsApp Image 2026-08-10 at 1.05.53 PM (1).jpeg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    print(f"Reading {img_path}...")
    frame = cv2.imread(img_path)
    
    print("Processing frame...")
    results, detections = monitor.process_frame(frame)
    
    print("Detections:")
    print({k: len(v) for k, v in detections.items()})
    
    print("Worker Results:")
    for res in results:
        print(f"Worker {res['id']}: Status={res['status']}, Score={res['score']}, Posture={res['posture']}, Interaction={res['interaction']}")
        print(f"  Reasons: {res['reasons']}")
        
if __name__ == '__main__':
    test()
