import cv2
import sounddevice as sd
import numpy as np
import winsound
import time
import threading
import os
import wave
import subprocess
import imageio_ffmpeg

from src.core.video import initialize_video_capture, initialize_video_writer
from src.core.models import load_yolo_model

# Audio Settings
SAMPLE_RATE = 16000
BLOCK_SIZE = 1000  
CALIBRATION_DURATION = 3.0
SENSITIVITY_MULTIPLIER = 1.5
MIN_ABSOLUTE_THRESHOLD = 0.000 

# Global State
alarm_active = False
alarm_cooldown_until = 0
status_message = "SYSTEM SECURE"
status_color = (0, 255, 0)
audio_rms = 0.0
baseline_rms = 0.0
is_calibrating = True
calibration_samples = []
start_time = time.time()

recorded_audio_chunks = []

def trigger_alarm(reason):
    global alarm_active, alarm_cooldown_until, status_message, status_color
    
    if alarm_active or time.time() < alarm_cooldown_until:
        return
        
    alarm_active = True
    status_message = f"ALARM: {reason.upper()}"
    status_color = (0, 0, 255) 
    print(f"\n🚨 {status_message} 🚨")

    threading.Thread(target=play_siren, daemon=True).start()

def play_siren():
    global alarm_active, alarm_cooldown_until, status_message, status_color
    
    for _ in range(4):
        winsound.Beep(2000, 200)
        winsound.Beep(1500, 200)
        
    alarm_active = False
    alarm_cooldown_until = time.time() + 2.0 
    status_message = "SYSTEM SECURE"
    status_color = (0, 255, 0)

def audio_callback(indata, frames, time_info, status):
    global audio_rms, is_calibrating, baseline_rms, calibration_samples, recorded_audio_chunks, start_time
    
    recorded_audio_chunks.append(indata.copy())
    
    rms = np.sqrt(np.mean(indata**2))
    peak = np.max(np.abs(indata))
    audio_rms = rms 
    
    if is_calibrating:
        calibration_samples.append(rms)
        if time.time() - start_time >= CALIBRATION_DURATION:
            is_calibrating = False
            baseline_rms = max(np.mean(calibration_samples), 0.001)
            print(f"\n[Audio] Calibration Complete. Baseline RMS: {baseline_rms:.5f}")
    else:
        trigger_peak = baseline_rms * SENSITIVITY_MULTIPLIER * 1.5
        trigger_rms = baseline_rms * SENSITIVITY_MULTIPLIER
        
        if peak > trigger_peak or rms > trigger_rms:
            trigger_alarm(f"Loud Sound Spike (Peak: {peak:.2f})")

def run(list_audio=False, audio_device=None, output_path='security_recording.mp4'):
    global start_time
    
    if list_audio:
        print("\nAvailable Audio Devices:")
        print(sd.query_devices())
        return

    print("Initializing Comprehensive Security Monitor...")
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    custom_model_path = os.path.join(models_dir, "weapon_model.pt")
    
    using_custom_model = False
    if os.path.exists(custom_model_path):
        print(f"Loading custom weapon model from {custom_model_path}...")
        model = load_yolo_model(custom_model_path)
        using_custom_model = True
    else:
        print("Custom weapon model not found. Using default YOLOv8s.")
        model = load_yolo_model('yolov8s.pt')
        
    DANGEROUS_COCO_CLASSES = [43, 34]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
    
    # We must save a temp file to output, because the final file is combined with audio
    os.makedirs('outputs', exist_ok=True)
    basename = os.path.basename(output_path)
    final_output_path = os.path.join('outputs', basename)
    
    temp_video_file = os.path.join('outputs', "temp_video_no_audio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_file, fourcc, fps, (width, height))
    print(f"Live monitoring started. Will save final video + audio to: {final_output_path}")

    start_time = time.time()
    audio_stream = sd.InputStream(device=audio_device, callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE)
    audio_stream.start()

    print("\nSecurity System Armed! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=0.4)
        
        threat_detected = False
        threat_name = ""

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                
                is_threat = False
                if using_custom_model:
                    is_threat = True
                elif cls_id in DANGEROUS_COCO_CLASSES:
                    is_threat = True
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if is_threat:
                    threat_detected = True
                    threat_name = name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"THREAT: {name} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)

        if threat_detected:
            trigger_alarm(f"Weapon Detected ({threat_name})")

        overlay = frame.copy()
        
        meter_w = int(min(audio_rms * 2000, 400))
        cv2.rectangle(overlay, (20, 20), (420, 40), (50, 50, 50), -1)
        
        if is_calibrating:
            cv2.putText(overlay, "AUDIO: CALIBRATING...", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            meter_color = (0, 255, 0)
            if audio_rms > (baseline_rms * SENSITIVITY_MULTIPLIER):
                meter_color = (0, 0, 255)
            cv2.rectangle(overlay, (20, 20), (20 + meter_w, 40), meter_color, -1)
            cv2.putText(overlay, f"AUDIO RMS: {audio_rms:.4f}", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        (h, w) = frame.shape[:2]
        cv2.rectangle(overlay, (0, h - 80), (w, h), status_color, -1)
        cv2.putText(overlay, status_message, (30, h - 30), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        out.write(frame)

        cv2.imshow('Comprehensive Security Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    audio_stream.stop()
    audio_stream.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\nSaving audio and video...")
    
    if len(recorded_audio_chunks) > 0:
        temp_audio_file = os.path.join('outputs', "temp_audio.wav")
        full_audio = np.concatenate(recorded_audio_chunks, axis=0)
        
        full_audio_clamped = np.clip(full_audio, -1.0, 1.0)
        audio_int16 = np.int16(full_audio_clamped * 32767)
        
        with wave.open(temp_audio_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
            
        print("Merging audio and video (this may take a moment)...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        command = [
            ffmpeg_exe,
            '-y', 
            '-i', temp_video_file,
            '-i', temp_audio_file,
            '-c:v', 'copy',
            '-c:a', 'aac',
            final_output_path
        ]
        
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("Error merging audio/video!")
            print(result.stderr.decode('utf-8'))
        else:
            print(f"✅ Final recording successfully saved with sound: {final_output_path}")
            
        try:
            os.remove(temp_video_file)
            os.remove(temp_audio_file)
        except:
            pass
    else:
        print(f"Warning: No audio recorded. Video saved to {temp_video_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Security Monitor")
    parser.add_argument('--list_audio', action='store_true', help="List all available audio input devices and exit")
    parser.add_argument('--audio_device', type=int, default=None, help="The ID of the external microphone to use")
    parser.add_argument('--output', type=str, default='security_recording.mp4', help="Path to save the final recording with sound")
    args = parser.parse_args()
    run(args.list_audio, args.audio_device, args.output)
