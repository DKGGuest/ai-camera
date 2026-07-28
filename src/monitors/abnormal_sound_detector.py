import sounddevice as sd
import numpy as np
import winsound
import time
import queue
import soundfile as sf

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  
CALIBRATION_DURATION = 3.0 
SENSITIVITY_MULTIPLIER = 5.0 
MIN_ABSOLUTE_THRESHOLD = 0.05 

class AbnormalSoundDetector:
    def __init__(self):
        self.q = queue.Queue()
        self.is_calibrating = True
        self.calibration_samples = []
        self.baseline_rms = 0.0
        self.start_time = time.time()
        
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        rms = np.sqrt(np.mean(indata**2))
        self.q.put(rms)

    def trigger_alarm(self):
        print("\n" + "!"*50)
        print("🚨 ABNORMAL SOUND DETECTED! 🚨")
        print("!"*50 + "\n")
        
        for _ in range(8):
            winsound.Beep(3500, 100) 
            winsound.Beep(1500, 100) 
            winsound.Beep(4500, 100) 
            winsound.Beep(1500, 100) 
            
        print("Resuming monitoring...\n")
        time.sleep(1)
        
    def process_rms(self, current_rms):
        if self.is_calibrating:
            self.calibration_samples.append(current_rms)
            
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= CALIBRATION_DURATION:
                self.is_calibrating = False
                self.baseline_rms = np.mean(self.calibration_samples)
                self.baseline_rms = max(self.baseline_rms, 0.001) 
                
                print("\n--- Calibration Complete ---")
                print(f"Background Noise Level: {self.baseline_rms:.5f}")
                print(f"Alarm will trigger if volume exceeds: {max(self.baseline_rms * SENSITIVITY_MULTIPLIER, MIN_ABSOLUTE_THRESHOLD):.5f}")
                print("----------------------------\n")
                print("🟢 SYSTEM ARMED: Listening for abnormal sounds...")
                
        else:
            meter_length = min(int(current_rms * 100), 20)
            print(f"Listening... [{'|'*meter_length:<20}] {current_rms:.4f}", end='\r')
            
            if current_rms > (self.baseline_rms * SENSITIVITY_MULTIPLIER) and current_rms > MIN_ABSOLUTE_THRESHOLD:
                print("\n\nSpike Detected:", current_rms)
                self.trigger_alarm()
                with self.q.mutex:
                    self.q.queue.clear()

    def start_monitoring(self, audio_file=None):
        print("Initializing Abnormal Sound Detector...")
        self.start_time = time.time()
        
        try:
            if audio_file:
                if audio_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    import os
                    import tempfile
                    from moviepy.editor import AudioFileClip
                    print(f"Extracting audio from video file: {audio_file}...")
                    audio = AudioFileClip(audio_file)
                    temp_wav = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
                    audio.write_audiofile(temp_wav, fps=SAMPLE_RATE, logger=None)
                    audio_file = temp_wav
                    print("Audio extraction complete.")

                print(f"Reading from audio file: {audio_file}")
                data, fs = sf.read(audio_file)
                if len(data.shape) > 1:
                    data = data[:, 0]  
                
                actual_block_size = int((BLOCK_SIZE / SAMPLE_RATE) * fs)
                num_blocks = len(data) // actual_block_size
                
                print(f"CALIBRATING for {CALIBRATION_DURATION} seconds...")
                for i in range(num_blocks):
                    block_start = i * actual_block_size
                    block_end = (i + 1) * actual_block_size
                    block = data[block_start:block_end]
                    
                    rms = np.sqrt(np.mean(block**2))
                    self.process_rms(rms)
                    
                    time.sleep(actual_block_size / fs)
                    
            else:
                print("Microphone starting...")
                with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
                    print(f"CALIBRATING for {CALIBRATION_DURATION} seconds. Please keep the room quiet...")
                    
                    while True:
                        current_rms = self.q.get()
                        self.process_rms(current_rms)
                        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")

def run(audio_file=None):
    detector = AbnormalSoundDetector()
    detector.start_monitoring(audio_file=audio_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Abnormal Sound Detector")
    parser.add_argument('--audio', type=str, default=None, help='Path to audio file (.wav, .flac) for testing instead of mic')
    args = parser.parse_args()
    run(args.audio)
