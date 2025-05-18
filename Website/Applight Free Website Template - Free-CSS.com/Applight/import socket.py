import socket
import json
import subprocess
import sounddevice as sd
import numpy as np
import os
import struct
import cv2
import time
import threading
import queue
from vosk import Model, KaldiRecognizer, SetLogLevel
from picamera2 import Picamera2
from contextlib import contextmanager
import pyaudio
import pvporcupine
from pathlib import Path

#Display Imports
import os
import time
import cv2
import threading
import random
from PIL import Image
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio
import LCD_2inch

# --- LCD Setup ---
lcd = LCD_2inch.LCD_2inch()
lcd.Init()
lcd.clear()

# --- Touch Sensor Setup ---
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
TOUCH_PIN = 17  # Change to your actual GPIO pin number
GPIO.setup(TOUCH_PIN, GPIO.IN)

# --- Servo Setup ---
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50

def set_servo_angle(channel, angle):
    pulse_length = 1000 + (angle / 180) * 1000
    duty_cycle = int(pulse_length / 1000000 * 50 * 65535)
    pca.channels[channel].duty_cycle = duty_cycle

# --- Emotion Setup ---
base_folder = "/home/nishchal/Desktop/emotions"
emotions = {
    "happy": os.path.join(base_folder, "happy"),
    "blink2": os.path.join(base_folder, "blink2"),
    "neutral": os.path.join(base_folder, "neutral"),
    "buffering": os.path.join(base_folder, "buffering"),
    "mic": os.path.join(base_folder, "mic"),
    "angry": os.path.join(base_folder, "angry"),
    "sad": os.path.join(base_folder, "sad"),
    "excited": os.path.join(base_folder, "excited"),
    "sleep": os.path.join(base_folder, "sleep"),
    "speaking": os.path.join(base_folder, "speaking"),
    "buffering2": os.path.join(base_folder, "buffering2")
}

current_emotion = "blink2"
emotion_lock = threading.Lock()

# Add these after the emotion_lock definition (around line 92)
last_interaction_time = time.time()
is_sleeping = False

# --- Touch Interaction Handler ---
def monitor_touch_interaction():
    global current_emotion, last_interaction_time, is_sleeping
    last_touch_time = time.time()
    touch_count = 0
    
    while True:
        if GPIO.input(TOUCH_PIN) == GPIO.HIGH:  # Adjust based on your sensor's logic
            last_interaction_time = time.time()
            is_sleeping = False
            current_time = time.time()
            # Reset count if too much time passed between touches
            if current_time - last_touch_time > 3.0:  # Added 3 second buffer for touch registration
                touch_count = 1
            else:
                touch_count += 1
            
            last_touch_time = current_time
            
            # Different actions based on touch count
            if touch_count == 1:
                with emotion_lock:
                    current_emotion = "sad"
                    # Droopy head movement for sad (channel 11)
                    set_servo_angle(11, 45)  # Downward tilt for sad
                    # Droopy hand movement
                    for ch in [0, 1]:
                        set_servo_angle(ch, 45)
            elif touch_count == 2:
                with emotion_lock:
                    current_emotion = "happy"
                    # Upward head movement for happy (channel 11)
                    set_servo_angle(11, 120)  # Upward tilt for happy
                    # Wave hands happily
                    for ch in [0, 1]:
                        set_servo_angle(ch, 110)
                    time.sleep(0.3)
                    for ch in [0, 1]:
                        set_servo_angle(ch, 70)
            elif touch_count == 3:
                with emotion_lock:
                    current_emotion = "angry"
                    # Head shaking movement for angry (channel 11)
                    set_servo_angle(11, 90)  # Center first
                    time.sleep(0.2)
                    set_servo_angle(11, 60)  # Left
                    time.sleep(0.2)
                    set_servo_angle(11, 120)  # Right
                    # Move hands aggressively
                    for ch in [0, 1]:
                        set_servo_angle(ch, 180)
                    time.sleep(0.3)
                    for ch in [0, 1]:
                        set_servo_angle(ch, 0)
            elif touch_count >= 4:
                with emotion_lock:
                    current_emotion = "excited"
                    # Nodding movement for excited (channel 11)
                    for i in range(2):
                        set_servo_angle(11, 120)  # Up
                        time.sleep(0.3)
                        set_servo_angle(11, 70)  # Down
                        time.sleep(0.3)
                    # Excited waving
                    for i in range(3):
                        for ch in [0, 1]:
                            set_servo_angle(ch, 120)
                        time.sleep(0.2)
                        for ch in [0, 1]:
                            set_servo_angle(ch, 60)
                        time.sleep(0.2)
                touch_count = 0
            
            # Set timer to revert to blink mode after 10 seconds
            threading.Timer(10.0, revert_to_blink).start()
            
            # Avoid multiple triggers
            time.sleep(0.5)
        time.sleep(0.1)
def revert_to_blink():
    global current_emotion
    with emotion_lock:
        # Only change if not already changed by something else
        if current_emotion in ["sad", "happy", "angry", "excited"]:
            current_emotion = "blink2"
            # Reset head position
            set_servo_angle(11, 90)  # Neutral head position



# Add this function after monitor_touch_interaction
def check_sleep_status():
    """Monitor idle time and trigger sleep mode when needed"""
    global current_emotion, last_interaction_time, is_sleeping
    
    while True:
        current_time = time.time()
        idle_time = current_time - last_interaction_time
        
        with emotion_lock:
            # Go to sleep if idle for 30 seconds (instead of 60s in neutral or 10s in buffer)
            if not is_sleeping and idle_time > 30:
                print("Going to sleep mode due to inactivity")
                current_emotion = "sleep"
                # Move head to sleep position
                set_servo_angle(11, 30)  # Drooping head for sleep
                # Hands drooping
                for ch in [0, 1]:
                    set_servo_angle(ch, 30)
                is_sleeping = True
            # If we're interacting, reset sleep state
            elif idle_time < 5:  # Any recent activity wakes up
                is_sleeping = False
                
        time.sleep(1)  # Check every second



# --- Helper: Responsive wait ---
def wait_with_check(duration, step=0.1):
    """Wait for 'duration' seconds, checking emotion every 'step' seconds."""
    global current_emotion
    waited = 0
    with emotion_lock:
        start_emotion = current_emotion

    while waited < duration:
        time.sleep(step)
        waited += step
        with emotion_lock:
            if current_emotion != start_emotion:
                return  # Exit early if emotion changed

# --- LCD Thread ---
def play_emotion_loop():
    global current_emotion
    while True:
        with emotion_lock:
            emotion = current_emotion
        
        frames_folder = emotions.get(emotion)
        if not frames_folder:
            time.sleep(0.1)
            continue

        frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
        for frame_file in frame_files:
            with emotion_lock:
                if current_emotion != emotion:
                    break
            img = cv2.imread(os.path.join(frames_folder, frame_file))
            if img is None:
                continue
            img = cv2.resize(img, (lcd.width, lcd.height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lcd.ShowImage(Image.fromarray(img))
            time.sleep(0.01)
# Add this function for smoother transitions
def smooth_servo_move(channel, target_angle, steps=10, delay=0.02):
    """Move servo smoothly from current position to target angle"""
    # Get approximate current angle based on duty cycle
    try:
        current_duty = pca.channels[channel].duty_cycle
        # Convert duty cycle back to angle (approximation)
        current_angle = ((current_duty / 65535) * 20 - 1) * 180
    except:
        # If we can't determine current position, just set directly
        set_servo_angle(channel, target_angle)
        return
        
    # Ensure we have valid values
    if current_angle < 0 or current_angle > 180:
        current_angle = 90  # Default to center if out of range
    
    # Calculate step size
    angle_diff = target_angle - current_angle
    step_size = angle_diff / steps
    
    # Move in steps
    for i in range(steps):
        next_angle = current_angle + step_size * (i + 1)
        set_servo_angle(channel, next_angle)
        time.sleep(delay)

        
# --- Servo Thread ---
def servo_behavior_loop():
    global current_emotion
    while True:
        with emotion_lock:
            emotion = current_emotion

        if emotion == "happy":
            # Happy hand movements - gentler and slower waving
            for ch in [0, 1]:
                angle = random.randint(70, 110)
                set_servo_angle(ch, angle)
                time.sleep(0.2)  # Stagger movements slightly
            set_servo_angle(11, 110)  # Head upward tilt for happy
            wait_with_check(0.8)
            wait_with_check(4.0)  # Longer pauses between movements

        elif emotion == "angry":
            # Angry hand movements - slower but still sharp
            set_servo_angle(11, 70)  # Slightly downward head tilt
            wait_with_check(0.4)
            
            for ch in [0, 1]:
                set_servo_angle(ch, 160)  # Not quite 180 for less aggressive
            wait_with_check(0.5)  # Slower movement

            for ch in [0, 1]:
                set_servo_angle(ch, 20)  # Not quite 0 for less aggressive
            wait_with_check(0.5)
            
        elif emotion == "sad":
            # Sad hand movements - slower drooping down
            set_servo_angle(11, 45)  # Head drooping down
            wait_with_check(0.6)
            
            for ch in [0, 1]:
                set_servo_angle(ch, 50)
            wait_with_check(1.0)
            
            for ch in [0, 1]:
                set_servo_angle(ch, 30)
            wait_with_check(3.0)
            
        elif emotion == "excited":
            # Excited hand movements - still energetic but smoother
            set_servo_angle(11, 120)  # Head up high
            for i in range(2):  # Fewer repetitions
                for ch in [0, 1]:
                    set_servo_angle(ch, 120)
                wait_with_check(0.4)  # Slower transitions
                for ch in [0, 1]:
                    set_servo_angle(ch, 60)
                wait_with_check(0.4)
            set_servo_angle(11, 90)  # Return head to neutral
            
        elif emotion == "blink2":
            # Gentle "looking around" motion
            down_angle = random.randint(65, 75)
            up_angle = random.randint(105, 115)
            rest_angle = 90

            for ch in [0, 1]:
                set_servo_angle(ch, down_angle)
            wait_with_check(0.3)  # Slightly slower

            for ch in [0, 1]:
                set_servo_angle(ch, up_angle)
            wait_with_check(0.3)

            for ch in [0, 1]:
                set_servo_angle(ch, rest_angle)
            wait_with_check(0.8)

            # Head movement
            head_angle = random.randint(80, 100)
            set_servo_angle(11, head_angle)
            wait_with_check(random.uniform(1.5, 3.0))  # Longer pauses

        elif emotion == "neutral":
            # Neutral position with slight movement
            down_angle = random.randint(70, 85)
            up_angle = random.randint(95, 110)
            rest_angle = 90

            # Head gentle movement
            head_angle = random.randint(85, 95)
            set_servo_angle(11, head_angle)
            wait_with_check(0.5)

            for ch in [0, 1]:
                set_servo_angle(ch, down_angle)
            wait_with_check(0.3)

            for ch in [0, 1]:
                set_servo_angle(ch, up_angle)
            wait_with_check(0.3)

            for ch in [0, 1]:
                set_servo_angle(ch, rest_angle)
            wait_with_check(0.8)

            wait_with_check(random.uniform(1.5, 3.0))
            
        elif emotion == "sleep":
            # Sleep position - very little movement
            set_servo_angle(11, 30)  # Head drooping
            for ch in [0, 1]:
                set_servo_angle(ch, 30)  # Arms drooping
            wait_with_check(5.0)  # Long pause between slight movements
            
            # Optional very slight movement while sleeping
            for ch in [0, 1]:
                current_angle = 30
                new_angle = random.randint(25, 35)
                set_servo_angle(ch, new_angle)
            wait_with_check(3.0)
            
        # Add new emotions for buffering2 and speaking
        elif emotion == "buffering2":
            # Waiting/processing motion
            head_tilt = random.randint(85, 95)
            set_servo_angle(11, head_tilt)
            
            # Slight hand movement like "calculating"
            for ch in [0, 1]:
                angle = random.randint(80, 100)
                set_servo_angle(ch, angle)
            wait_with_check(0.8)
            
            # Small circular motions
            for angle in range(85, 95, 2):
                set_servo_angle(0, angle)
                set_servo_angle(1, 180 - angle)
                wait_with_check(0.2)
                
        elif emotion == "speaking":
            # Talking animation with gestures
            # Slightly animated head movement
            for head_angle in [95, 90, 85, 90]:
                set_servo_angle(11, head_angle)
                wait_with_check(0.3)
                
            # Hand gestures while talking
            gestures = [
                (70, 70),  # Both hands down slightly
                (90, 90),  # Both hands neutral
                (110, 70), # Right hand up, left down
                (70, 110)  # Left hand up, right down
            ]
            
            for left_angle, right_angle in gestures:
                set_servo_angle(0, left_angle)
                set_servo_angle(1, right_angle)
                wait_with_check(0.6)
                
        else:
            wait_with_check(0.5)

# --- Input Thread ---

# def listen_for_input():
#     global current_emotion
#     while True:
#         new_emotion = input("Enter emotion (happy, blink2, angry): ").strip().lower()
#         if new_emotion in emotions:
#             with emotion_lock:
#                 current_emotion = new_emotion
#         else:
#             print(f"Emotion '{new_emotion}' not recognized.")

# --- Start Threads ---

# listen_for_input()  # Run input on main thread


# Suppress Vosk logs
SetLogLevel(-1)

# ========== Configuration ==========
INPUT_DEVICE = 2  # Your microphone device index
OUTPUT_DEVICE = "bluez_sink.32_96_49_25_C8_58.handsfree_head_unit"  # Replace with your Bluetooth sink name
LAPTOP_IP = "192.168.137.196"
VIDEO_PORT = 5000
DATA_PORT = 5001
VOSK_MODEL_PATH = "/home/nishchal/Desktop/vosk-model"
PIPER_TTS_PATH = "/home/nishchal/Desktop/models/piper/piper/piper"  # Path to Piper model
PIPER_MODEL_PATH="/home/nishchal/Desktop/models/piper/amy.onnx"
PORCUPINE_ACCESS_KEY = "sQcQAi/9P4RueqNXzoniLIXnvLq8BVVcGhgaIKffr7ucA2OC2zzG5A=="  # Your Porcupine access key
WAKE_WORD_MODEL = "/home/nishchal/Downloads/Hey-emo_en_raspberry-pi_v3_0_0.ppn"  # Path to your Porcupine wake word model
MAX_TIMEOUT = 60  # Increased timeout to 60 seconds

# ========== Logger Setup ==========
import logging
logger = logging.getLogger('pi_assistant')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('pi_assistant.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Audio queue setup for threaded playback
audio_queue = queue.Queue()
# Add a flag to track if audio is currently playing
audio_playing = threading.Event()

# ========== Connection Setup ==========

def establish_connection(retries=5, delay=2):
    for attempt in range(retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((LAPTOP_IP, DATA_PORT))
            logger.info("Connected to laptop!")
            print("Connected to laptop!")
            return sock
        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise ConnectionError("Could not establish connection to laptop.")

def connection_handshake(sock):
    try:
        sock.sendall(b"PI_RDY")
        print("Sent handshake to laptop.")
        confirmation = sock.recv(len(b"LAPTOP_OK"))
        if confirmation == b"LAPTOP_OK":
            print("Handshake confirmed.")
            return True
        return False
    except Exception as e:
        print(f"Handshake failed: {e}")
        logger.error(f"Handshake failed: {e}")
        return False

# ========== Streaming, Video ==========

def start_video_stream():
    """Initialize camera and UDP socket for streaming"""
    # Lower priority for video thread
    os.nice(10)
    
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 30}
    )
    picam2.configure(config)
    picam2.start()
    
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        while True:
            frame = picam2.capture_array()
            
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Encode frame to JPEG with quality adjustment
            _, jpeg_frame = cv2.imencode('.jpg', frame, 
                [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            jpeg_bytes = jpeg_frame.tobytes()
            
            # Send frame size followed by JPEG data
            size = struct.pack('!I', len(jpeg_bytes))
            udp_socket.sendto(size + jpeg_bytes, (LAPTOP_IP, VIDEO_PORT))
            
            time.sleep(0.033)  # ~30fps
            
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        udp_socket.close()
        logger.info("Video stream shutdown")

# Wake word detection using Porcupine
def wake_word_detection():
    """Listen for wake word using Porcupine"""
    try:
        # Create Porcupine instance
        porcupine_instance = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keyword_paths=[WAKE_WORD_MODEL]
        )
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        audio_stream = audio.open(
            rate=porcupine_instance.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine_instance.frame_length,
            input_device_index=INPUT_DEVICE
        )
        
        print("Listening for wake word 'Hey Emo'...")
        
        # Main detection loop
        while True:
            pcm = audio_stream.read(porcupine_instance.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine_instance.frame_length, pcm)
            
            keyword_index = porcupine_instance.process(pcm)
            if keyword_index >= 0:
                print("Wake word detected!")
                global last_interaction_time, is_sleeping
                last_interaction_time = time.time()
                is_sleeping = False
                break
                
    except Exception as e:
        logger.error(f"Wake word detection error: {e}")
        print(f"Wake word detection error: {e}")
        return False
    finally:
        if 'audio_stream' in locals() and audio_stream:
            audio_stream.close()
        if 'audio' in locals() and audio:
            audio.terminate()
        if 'porcupine_instance' in locals() and porcupine_instance:
            porcupine_instance.delete()
    
    return True

def speech_to_text(max_listen_time=10):
    import tempfile
    import soundfile as sf
    import numpy as np
    try:
        # Record audio to a temporary WAV file for Whisper
        samplerate = 16000
        channels = 1
        logger.info("Recording audio for Whisper STT with silence detection...")
        print("Recording for Whisper with silence detection...")
        import sounddevice as sd
        import time
        import collections
        silence_threshold = 500  # Adjust as needed
        silence_max_blocks = int(2.5* samplerate / 1024)  # 3 seconds of silence (block size 1024)
        blocksize = 1024
        max_blocks = int(max_listen_time * samplerate / blocksize)
        audio_frames = []
        silence_buffer = collections.deque(maxlen=silence_max_blocks)
        start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16', blocksize=blocksize) as stream:
                print("Listening for speech (Whisper)...")
                while True:
                    block, _ = stream.read(blocksize)
                    audio_frames.append(block)
                    peak = np.abs(block).max()
                    silence_buffer.append(peak < silence_threshold)
                    # If silence for >4s, stop
                    if all(silence_buffer) and len(silence_buffer) == silence_max_blocks:
                        print("Detected >2s silence, stopping recording.")
                        break
                    # Or if max duration reached
                    if (time.time() - start_time) > max_listen_time or len(audio_frames) >= max_blocks:
                        print("Max listen time for Whisper reached.")
                        break
            audio = np.concatenate(audio_frames, axis=0)
            sf.write(tmpfile.name, audio, samplerate)
            tmpfile.flush()
            whisper_success = False
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel(model_size_or_path="tiny", device="cpu", compute_type="int8")
                segments, info = model.transcribe(tmpfile.name, beam_size=1)
                text = " ".join([segment.text for segment in segments])
                # Remove trailing period for username detection
                username = text.strip()
                if username.endswith('.'):
                    username = username[:-1]
                logger.info(f"Whisper recognized: {username}")
                print(f"Whisper recognized: {username}")
                whisper_success = True
                # Clean up temp file
                try:
                    os.unlink(tmpfile.name)
                except Exception:
                    pass
                return username.strip()
            except Exception as e:
                logger.error(f"Whisper STT failed: {e}")
                print(f"Whisper STT failed: {e}, falling back to Vosk")
            # Clean up temp file before fallback
            try:
                os.unlink(tmpfile.name)
            except Exception:
                pass
    except Exception as rec_e:
        logger.error(f"Audio recording failed: {rec_e}")
        print(f"Audio recording failed: {rec_e}, falling back to Vosk")
    # --- FALLBACK TO VOSK ---
    try:
        if not os.path.exists(VOSK_MODEL_PATH):
            logger.error("Vosk model not found!")
            print("Vosk model not found!")
            return ""
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)
        print("Initializing microphone for Vosk fallback...")
        with sd.RawInputStream(
            samplerate=48000,
            blocksize=8000,
            dtype='int16',
            channels=1,
            device=INPUT_DEVICE
        ) as stream:
            print("Listening for voice input (Vosk fallback)...")
            silence_counter = 0
            max_silence_blocks = 50  # ~3s at 48kHz
            full_text = ""
            last_partial = ""
            start_time = time.time()
            while True:
                if time.time() - start_time > max_listen_time:
                    print("Max listening time reached (Vosk fallback).")
                    return (full_text + " " + last_partial).strip()
                data, _ = stream.read(8000)
                audio_np = np.frombuffer(data, dtype=np.int16)
                peak_volume = np.abs(audio_np).max()
                if peak_volume < 500:
                    silence_counter += 1
                    if silence_counter > max_silence_blocks:
                        print("Timeout: No speech detected (Vosk fallback).")
                        return (full_text + " " + last_partial).strip()
                    continue
                else:
                    silence_counter = 0
                audio_downsampled = audio_np[::3]
                if recognizer.AcceptWaveform(audio_downsampled.tobytes()):
                    result = json.loads(recognizer.Result())
                    if result.get("text"):
                        text = result["text"]
                        full_text += " " + text
                        logger.info(f"Vosk recognized: {text}")
                        print(f"Vosk recognized: {text}")
                        last_partial = ""
                else:
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    if partial:
                        last_partial = partial
                        print(f"Partial (Vosk): {partial}")
    except Exception as vosk_e:
        logger.error(f"Vosk STT failed: {vosk_e}")
        print(f"Vosk STT failed: {vosk_e}")
        return ""


def audio_worker():
    """Thread worker for audio playback to prevent blocking"""
    while True:
        text = audio_queue.get()
        if text is None:
            break
        try:
            # Set the flag to indicate audio is playing
            audio_playing.set()
            text_to_speech(text)
        except Exception as e:
            logger.error(f"Audio worker error: {e}")
        finally:
            # Clear the flag to indicate audio playback is complete
            audio_playing.clear()
            audio_queue.task_done()



# Text-to-speech using Piper

def text_to_speech(text):
    """Convert text to speech using Piper TTS"""
    logger.info(f"[TTS] Requested to speak: '{text}'")
    print(f"[TTS] Requested to speak: '{text}'")
    try:
        # Set speaking animation at the start
        global current_emotion
        with emotion_lock:
            current_emotion = "speaking"
            
        # Create a temporary file for the text
        temp_dir = Path("/t/tmp/piper_tts")
        temp_dir.mkdir(exist_ok=True)
        logger.info(f"[TTS] Temp dir: {temp_dir}")
        print(f"[TTS] Temp dir: {temp_dir}")
        temp_file = temp_dir / "text.txt"
        with open(temp_file, "w") as f:
            f.write(text)
        logger.info(f"[TTS] Wrote text to: {temp_file}")
        print(f"[TTS] Wrote text to: {temp_file}")
        # Use Piper TTS to generate speech
        output_file = temp_dir / "output.wav"
        logger.info(f"[TTS] Output file will be: {output_file}")
        print(f"[TTS] Output file will be: {output_file}")
        # Check if Piper executable exists and is executable
        logger.info(f"[TTS] Checking Piper executable at: {PIPER_TTS_PATH}")
        print(f"[TTS] Checking Piper executable at: {PIPER_TTS_PATH}")
        if not (os.path.isfile(PIPER_TTS_PATH) and os.access(PIPER_TTS_PATH, os.X_OK)):
            logger.error("[TTS] Piper TTS executable not found or not executable. Falling back to espeak.")
            print("[TTS] Piper TTS executable not found or not executable. Falling back to espeak.")
            subprocess.run(["espeak-ng", "-s150", "-ven+f3", text])
            # Set back to neutral after speaking
            with emotion_lock:
                current_emotion = "neutral"
            return
        # Run Piper TTS with model, handle permission errors explicitly
        try:
            logger.info(f"[TTS] Running Piper: {PIPER_TTS_PATH} --model {PIPER_MODEL_PATH} --output_file {output_file} --text_file {temp_file}")
            print(f"[TTS] Running Piper: {PIPER_TTS_PATH} --model {PIPER_MODEL_PATH} --output_file {output_file} --text_file {temp_file}")
            subprocess.run([
                PIPER_TTS_PATH,
                "--model", PIPER_MODEL_PATH,
                "--output_file", str(output_file),
                "--text_file", str(temp_file)
            ], check=True)
            logger.info("[TTS] Piper ran successfully.")
            print("[TTS] Piper ran successfully.")
        except PermissionError as pe:
            logger.error(f"[TTS] Permission denied when executing Piper: {pe}. Falling back to espeak.")
            print(f"[TTS] Permission denied when executing Piper: {pe}. Falling back to espeak.")
            subprocess.run(["espeak-ng", "-s150", "-ven+f3", text])
            # Set back to neutral after speaking
            with emotion_lock:
                current_emotion = "neutral"
            return
        except Exception as e:
            logger.error(f"[TTS] Error running Piper: {e}. Falling back to espeak.")
            print(f"[TTS] Error running Piper: {e}. Falling back to espeak.")
            subprocess.run(["espeak-ng", "-s150", "-ven+f3", text])
            # Set back to neutral after speaking
            with emotion_lock:
                current_emotion = "neutral"
            return
        
        # Play the generated audio using aplay
        logger.info(f"[TTS] Playing generated audio: {OUTPUT_DEVICE}")
        print(f"[TTS] Playing generated audio: {OUTPUT_DEVICE}")
        subprocess.run(["aplay", "-D",OUTPUT_DEVICE,str(output_file)], check=True)
        logger.info("[TTS] Audio playback complete.")
        print("[TTS] Audio playback complete.")
        
        # Clean up temporary files
        logger.info(f"[TTS] Cleaning up files: {temp_file}, {output_file}")
        print(f"[TTS] Cleaning up files: {temp_file}, {output_file}")
        temp_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)
        logger.info("[TTS] Cleanup complete.")
        print("[TTS] Cleanup complete.")
        
        # Set back to neutral after successful playback
        with emotion_lock:
            current_emotion = "neutral"
            
    except Exception as e:
        logger.error(f"[TTS] Text-to-speech error: {e}")
        print(f"[TTS] Text-to-speech error: {e}, falling back to espeak")
        # Fallback to espeak if Piper fails
        try:
            logger.info("[TTS] Entering fallback to espeak with pulseaudio context.")
            print("[TTS] Entering fallback to espeak with pulseaudio context.")
            with pulseaudio_ctx():
                subprocess.run(["espeak-ng", "-s150", "-ven+f3", text])
        except Exception as e_fallback:
            logger.error(f"[TTS] Fallback TTS error: {e_fallback}")
            print(f"[TTS] Fallback TTS error: {e_fallback}")
        finally:
            # Make sure to set back to neutral even if fallback fails
            with emotion_lock:
                current_emotion = "neutral"


# Fix: Properly implement context manager for pulseaudio
@contextmanager
def pulseaudio_ctx():
    """Context manager for pulseaudio connections"""
    try:
        # Set up pulseaudio connection
        subprocess.run(["pactl", "set-default-sink", OUTPUT_DEVICE], check=True)
        yield
    finally:
        # Clean up pulseaudio connection if needed
        pass

# ========== Communication ==========

def send_text(sock, data):
    """Send data to laptop with proper error handling and reconnection"""
    try:
        payload = json.dumps(data).encode('utf-8')
        sock.sendall(struct.pack("!I", len(payload)) + payload)
        print(f"Sent to laptop: {data}")
        return sock
    except (BrokenPipeError, ConnectionResetError):
        logger.warning("Connection lost. Reconnecting...")
        print("Connection lost. Reconnecting...")
        try:
            new_sock = establish_connection()
            return send_text(new_sock, data)
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            print(f"Reconnection failed: {e}")
            return None
    except Exception as e:
        logger.error(f"Send error: {e}")
        print(f"Send error: {e}")
        return None
        
def receive_response(sock, base_timeout=15):
    """Receive response with dynamic timeout adjustment"""
    dynamic_timeout = base_timeout
    start_time = time.time()
    
    while time.time() - start_time < MAX_TIMEOUT:  # Increased to 60 seconds max wait
        try:
            sock.settimeout(dynamic_timeout)
            
            raw_len = sock.recv(4)
            if not raw_len or len(raw_len) != 4:
                logger.warning("No or incomplete length header received.")
                print("No or incomplete length header received.")
                dynamic_timeout *= 1.3  # Exponential backoff
                continue

            msg_len = struct.unpack("!I", raw_len)[0]
            
            # Sanity check on message length
            if msg_len > 1000000:  # 1MB limit
                logger.warning(f"Message length too large: {msg_len} bytes")
                print(f"Message length too large: {msg_len} bytes")
                dynamic_timeout *= 1.3
                continue
                
            received = b""

            while len(received) < msg_len:
                chunk = sock.recv(min(4096, msg_len - len(received)))
                if not chunk:
                    break
                received += chunk

            if len(received) < msg_len:
                logger.warning(f"Incomplete message: got {len(received)} of {msg_len} bytes")
                print(f"Incomplete message: got {len(received)} of {msg_len} bytes")
                dynamic_timeout *= 1.3
                continue

            response = received.decode('utf-8').strip()
            if response:
                print(f"Received from laptop: {response}")
                return response
            else:
                logger.warning("Empty response received")
                print("Empty response received, retrying...")
                dynamic_timeout *= 1.3

        except socket.timeout:
            logger.warning(f"Timeout after {dynamic_timeout}s: No response received")
            print(f"Timeout after {dynamic_timeout}s: No response received")
            dynamic_timeout *= 1.3

        except Exception as e:
            logger.error(f"Receive error: {e}")
            print(f"Receive error: {e}")
            dynamic_timeout *= 1.3

    logger.error("Failed to receive response after maximum wait time")
    print("Failed to receive response after maximum wait time")
    return ""

class ProtocolError(Exception):
    """Custom exception for protocol errors"""
    pass

def extract_username(username_text):
    """Helper function to extract username from text"""
    if not username_text:
        return ""
    
    username = username_text
    if "my name is" in username_text.lower():
        username = username_text.lower().split("my name is")[1].strip()
    elif "i am" in username_text.lower():
        username = username_text.lower().split("i am")[1].strip()
    
    return username

# Start touch monitor thread
touch_monitor_thread = threading.Thread(target=monitor_touch_interaction, daemon=True)
touch_monitor_thread.start()

# Add this right before the main function loop (after the touch monitor thread start)
sleep_monitor_thread = threading.Thread(target=check_sleep_status, daemon=True)
sleep_monitor_thread.start()

# ========== Main Function ==========
def main():
    global current_emotion
    global last_interaction_time

    with emotion_lock:
        current_emotion = "buffering"

    last_interaction_time = time.time()
    
    # Initialize with priority management
    video_thread = threading.Thread(target=start_video_stream, name="VideoStream")
    video_thread.daemon = True
    video_thread.start()
    
    # Start audio worker thread
    audio_thread = threading.Thread(target=audio_worker, name="AudioWorker")
    audio_thread.daemon = True
    audio_thread.start()
    
    data_sock = None

    # Start Servo and Display Thread
    threading.Thread(target=play_emotion_loop, daemon=True).start()
    threading.Thread(target=servo_behavior_loop, daemon=True).start()

    # Initialize body motor to neutral position
    set_servo_angle(11, 90)  # Body motor at neutral position

    try:
        # Connection phase with staggered retries
        data_sock = establish_connection()
        if not connection_handshake(data_sock):
            raise ConnectionError("Handshake failed")

        # Command sequence with acknowledgement
        def execute_command(command, expect_response=True):
            nonlocal data_sock
            data_sock = send_text(data_sock, command)
            if data_sock is None:
                raise ConnectionError("Failed to send message")
            if expect_response:
                response = receive_response(data_sock)
                return response
            return None

        # Protocol sequence
        while True:
            # Use the wake word detection instead of listening for "hello"
            print("Waiting for wake word 'Hey Emo'...")
            with emotion_lock:
                current_emotion = "neutral"    # looking around
                
            # Subtle body movements while in idle/waiting state
            set_servo_angle(11, 85)  # Slight body tilt to one side
            time.sleep(3)
            set_servo_angle(11, 95)  # Slight body tilt to other side
            time.sleep(3)
            set_servo_angle(11, 90)  # Return to center

            if wake_word_detection():
                # Make an immediate audio response when wake word is detected
                # Turn body toward user when wake word detected
                set_servo_angle(11, 110)  # Turn body toward user
                time.sleep(0.3)
                set_servo_angle(11, 90)  # Return to neutral
                
                text_to_speech("Hello! I am your mental health assistant. What's your name?")
                with emotion_lock:
                    current_emotion = "blink2"   # Listening 
                
                # Get username with inline validation
                username = ""
                while not username:
                    # Attentive body posture while listening
                    set_servo_angle(11, 100)  # Slight lean forward to show attention
                    username_text = speech_to_text(max_listen_time=12)
                    set_servo_angle(11, 90)  # Return to neutral
                    
                    username = extract_username(username_text)
                    if not username:
                        # Show slight disappointment with body language
                        set_servo_angle(11, 80)  # Lean back slightly
                        time.sleep(0.3)
                        text_to_speech("Let's try again. Please say your name clearly.")
                        set_servo_angle(11, 90)  # Return to neutral
                        with emotion_lock:
                            current_emotion = "blink2"   # Listening 

                # Start new interaction cycle with the username
                response = execute_command({
                    "command": "username",
                    "username": username
                })
                
                # After username is processed, start the conversation
                execute_command({"command": "start", "text": "Conversation started"})
                
                # Handle response for username
                try:
                    response_data = json.loads(response)
                    if response_data.get("status") == "success":
                        with emotion_lock:
                            current_emotion = "happy"
                        # Wave hand while greeting and move body with excitement
                        set_servo_angle(11, 105)  # Body leans slightly forward in enthusiasm
                        set_servo_angle(0, 150)  # Lift one hand to wave
                        time.sleep(0.5)
                        set_servo_angle(0, 90)   # Hand back to neutral
                        set_servo_angle(11, 90)  # Body back to neutral
                        text_to_speech(f"Nice to meet you, {username}. How's your day going?")
                        with emotion_lock:
                            current_emotion = "mic"   # Listening
                    else:
                        # Body shows uncertainty
                        set_servo_angle(11, 80)  # Lean back as if unsure
                        text_to_speech(response_data.get("message", "Registration issue. Let's try again."))
                        set_servo_angle(11, 90)  # Return to neutral
                        continue
                except (json.JSONDecodeError, TypeError):
                    # Body shows confusion
                    set_servo_angle(11, 75)  # More pronounced lean back 
                    time.sleep(0.3)
                    set_servo_angle(11, 105)  # Then forward
                    time.sleep(0.3)
                    set_servo_angle(11, 90)  # Back to neutral
                    text_to_speech("System busy. Please try again.")
                    continue

                # Conversation loop with timeout control
                convo_timeout = time.time() + 300  # 5-minute session
                while time.time() < convo_timeout:
                    # Wait for any ongoing audio playback to finish before listening again
                    while audio_playing.is_set():
                        time.sleep(0.5)
                    
                    # Attentive body posture while listening
                    set_servo_angle(11, 100)  # Slight lean forward to show attention    
                    user_text = speech_to_text(max_listen_time=7)
                    set_servo_angle(11, 90)  # Return to neutral
                    
                    if not user_text:
                        with emotion_lock:
                            current_emotion = "blink2"   # Listening
                        # Body shows confusion with a slight tilt
                        set_servo_angle(11, 85)  # Tilt slightly
                        time.sleep(0.3)
                        set_servo_angle(11, 95)  # Tilt other way
                        time.sleep(0.3)
                        set_servo_angle(11, 90)  # Return to center
                        text_to_speech("I didn't hear anything. Let's try again.")
                        continue
                    
                    if "bye" in user_text.lower():
                        execute_command({
                            "command": "stop",
                            "text": "Conversation stopped"
                        }, expect_response=False)
                        with emotion_lock:
                            current_emotion = "happy"
                        # Wave goodbye with hand motion and body movement
                        # First bow slightly with body
                        set_servo_angle(11, 110)  # Lean forward in a slight bow
                        time.sleep(0.5)
                        set_servo_angle(11, 90)  # Return to neutral
                        
                        # Then wave with hand
                        for ch in [0]:
                            set_servo_angle(ch, 45)
                        time.sleep(0.3)
                        for ch in [0]:
                            set_servo_angle(ch, 170)
                        time.sleep(0.5)
                        # Wave back and forth with body movement
                        for i in range(2):
                            set_servo_angle(0, 130)
                            set_servo_angle(11, 95)  # Body leans slightly to one side
                            time.sleep(0.2)
                            set_servo_angle(0, 170)
                            set_servo_angle(11, 85)  # Body leans slightly to other side
                            time.sleep(0.2)
                        # Return to neutral
                        set_servo_angle(0, 90)
                        set_servo_angle(11, 90)
                        text_to_speech(f"Goodbye, {username}! Have a great day.")
                        break
                        
                    # Send and process with sequence tracking
                    if user_text and len(user_text.strip()) > 3:
                        # Body shows processing movement
                        set_servo_angle(11, 95)  # Slight tilt as if thinking
                        response = execute_command({"text": user_text})
                        if response:
                            with emotion_lock:
                                current_emotion = "blink2"   # Listening 
                            # Body shows active response
                            set_servo_angle(11, 105)  # Lean forward to deliver response
                            time.sleep(0.3)
                            set_servo_angle(11, 90)  # Return to neutral
                            # Queue response for smooth playback
                            audio_queue.put(response)
                            
                            # Subtle body movements while talking
                            for i in range(3):
                                set_servo_angle(11, 92)  # Small movements to simulate talking
                                time.sleep(0.7)
                                set_servo_angle(11, 88)
                                time.sleep(0.7)
                            set_servo_angle(11, 90)  # Return to neutral
                            
                            # Wait for audio playback to complete before continuing
                            print("Waiting for audio playback to complete...")
                            audio_queue.join()
                        else:
                            # Body shows confusion
                            set_servo_angle(11, 80)  # Lean back in confusion
                            time.sleep(0.3)
                            set_servo_angle(11, 90)  # Return to neutral
                            text_to_speech("Sorry, I didn't get that. Could you repeat?")

    except ConnectionError as e:
        # Body shows distress
        set_servo_angle(11, 70)  # Major lean back showing problem
        logger.error(f"Connection error: {e}")
        text_to_speech("I lost connection to the server. Please try again later.")
        set_servo_angle(11, 90)  # Return to neutral
    except ProtocolError as e:
        # Body shows confusion
        set_servo_angle(11, 75)  # Lean back
        logger.error(f"Protocol error: {e}")
        text_to_speech("There was a communication error with the system.")
        set_servo_angle(11, 90)  # Return to neutral
    except Exception as e:
        # Body shows malfunction
        set_servo_angle(11, 65)  # Major lean back
        time.sleep(0.3)
        set_servo_angle(11, 115)  # Then forward
        time.sleep(0.3)
        set_servo_angle(11, 90)  # Back to neutral
        logger.error(f"Main loop failure: {e}")
        text_to_speech("I encountered a problem and need to restart. Please try again.")
    finally:
        # Return body to neutral position before cleanup
        set_servo_angle(11, 90)
        if data_sock:
            data_sock.close()
        # Signal audio thread to terminate
        audio_queue.put(None)
        logger.info("Cleaned up resources.")
        print("Cleaned up resources.")
        
if _name_ == "_main_":
    main()