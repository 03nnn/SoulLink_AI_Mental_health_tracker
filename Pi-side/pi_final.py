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
from pathlib import Path
from picamera2 import Picamera2
import pyaudio
import pvporcupine
import tempfile
import soundfile as sf
import logging
import RPi.GPIO as GPIO
from PIL import Image
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import busio
import LCD_2inch

# --- Logging Setup ---
logger = logging.getLogger('pi_assistant')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('pi_assistant.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- LCD Setup ---
lcd = LCD_2inch.LCD_2inch()
lcd.Init()
lcd.clear()

# --- Touch Sensor Setup ---
GPIO.setmode(GPIO.BCM)
TOUCH_PIN = 17  # Touch sensor GPIO pin
GPIO.setup(TOUCH_PIN, GPIO.IN)

# --- Servo Setup ---
i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# --- Configuration ---
INPUT_DEVICE = 2  # Microphone device index
OUTPUT_DEVICE = "bluez_sink.32_96_49_25_C8_58.handsfree_head_unit"  # Bluetooth output sink name
LAPTOP_IP = "192.168.137.196" # laptop(server) IP
VIDEO_PORT = 5000
DATA_PORT = 5001
PORCUPINE_ACCESS_KEY = "*"# enter the porcupine access key
WAKE_WORD_MODEL = "/Hey-emo_en_raspberry-pi_v3_0_0.ppn"
MAX_TIMEOUT = 300 # Max timeout in seconds
BUFFER_TIME_AFTER_CONVERSATION = 15  # Buffer time after conversation in seconds

# --- Emotion Setup ---
base_folder = " /emotions"
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

# --- Global Variables ---
current_emotion = "blink2"
emotion_lock = threading.Lock()
last_interaction_time = time.time()
is_sleeping = False
audio_queue = queue.Queue()
audio_playing = threading.Event()

# --- Functions ---
def set_servo_angle(channel, angle, speed=0.01):
    """Set servo angle with smooth transition for better movement"""
    # Get current position (if already set)
    try:
        current_duty = pca.channels[channel].duty_cycle
        current_angle = (current_duty / 65535 * 20) * 180 - 180
    except:
        # If it's first move, assume natural position
        if channel == 11:  # Body
            current_angle = 75
        elif channel == 0:  # Left hand
            current_angle = 120
        elif channel == 1:  # Right hand
            current_angle = 60
        else:
            current_angle = angle  # Default to target angle

    # For smoother movement, we'll step through angles
    steps = int(abs(angle - current_angle) / 5) + 1
    
    # Adjust speed based on channel (slower for body)
    actual_speed = speed * 3 if channel == 11 else speed
    
    for step in range(steps):
        intermediate_angle = current_angle + (angle - current_angle) * (step + 1) / steps
        pulse_length = 1000 + (intermediate_angle / 180) * 1000
        duty_cycle = int(pulse_length / 1000000 * 50 * 65535)
        pca.channels[channel].duty_cycle = duty_cycle
        time.sleep(actual_speed)

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

def revert_to_blink():
    """Revert to blink mode after displaying an emotion"""
    global current_emotion
    with emotion_lock:
        # Only change if not already changed by something else
        if current_emotion in ["sad", "happy", "angry", "excited"]:
            current_emotion = "blink2"
            # Reset head position
            set_servo_angle(11, 75)  # Neutral head position

def monitor_touch_interaction():
    """Monitor touch sensor and respond to interactions"""
    global current_emotion, last_interaction_time, is_sleeping
    last_touch_time = time.time()
    touch_count = 0
    touch_buffer_active = False
    touch_buffer_start = 0
    
    while True:
        if GPIO.input(TOUCH_PIN) == GPIO.HIGH:
            last_interaction_time = time.time()
            is_sleeping = False
            current_time = time.time()
            
            # Start the touch buffer if not active
            if not touch_buffer_active:
                touch_buffer_active = True
                touch_buffer_start = current_time
                touch_count = 1
            else:
                # If within buffer time, increment count
                if current_time - last_touch_time < 3.0:
                    touch_count += 1
                else:
                    # If too much time has passed, reset count
                    touch_count = 1
                    touch_buffer_start = current_time
            
            last_touch_time = current_time
            time.sleep(0.2)  # Debounce
            
        # Process touch count after buffer time
        if touch_buffer_active and time.time() - touch_buffer_start >= 2.0:
            # Different actions based on touch count
            if touch_count == 1:
                with emotion_lock:
                    current_emotion = "sad"
                    # Droopy head movement for sad
                    set_servo_angle(11, 30)  # Downward tilt for sad
                    # Droopy hand movement
                    set_servo_angle(1, 45)
                    set_servo_angle(0,135)
            elif touch_count == 2:
                with emotion_lock:
                    current_emotion = "happy"
                    # Wave hands happily
                    for ch in [0, 1]:
                        set_servo_angle(ch, 110)
                    time.sleep(0.5)
                    for ch in [0, 1]:
                        set_servo_angle(ch, 70)
                    set_servo_angle(11, 75)  # Upward tilt for happy
                    time.sleep(0.3)
            elif touch_count == 3:
                with emotion_lock:
                    current_emotion = "angry"
                    # Head shaking movement for angry
                    set_servo_angle(11, 75)  # Center first
                    time.sleep(0.3)
                    set_servo_angle(11, 40) # Left
                    time.sleep(0.3)
                    set_servo_angle(11, 110)  # Right
                    # Move hands aggressively
                    set_servo_angle(1, 180)
                    set_servo_angle(0, 0)
            elif touch_count >= 4:
                with emotion_lock:
                    current_emotion = "excited"
                    # Nodding movement for excited
                    set_servo_angle(11,75)
                    # Excited waving
                    for i in range(2):  # Reduced repetitions
                        for ch in [0, 1]:
                            set_servo_angle(ch, 120)
                        time.sleep(0.3)
                        for ch in [0, 1]:
                            set_servo_angle(ch, 60)
                        time.sleep(0.3)
            
            # Reset touch buffer
            touch_buffer_active = False
            
            # Set timer to revert to blink mode after 10 seconds
            threading.Timer(10.0, revert_to_blink).start()
            
        time.sleep(0.1)


def check_sleep_status():
    """Monitor idle time and trigger sleep mode when needed"""
    global current_emotion, last_interaction_time, is_sleeping
    
    sleep_countdown_started = False
    sleep_countdown_start = 0
    
    while True:
        current_time = time.time()
        
        with emotion_lock:
            # Only start sleep countdown if in blink2 mode
            if current_emotion == "blink2" and not sleep_countdown_started:
                sleep_countdown_started = True
                sleep_countdown_start = current_time
            
            # Reset countdown if not in blink2 mode
            elif current_emotion != "blink2":
                sleep_countdown_started = False
            
            # Check if we should go to sleep
            if sleep_countdown_started and not is_sleeping and (current_time - sleep_countdown_start > 30):
                logger.info("Going to sleep mode due to inactivity")
                current_emotion = "sleep"
                # Move head to sleep position
                set_servo_angle(11, 30)  # Drooping head for sleep
                # Hands drooping
                for ch in [0, 1]:
                    set_servo_angle(ch, 30)
                is_sleeping = True
                sleep_countdown_started = False
        
        time.sleep(1)

def play_emotion_loop():
    """Display emotions on LCD screen"""
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

def calculate_happy_angle(i, part):
    if i <= 30:
        return {'right': 90+i, 'left': 90-i, 'base': 90-i}[part]
    elif i <= 90:
        return {'right': 150-i, 'left': i+30, 'base': i+30}[part]
    else:
        return {'right': i-30, 'left': 210-i, 'base': 210-i}[part]


def servo_behavior_loop():
    """Control servo movements based on current emotion"""
    global current_emotion
    while True:
        with emotion_lock:
            emotion = current_emotion
            
        if emotion == "happy":
            # Smooth sinusoidal arm waves with coordinated head movement
            for _ in range(1):  # Single cycle but continuous until emotion changes
                for i in range(30, 120,2):  # Step by 1 for smoothness
                    if emotion != "happy":
                        break
                    angle_r = calculate_happy_angle(i, 'right')
                    angle_l = calculate_happy_angle(i, 'left')
                    angle_b = calculate_happy_angle(i, 'base')
                    set_servo_angle(1, angle_r)
                    set_servo_angle(0, angle_l)
                    set_servo_angle(11, angle_b)
                    time.sleep(0.016)  # Adjusted for smooth 60fps-like motion
                wait_with_check(1.0)

        elif emotion == "angry":
            # Angry hand movements - slower but still sharp
            for i in range(0,1):  # Fewer repetitions
                for ch in [0,1,11]:
                    if emotion != "angry":
                          break
                    set_servo_angle(1, 180)
                    set_servo_angle(0,0)
                    set_servo_angle(ch, 30)
                    set_servo_angle(ch ,120)
                wait_with_check(0.5)
          
            
        elif emotion == "sad":
            # Sad hand movements - slower drooping down
            set_servo_angle(11, 35)  # Head drooping down
            wait_with_check(0.7)
            
            for ch in [0, 1]:
                set_servo_angle(ch, 50)
            wait_with_check(1.2)
            
            for ch in [0, 1]:
                set_servo_angle(ch, 30)
            wait_with_check(3.0)
            
        elif emotion == "excited":
            # Excited hand movements - still energetic but smoother
           
            for i in range(2):  # Fewer repetitions
                for ch in [11]:
                    set_servo_angle(ch, 120)
                wait_with_check(0.5)
                for ch in [11]:
                    set_servo_angle(ch, 30)
                wait_with_check(0.5)
            set_servo_angle(11, 75)  # Return head to neutral
            
        elif emotion == "blink2":
            # Gentle "looking around" motion
            down_angle = random.randint(65, 75)
            up_angle = random.randint(105, 115)
            rest_angle = 90

            for ch in [0, 1]:
                set_servo_angle(ch, down_angle)
            wait_with_check(0.4)

            for ch in [0, 1]:
                set_servo_angle(ch, up_angle)
            wait_with_check(0.4)

            for ch in [0, 1]:
                set_servo_angle(ch, rest_angle)
            wait_with_check(0.9)

            # Head movement
            head_angle = random.randint(60, 90)
            set_servo_angle(11, head_angle)
            wait_with_check(random.uniform(2.0, 3.5))

        elif emotion == "neutral":
            # Neutral position with slight movement
            down_angle = random.randint(70, 85)
            up_angle = random.randint(95, 110)
            rest_angle = 90

            # Head gentle movement
            head_angle = random.randint(65, 85)
            set_servo_angle(11, head_angle)
            wait_with_check(0.6)

            for ch in [0, 1]:
                set_servo_angle(ch, down_angle)
            wait_with_check(0.4)

            for ch in [0, 1]:
                set_servo_angle(ch, up_angle)
            wait_with_check(0.4)

            for ch in [0, 1]:
                set_servo_angle(ch, rest_angle)
            wait_with_check(0.9)

            wait_with_check(random.uniform(1.8, 3.5))
            
        elif emotion == "sleep":
            # Sleep position - very little movement
            set_servo_angle(11, 75) 
            set_servo_angle(1, 90)
            set_servo_angle(0, 90) # Head drooping
          # Arms drooping
            wait_with_check(5.0)

            
        elif emotion == "buffering2":
            # Waiting/processing motion
            head_tilt = random.randint(65, 85)
            set_servo_angle(11, head_tilt)
            
            # Slight hand movement like "calculating"
            for ch in [0, 1]:
                angle = random.randint(80, 100)
                set_servo_angle(ch, angle)
            wait_with_check(0.9)
            
            # Small circular motions
            for angle in range(85, 95, 2):
                set_servo_angle(0, angle)
                set_servo_angle(1, 180 - angle)
                wait_with_check(0.25)
                
        elif emotion == "speaking":
            # Talking animation with gestures
            # Slightly animated head movement
            for head_angle in [85, 75, 65, 75]:
                set_servo_angle(11, head_angle)
                wait_with_check(0.4)
                
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
                wait_with_check(0.7)
                
        else:
            wait_with_check(0.5)


def audio_worker():
    """Thread worker for audio playback"""
    while True:
        text = audio_queue.get()
        if text is None:
            break
        try:
            audio_playing.set()
            text_to_speech(text)
        except Exception as e:
            logger.error(f"Audio worker error: {e}")
        finally:
            audio_playing.clear()
            audio_queue.task_done()

def text_to_speech(text):
    """
    Convert text to speech using Piper, show buffering animation while audio is being generated, 
    and speaking animation while audio is played.
    """
    import subprocess
    import logging
    import os
    import tempfile
    import threading
    import time

    logger = logging.getLogger(__name__)

    piper_dir = "/piper"
    model_path = os.path.join(piper_dir, "en_US-amy-low.onnx")
    piper_exe = os.path.join(piper_dir, "piper")

    if not os.path.exists(model_path) or not os.path.exists(piper_exe):
        logger.error("[PiperTTS] Model or Piper binary not found.")
        print("[PiperTTS] Model or Piper binary not found.")
        return

    logger.info(f"[PiperTTS] Speaking: '{text}'")
    print(f"[PiperTTS] Speaking: '{text}'")

    output_file = None

    try:
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_file = tmp.name

        # Run Piper to generate speech (audio)
        piper_cmd = [
            piper_exe,
            "--model", model_path,
            "--output_file", output_file
        ]
        
        # Start Piper process
        process = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=piper_dir
        )

        # Show buffering animation until Piper finishes
        with emotion_lock:
            global current_emotion
            current_emotion = "buffering2"
        
        # Wait for Piper process to finish generating audio
        process.communicate(input=text)

        if process.returncode != 0 or not os.path.exists(output_file):
            logger.error("[PiperTTS] Piper failed or output file missing")
            return

        # Buffering animation loop until audio is ready for playback
        while process.returncode == 0 and not os.path.exists(output_file):
            time.sleep(0.1)

        # Thread target to play audio
        def play_audio():
            try:
                subprocess.run(["aplay", output_file], check=True)
            except Exception as e:
                logger.warning(f"[PiperTTS] Audio playback error: {e}")

        # Set emotion to speaking and start audio playback
        with emotion_lock:
            current_emotion = "speaking"

        playback_thread = threading.Thread(target=play_audio)
        playback_thread.start()

        # Show speaking animation while audio is playing
        while playback_thread.is_alive():
         
            time.sleep(0.1)

        # Reset emotion to neutral after playback finishes
        with emotion_lock:
            current_emotion = "neutral"

    except Exception as e:
        logger.error(f"[PiperTTS] Exception: {e}")
        print(f"[PiperTTS] Exception: {e}")
        with emotion_lock:
            current_emotion = "neutral"

    finally:
        if output_file and os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except Exception:
                pass
def speech_to_text(max_listen_time=100):  # Changed default to 100 seconds
    """Convert speech to text using Whisper"""
    try:
        # Set emotion to mic animation when starting to listen
        with emotion_lock:
            global current_emotion
            current_emotion = "mic"
        
        # Record audio to a temporary WAV file for Whisper
        samplerate = 16000
        channels = 1
        logger.info(f"Recording audio for Whisper STT with silence detection (max {max_listen_time} seconds)...")
        print(f"Recording for Whisper with silence detection (max {max_listen_time} seconds)...")
        
        silence_threshold = 500  # Adjust as needed
        silence_max_blocks = int(2.5 * samplerate / 1024)  # 2.5 seconds of silence
        blocksize = 1024
        max_blocks = int(max_listen_time * samplerate / blocksize)  # Now correctly uses max_listen_time
        audio_frames = []
        silence_counter = 0
        
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16', blocksize=blocksize) as stream:
                print("Listening for speech (Whisper)...")
                for _ in range(max_blocks):  # Explicitly use max_blocks in loop to ensure correct limit
                    block, _ = stream.read(blocksize)
                    audio_frames.append(block)
                    peak = np.abs(block).max()
                    
                    if peak < silence_threshold:
                        silence_counter += 1
                    else:
                        silence_counter = 0
                        
                    # If silence for >2.5s, stop
                    if silence_counter >= silence_max_blocks:
                        print("Detected >2.5s silence, stopping recording.")
                        break
                        
                    # Check time elapsed against max_listen_time parameter
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_listen_time:
                        print(f"Max listen time of {max_listen_time} seconds reached.")
                        break
                        
            # Change emotion to buffering while processing
            with emotion_lock:
                current_emotion = "buffering2"
                
            audio = np.concatenate(audio_frames, axis=0)
            sf.write(tmpfile.name, audio, samplerate)
            tmpfile.flush()
            
            # Process with Whisper in current thread (avoiding creating additional thread)
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel(model_size_or_path="tiny", device="cpu", compute_type="int8")
                
                # Configure Whisper to use English (US) language only
                segments, info = model.transcribe(
                    tmpfile.name, 
                    beam_size=1,
                    language="en",        # Force English language
                    task="transcribe",    # Transcription task
                    initial_prompt="This is English US speech."  # Help with context
                )
                
                text = " ".join([segment.text for segment in segments])
                # Remove trailing period for username detection
                username = text.strip()
                if username.endswith('.'):
                    username = username[:-1]
                logger.info(f"Whisper recognized: {username}")
                print(f"Whisper recognized: {username}")
                
                # Clean up temp file
                try:
                    os.unlink(tmpfile.name)
                except Exception:
                    pass
                    
                return username.strip()
                
            except Exception as e:
                logger.error(f"Whisper STT failed: {e}")
                print(f"Whisper STT failed: {e}")
                
                try:
                    os.unlink(tmpfile.name)
                except Exception:
                    pass
                    
                return ""
                
    except Exception as rec_e:
        logger.error(f"Audio recording failed: {rec_e}")
        print(f"Audio recording failed: {rec_e}")
        return ""
    finally:
        # Reset emotion if function exits due to error
        with emotion_lock:
            if current_emotion == "mic" or current_emotion == "buffering2":
                current_emotion = "neutral"
                
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

def establish_connection(retries=5, delay=2):
    """Establish connection to laptop"""
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
    """Perform handshake with laptop"""
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
        
def receive_response(sock, base_timeout=60):
    """Receive response with dynamic timeout adjustment"""
    global current_emotion
    with emotion_lock:
        current_emotion = "buffering2"  # Show buffering while waiting for response
        
    dynamic_timeout = base_timeout
    start_time = time.time()
    
    while time.time() - start_time < MAX_TIMEOUT:
        try:
            sock.settimeout(dynamic_timeout)
            
            raw_len = sock.recv(4)
            if not raw_len or len(raw_len) != 4:
                logger.warning("No or incomplete length header received.")
                print("No or incomplete length header received.")
                dynamic_timeout *= 1.3
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
                # Reset emotion after receiving response
                with emotion_lock:
                    current_emotion = "neutral"
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
    with emotion_lock:
        current_emotion = "neutral"
    return ""

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

class ProtocolError(Exception):
    """Custom exception for protocol errors"""
    pass

# Start touch monitor thread
touch_monitor_thread = threading.Thread(target=monitor_touch_interaction, daemon=True)
touch_monitor_thread.start()

# Add sleep monitor thread
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
    set_servo_angle(11, 75)  # Body motor at neutral position (75 is straight as per new config)

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
            response = receive_response(data_sock)
            return response

        # Protocol sequence
        while True:
            print("Waiting for wake word 'Hey Emo'...")
            with emotion_lock:
                current_emotion = "neutral"    # looking around
                
            # Subtle body movements while in idle/waiting state
            set_servo_angle(11, 65)  # Slight body tilt to one side (left)
            time.sleep(3)
            set_servo_angle(11, 85)  # Slight body tilt to other side (right)
            time.sleep(3)
            set_servo_angle(11, 75)  # Return to center (straight)

            if wake_word_detection():
                # Make an immediate audio response when wake word is detected
                # Turn body toward user when wake word detected
                set_servo_angle(11, 100)  # Turn body toward user (right)
                time.sleep(0.5)  # Slower movement
                set_servo_angle(11, 75)  # Return to neutral (straight)
                
                text_to_speech("Hello! I am your mental health assistant. What's your name?")
                with emotion_lock:
                    current_emotion = "blink2"   # Listening 
                
                # Get username with inline validation
                username = ""
                while not username:
                    # Attentive body posture while listening
                    set_servo_angle(11, 85)  # Slight lean right to show attention
                    username_text = speech_to_text(max_listen_time=12)
                    set_servo_angle(11, 75)  # Return to neutral (straight)
                    
                    username = extract_username(username_text)
                    if not username:
                        # Show slight disappointment with body language
                        set_servo_angle(11, 60)  # Lean left slightly
                        time.sleep(0.5)  # Slower movement
                        text_to_speech("Let's try again. Please say your name clearly.")
                        set_servo_angle(11, 75)  # Return to neutral (straight)
                        with emotion_lock:
                            current_emotion = "blink2"   # Listening 

                # Start new interaction cycle with the username
                response = execute_command({
                    "command": "username",
                    "username": username
                })
                
                # Handle response for username
                try:
                    response_data = json.loads(response)
                    if response_data.get("status") == "success":
                         # After username is processed, start the conversation
                        execute_command({"command": "start", "text": "Conversation started"})

                        with emotion_lock:
                            current_emotion = "happy"
                        # Wave hand while greeting and move body with excitement
                        set_servo_angle(11, 90)  # Body leans right in enthusiasm
                        set_servo_angle(0, 50)  # Lift left hand to wave (up from 120 which is down)
                        time.sleep(0.8)  # Slower movement
                        set_servo_angle(0, 120)   # Hand back to neutral (down)
                        set_servo_angle(11, 75)  # Body back to neutral (straight)
                        text_to_speech(f"Nice to meet you, {username}. How's your day going?")
                        with emotion_lock:
                            current_emotion = "mic"   # Listening
                    else:
                        # Body shows uncertainty
                        set_servo_angle(11, 60)  # Lean left as if unsure
                        text_to_speech(response_data.get("message", "Registration issue. Let's try again."))
                        set_servo_angle(11, 75)  # Return to neutral (straight)
                        continue
                except (json.JSONDecodeError, TypeError):
                    # Body shows confusion
                    set_servo_angle(11, 50)  # More pronounced lean left
                    time.sleep(0.5)  # Slower movement
                    set_servo_angle(11, 100)  # Then right
                    time.sleep(0.5)  # Slower movement
                    set_servo_angle(11, 75)  # Back to neutral (straight)
                    text_to_speech("System busy. Please try again.")
                    continue

                # Conversation loop with timeout control
                convo_timeout = time.time() + 300  # 5-minute session
                while time.time() < convo_timeout:
                    # Wait for any ongoing audio playback to finish before listening again
                    while audio_playing.is_set():
                        time.sleep(0.5)
                    
                    # Attentive body posture while listening
                    set_servo_angle(11, 85)  # Slight lean right to show attention    
                    user_text = speech_to_text(max_listen_time=100)
                    set_servo_angle(11, 75)  # Return to neutral (straight)
                    
                    if not user_text:
                        with emotion_lock:
                            current_emotion = "blink2"   # Listening
                        # Body shows confusion with a slight tilt
                        set_servo_angle(11, 65)  # Tilt left slightly
                        time.sleep(0.5)  # Slower movement
                        set_servo_angle(11, 85)  # Tilt right
                        time.sleep(0.5)  # Slower movement
                        set_servo_angle(11, 75)  # Return to center (straight)
                        text_to_speech("I didn't hear anything. Let's try again.")
                        continue
                    
                    if "bye" in user_text.lower():
                        with emotion_lock:
                            current_emotion = "happy"
                        # Wave goodbye with hand motion and body movement
                        # First bow slightly with body
                        set_servo_angle(11, 100)  # Lean right in a slight bow
                        time.sleep(0.8)  # Slower movement
                        set_servo_angle(11, 75)  # Return to neutral (straight)
                        
                        # Then wave with left hand
                        set_servo_angle(0, 50)  # Lift hand up (from 120 which is down)
                        time.sleep(0.5)  # Slower movement
                        # Wave back and forth with body movement
                        for i in range(2):
                            set_servo_angle(0, 80)
                            set_servo_angle(11, 85)  # Body leans right
                            time.sleep(0.4)  # Slower movement
                            set_servo_angle(0, 0)
                            set_servo_angle(11, 65)  # Body leans left
                            time.sleep(0.4)  # Slower movement
                        # Return to neutral
                        set_servo_angle(0, 120)  # Hand down
                        set_servo_angle(11, 75)  # Straight
                        text_to_speech(f"Goodbye, {username}! Have a great day.")
                        # Add buffer after conversation ends
                        with emotion_lock:
                            current_emotion = "buffering"                        
                        execute_command({
                            "command": "stop",
                            "text": "Conversation stopped"
                        })                        
                        
                        with emotion_lock:
                            current_emotion = "blink2"  
                        time.sleep(BUFFER_TIME_AFTER_CONVERSATION)# Return to blink mode and start sleep timer
                        with emotion_lock:
                            current_emotion = "sleep"     
                        break
                        
                    # Send and process with sequence tracking
                    if user_text and len(user_text.strip()) > 3:
                        # Body shows processing movement
                        set_servo_angle(11, 85)  # Slight tilt right as if thinking
                        
                        # Show buffering2 emotion while waiting for laptop response
                        with emotion_lock:
                            current_emotion = "buffering2"
                            
                        response = execute_command({"text": user_text})
                        if response:
                            with emotion_lock:
                                current_emotion = "blink2"   # Listening 
                            # Body shows active response
                            set_servo_angle(11, 95)  # Lean right to deliver response
                            time.sleep(0.5)  # Slower movement
                            set_servo_angle(11, 75)  # Return to neutral (straight)
                            # Queue response for smooth playback
                            audio_queue.put(response)
                            
                            # Subtle body movements while talking
                            for i in range(3):
                                set_servo_angle(11, 80)  # Small movements to simulate talking (right)
                                time.sleep(0.8)  # Slower movement
                                set_servo_angle(11, 70)  # Small movements to simulate talking (left)
                                time.sleep(0.8)  # Slower movement
                            set_servo_angle(11, 75)  # Return to neutral (straight)
                            
                            # Wait for audio playback to complete before continuing
                            print("Waiting for audio playback to complete...")
                            audio_queue.join()
                        else:
                            # Body shows confusion
                            set_servo_angle(11, 60)  # Lean left in confusion
                            time.sleep(0.5)  # Slower movement
                            set_servo_angle(11, 75)  # Return to neutral (straight)
                            text_to_speech("Sorry, I didn't get that. Could you repeat?")

    except ConnectionError as e:
        # Body shows distress
        set_servo_angle(11, 45)  # Major lean left showing problem
        logger.error(f"Connection error: {e}")
        text_to_speech("I lost connection to the server. Please try again later.")
        set_servo_angle(11, 75)  # Return to neutral (straight)
    except ProtocolError as e:
        # Body shows confusion
        set_servo_angle(11, 50)  # Lean left
        logger.error(f"Protocol error: {e}")
        text_to_speech("There was a communication error with the system.")
        set_servo_angle(11, 75)  # Return to neutral (straight)
    except Exception as e:
        # Body shows malfunction
        set_servo_angle(11, 40)  # Major lean left
        time.sleep(0.5)  # Slower movement
        set_servo_angle(11, 110)  # Then right
        time.sleep(0.5)  # Slower movement
        set_servo_angle(11, 75)  # Back to neutral (straight)
        logger.error(f"Main loop failure: {e}")
        text_to_speech("I encountered a problem and need to restart. Please try again.")
    finally:
        # Return body to neutral position before cleanup
        set_servo_angle(11, 75)  # Straight position
        if data_sock:
            data_sock.close()
        # Signal audio thread to terminate
        audio_queue.put(None)
        logger.info("Cleaned up resources.")
        print("Cleaned up resources.")
        
if __name__ == "__main__":
    import random  # Added import for random module
    main()
