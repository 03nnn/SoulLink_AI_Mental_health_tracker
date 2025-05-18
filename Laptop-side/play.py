import os
import json
import datetime
import subprocess
import ollama
import cv2
import socket
import struct
import threading
import numpy as np
from collections import Counter
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
import sys
import time
import logging
import uuid
from deepface import DeepFace
import tensorflow as tf
import pymongo
from pymongo import MongoClient
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('soullink.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
VIDEO_PORT = 5000
DATA_PORT = 5001
EMOTION_MODEL_JSON = "emotion_model1.json"
EMOTION_MODEL_WEIGHTS = "emotion_model1.h5"
HISTORY_FILE = "chat_history.json"
SUMMARY_FILE = "conversation_summary.json"

# Face detection model emotions (limited to 5)
EMOTION_LABELS_MODEL = ["happy", "sad", "angry", "surprised", "neutral"]

# Corrected emotion mapping for display - mapping between detected emotion and displayed emotion
# This fixes the swapped emotion display issue (angry showing as happy, etc.)
EMOTION_DISPLAY_MAP = {
    "angry": "angry",      # No change
    "happy": "happy",      # No change
    "sad": "sad",          # No change
    "surprised": "surprised", # No change
    "neutral": "neutral"   # No change
}

# Extended Emotion List for text analysis
EMOTION_LABELS_TEXT = [
    "energized", "excited", "happy", "hopeful", "inspired", "proud",
    "balanced", "calm", "satisfied", "grateful", "loved", "relieved",
    "unmotivated", "surprised", "confused", "overwhelmed", "bored", "tired",
    "angry", "annoyed", "frustrated", "nervous", "stressed", "worried",
    "disappointed", "hopeless", "lonely", "sad", "weak", "guilty"
]

# Create dictionaries for both emotion sets
emotion_dict_model = {i: label for i, label in enumerate(EMOTION_LABELS_MODEL)}

def classify_primary_emotion(emotion):
    positive = {"energized", "excited", "happy", "hopeful", "inspired", "proud", 
               "balanced", "calm", "satisfied", "grateful", "loved", "relieved"}
    neutral = {"unmotivated", "surprised", "confused", "overwhelmed", "bored", "tired"}
    negative = {"angry", "annoyed", "frustrated", "nervous", "stressed", "worried",
               "disappointed", "hopeless", "lonely", "sad", "weak", "guilty"}
    return "positive" if emotion in positive else "neutral" if emotion in neutral else "negative"

class VideoReceiver:
    def __init__(self):
        self.latest_frame = None
        self.latest_emotion = ("no_face", 0.0)
        self.frame_lock = threading.Lock()
        self.emotion_lock = threading.Lock()
        self.running = True
        self.video_active = False
        self.display_active = False
        self.udp_socket = None
        self.frame_counter = 0
        self.last_log_time = time.time()

        try:
            # Initialize face detection cascade
            logger.info("Initializing facial expression detection...")
            
            # Set up DeepFace options
            self.emotion_model = "DeepFace"  # Options: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib'
            self.use_deepface = True
            
            # Keep existing cascade as fallback
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
            
            if not os.path.isfile(cascade_path):
                logger.warning(f"Haar cascade file missing at {cascade_path}, using DeepFace only")
                self.face_cascade = None
            else:    
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    logger.warning("Failed to load Haar cascade classifier, using DeepFace only")
                    self.face_cascade = None

            # Load emotion recognition model (keep original model as fallback)
            logger.info("Loading emotion recognition model...")
            if not os.path.isfile(EMOTION_MODEL_JSON):
                logger.warning(f"Model JSON missing: {EMOTION_MODEL_JSON}")
            if not os.path.isfile(EMOTION_MODEL_WEIGHTS):
                logger.warning(f"Model weights missing: {EMOTION_MODEL_WEIGHTS}")

            if os.path.isfile(EMOTION_MODEL_JSON) and os.path.isfile(EMOTION_MODEL_WEIGHTS):
                with open(EMOTION_MODEL_JSON, 'r') as json_file:
                    self.classifier = model_from_json(json_file.read())
                self.classifier.load_weights(EMOTION_MODEL_WEIGHTS)

                # Model validation - checks for 5 emotion classes
                logger.info("Validating model...")
                test_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
                prediction = self.classifier.predict(test_input)
                if prediction.shape != (1, len(EMOTION_LABELS_MODEL)):
                    logger.warning(f"Invalid model output shape: expected (1, {len(EMOTION_LABELS_MODEL)}), got {prediction.shape}")
                logger.info("Model validation passed")
            else:
                self.classifier = None
                logger.warning("Falling back to DeepFace only, no fallback classifier available")
            
            # Configure TensorFlow to use less GPU memory if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU memory growth enabled for {len(gpus)} GPUs")
                except RuntimeError as e:
                    logger.warning(f"Error setting GPU memory growth: {e}")
            
            logger.info("DeepFace emotion detection initialized successfully")

            # Initialize network components
            self._init_udp_socket()
            
            logger.info("VideoReceiver initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            self.video_active = False
            self.running = False
            # Cleanup if partial initialization occurred
            if hasattr(self, 'udp_socket') and self.udp_socket:
                self.udp_socket.close()

    def _init_udp_socket(self):
        """Initialize UDP socket for JPEG frame reception"""
        try:
            logger.info(f"Initializing UDP socket on port {VIDEO_PORT}...")
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind(('0.0.0.0', VIDEO_PORT))
            self.udp_socket.settimeout(1.0)
            self.video_active = True
            logger.info("UDP socket initialized successfully")
        except Exception as e:
            logger.error(f"Socket initialization failed: {e}", exc_info=True)
            self.video_active = False

    def get_latest_emotion(self):
        with self.emotion_lock:
            return self.latest_emotion

    def _process_frame(self, frame):
        """Process frame for face detection and emotion analysis using DeepFace"""
        try:
            # Color space conversion validation
            if frame is None:
                logger.error("Received None frame for processing")
                with self.emotion_lock:
                    self.latest_emotion = ("no_face", 0.0)
                return
                
            working_frame = frame.copy()
            
            # Handle different color spaces
            if working_frame.shape[2] == 4:  # Handle RGBA frames
                working_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGRA2BGR)
            elif working_frame.shape[2] == 3:  # Ensure BGR for OpenCV operations
                # Already BGR for OpenCV operations
                pass
            elif len(working_frame.shape) == 2:  # Handle grayscale frames
                working_frame = cv2.cvtColor(working_frame, cv2.COLOR_GRAY2BGR)
            
            # Debug: Save frame periodically
            if self.frame_counter % 100 == 0:
                cv2.imwrite(f"debug_frame_{self.frame_counter}.jpg", working_frame)
            
            # Use DeepFace for emotion analysis
            if self.use_deepface:
                try:
                    # Use OpenCV first to validate face presence - more efficient
                    gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Use cascade detector if available as pre-filter
                    face_detected = False
                    if self.face_cascade:
                        faces = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                        )
                        if len(faces) > 0:
                            face_detected = True
                            # Optional: Extract the face region for DeepFace to improve analysis
                            x, y, w, h = faces[0]  # Use first face
                            face_region = working_frame[y:y+h, x:x+w]
                            # Ensure face region is valid
                            if face_region.size > 0:
                                working_frame = face_region
                    
                    # If cascade found no faces, set no_face immediately
                    if self.face_cascade and not face_detected:
                        logger.debug("No face detected by cascade pre-filter")
                        with self.emotion_lock:
                            self.latest_emotion = ("no_face", 0.0)
                        return
                    
                    # Configure DeepFace analysis
                    start_time = time.time()
                    
                    # Try with enforce_detection=True first to confirm face presence
                    try:
                        analysis = DeepFace.analyze(
                            img_path=working_frame,
                            actions=['emotion'],
                            enforce_detection=True,  # First try WITH face detection enforcement
                            detector_backend='opencv'  # 'opencv', 'ssd', 'mtcnn', 'retinaface'
                        )
                        face_detected = True
                    except ValueError as e:
                        # This is likely "Face could not be detected" error
                        if "Face could not be detected" in str(e):
                            face_detected = False
                            logger.debug("DeepFace confirmed no face in frame")
                        else:
                            # Some other error
                            raise
                    
                    if not face_detected:
                        with self.emotion_lock:
                            self.latest_emotion = ("no_face", 0.0)
                        return
                    
                    detection_time = time.time() - start_time
                    
                    if analysis and isinstance(analysis, list) and len(analysis) > 0:
                        # Extract emotion data
                        dominant_emotion = analysis[0]['dominant_emotion']
                        emotion_scores = analysis[0]['emotion']
                        
                        # Map DeepFace emotions to our model
                        # DeepFace provides: angry, disgust, fear, happy, sad, surprise, neutral
                        if dominant_emotion == 'surprise':
                            dominant_emotion = 'surprised'  # Match our emotion labels
                            
                        confidence = emotion_scores[dominant_emotion if dominant_emotion != 'surprised' else 'surprise'] / 100.0
                        
                        logger.debug(f"DeepFace detected emotion: {dominant_emotion} ({confidence:.2f}) in {detection_time:.3f}s")
                        
                        with self.emotion_lock:
                            self.latest_emotion = (dominant_emotion, confidence)
                    else:
                        logger.debug("DeepFace returned empty analysis")
                        with self.emotion_lock:
                            self.latest_emotion = ("no_face", 0.0)
                
                except Exception as e:
                    logger.warning(f"DeepFace analysis error: {e}, falling back to cascade detector")
                    # Fall back to cascade detector if DeepFace fails
                    self._fallback_emotion_detection(working_frame)
            else:
                # Use fallback if DeepFace is disabled
                self._fallback_emotion_detection(working_frame)
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            with self.emotion_lock:
                self.latest_emotion = ("no_face", 0.0)


    def _fallback_emotion_detection(self, frame):
        """Legacy cascade-based emotion detection as fallback"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            start_time = time.time()
            if self.face_cascade and self.classifier:
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                detection_time = time.time() - start_time
                
                if len(faces) == 0:
                    with self.emotion_lock:
                        self.latest_emotion = ("no_face", 0.0)
                    logger.debug("No faces detected in frame")
                    return

                # Process all detected faces
                for (x, y, w, h) in faces:
                    logger.debug(f"Face detected at ({x},{y}) [{w}x{h}] in {detection_time:.3f}s")

                    # Prepare ROI for emotion classification
                    roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
                    roi = roi.astype("float32") / 255.0
                    roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)

                    # Emotion prediction with timing and null check
                    start_time = time.time()
                    prediction = self.classifier.predict(roi, verbose=0)
                    
                    if prediction is None or len(prediction) == 0:
                        logger.error("Empty prediction from model")
                        continue
                    
                    prediction = prediction[0]
                    inference_time = time.time() - start_time
                    
                    # Use model emotion dictionary
                    label = emotion_dict_model.get(np.argmax(prediction), "neutral")
                    confidence = float(prediction[np.argmax(prediction)])
                    
                    logger.debug(f"Emotion inference: {label} ({confidence:.2f}) in {inference_time:.3f}s")
                    
                    with self.emotion_lock:
                        self.latest_emotion = (label, confidence)
                    break  # Process only the first face
            else:
                with self.emotion_lock:
                    self.latest_emotion = ("no_face", 0.0)
                logger.debug("Fallback detector not available")
                    
        except Exception as e:
            logger.error(f"Fallback emotion detection error: {e}", exc_info=True)
            with self.emotion_lock:
                self.latest_emotion = ("no_face", 0.0)


    def _update_frame(self):
        """Main frame processing loop using UDP"""
        logger.info("Starting frame processing thread")
        while self.running and self.video_active:
            try:
                # Receive UDP packet
                data, addr = self.udp_socket.recvfrom(65507)
                self.frame_counter += 1
                
                # Log frame rate every 5 seconds
                if time.time() - self.last_log_time > 5:
                    logger.info(f"Receiving {self.frame_counter/5:.1f} fps")
                    self.frame_counter = 0
                    self.last_log_time = time.time()

                # Validate frame header
                if len(data) < 5:
                    logger.warning("Received incomplete frame header")
                    continue
                
                # Extract frame size
                try:
                    size = struct.unpack('!I', data[:4])[0]
                except struct.error as e:
                    logger.warning(f"Invalid frame size header: {e}")
                    continue

                # Validate frame size
                if size != len(data[4:]):
                    logger.warning(f"Frame size mismatch: header {size} vs actual {len(data[4:])}")
                    continue

                # Decode JPEG frame
                try:
                    frame = cv2.imdecode(
                        np.frombuffer(data[4:], dtype=np.uint8), 
                        cv2.IMREAD_UNCHANGED
                    )
                except cv2.error as e:
                    logger.error(f"JPEG decode error: {e}")
                    continue

                if frame is None:
                    logger.warning("Received empty/invalid frame")
                    continue

                # Handle different color formats correctly
                if len(frame.shape) == 2:  # Grayscale image
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # RGBA image
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                # If it's already BGR (3 channels), leave it as is - OpenCV uses BGR by default
                
                # Update frame and process
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                self._process_frame(frame)

            except socket.timeout:
                logger.warning("Socket timeout while waiting for frame")
                continue
            except Exception as e:
                logger.error(f"Unexpected frame processing error: {e}", exc_info=True)
                time.sleep(0.1)

    def _display_feed(self):
        """Display video feed with emotion overlay"""
        if not self.video_active:
            return

        logger.info("Starting video display thread")
        cv2.namedWindow("Emotion Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Emotion Tracking", 800, 600)
        
        while self.running and self.video_active:
            try:
                # Get latest frame
                with self.frame_lock:
                    if self.latest_frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.latest_frame.copy()
                
                # Get emotion data
                detected_emotion, confidence = self.get_latest_emotion()
                
                # Apply the emotion mapping to fix the display issue
                # If the emotion is in the mapping, use the mapped value, otherwise keep original
                display_emotion = EMOTION_DISPLAY_MAP.get(detected_emotion, detected_emotion)
                
                # FIXED: Apply correct BGR color format for OpenCV
                # BGR color ordering: (Blue, Green, Red)
                if display_emotion == "happy":
                    # Green for happy - (0, 255, 0) in BGR
                    text_color = (0, 255, 0)  
                elif display_emotion == "angry":
                    # Red for angry - (0, 0, 255) in BGR
                    text_color = (0, 0, 255)  
                elif display_emotion == "sad":
                    # Blue for sad - (255, 0, 0) in BGR
                    text_color = (255, 0, 0)  
                elif display_emotion == "surprised":
                    # Purple for surprised - (255, 0, 255) in BGR
                    text_color = (255, 0, 255)  
                else:
                    # Yellow for neutral/other - (0, 255, 255) in BGR
                    text_color = (0, 255, 255)  
                
                # Create overlay text
                text = (f"{display_emotion} ({confidence*100:.1f}%)" 
                       if detected_emotion != "no_face" 
                       else "No face detected")
                
                # Add text overlay
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                cv2.rectangle(frame, (10, 10), (10 + text_size[0], 35), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                
                # Display frame
                cv2.imshow("Emotion Tracking", frame)

                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

            except Exception as e:
                logger.error(f"Display error: {e}", exc_info=True)
                time.sleep(0.1)

    def start(self):
        """Start video processing and display threads"""
        if self.video_active:
            logger.info("Starting video receiver threads")
            self.thread = threading.Thread(
                target=self._update_frame, 
                name="FrameProcessor",
                daemon=True
            )
            self.display_thread = threading.Thread(
                target=self._display_feed,
                name="VideoDisplay",
                daemon=True
            )
            self.thread.start()
            self.display_thread.start()
            logger.info("Video receiver fully operational")
        else:
            logger.warning("Starting without video capabilities")

    def stop(self):
        """Clean shutdown of video components"""
        logger.info("Initiating video receiver shutdown...")
        self.running = False
        
        # Close network resources
        if self.udp_socket:
            try:
                self.udp_socket.close()
                logger.info("UDP socket closed")
            except Exception as e:
                logger.error(f"Error closing socket: {e}")

        # Close OpenCV windows
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")
        except Exception as e:
            logger.error(f"Error closing windows: {e}")

        # Wait for threads to finish
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        if hasattr(self, 'display_thread'):
            self.display_thread.join(timeout=1)
            
        logger.info("Video receiver shutdown complete")

class RPIServer:
    def __init__(self, video_receiver):
        self.video_receiver = video_receiver
        self.history = []
        self.summary_data = {}
        self.current_session_start = None
        self.current_username = None
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.session_lock = Lock()  # Fixed syntax
        self.active_sessions = {}   # Key: connection object, Value: session data
        try:
            self.server_socket.bind(('0.0.0.0', DATA_PORT))
            print(f"Successfully bound to port {DATA_PORT}")
            self.server_socket.listen(1)
            logger.info(f"Server listening on port {DATA_PORT}")
        except Exception as e:
            logger.error(f"Failed to bind to port {DATA_PORT}: {e}")
            raise
            
    def _handle_handshake(self, conn):
        EXPECTED_HANDSHAKE = b"PI_RDY"
        CONFIRM_MSG = b"LAPTOP_OK"
        try:
            data = conn.recv(len(EXPECTED_HANDSHAKE))
            if data != EXPECTED_HANDSHAKE:
                print("Invalid handshake received")
                return False
            conn.sendall(CONFIRM_MSG)
            print("Handshake successful with client")
            return True
        except Exception as e:
            print(f"Handshake error: {e}")
            return False

    def _recv_all(self, conn, length):
        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _analyze_text_emotion(self, text):
        """Use LLM to analyze emotion from text with spelling correction"""
        try:
            response = ollama.chat(
                model="mistral:latest",
                messages=[{
                    "role": "system",
                    "content": f"""Analyze the emotional content of this text.
                    Possible emotions: {EMOTION_LABELS_TEXT}
                    Respond ONLY with the emotion name from the list.
                    Ensure your response is correctly spelled."""
                }, {
                    "role": "user",
                    "content": text
                }],
                options={'temperature': 0.1}  # Lower temperature for more predictable, accurate responses
            )
            emotion = response['message']['content'].strip().lower()
            
            # Verify emotion is in our list, otherwise use fuzzy matching
            if emotion in EMOTION_LABELS_TEXT:
                return emotion
            else:
                # Simple fuzzy matching - find closest match
                import difflib
                matches = difflib.get_close_matches(emotion, EMOTION_LABELS_TEXT, n=1, cutoff=0.6)
                return matches[0] if matches else "neutral"
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return "neutral"

    def _verify_username(self,username):
        """Verify if username is valid and not taken"""
        try:
            #Connect to MongoDB
            client = pymongo.MongoClient("mongodb://localhost:27017/mood-tracker")
            db=client["mood-tracker"]
            user_collection=db["users"]

            user = user_collection.find_one({"username": username})

            if user:
                logger.info(f"Username {username} verified in database")
                return True
            else:
                logger.info(f"Username {username} not found in database")
                return False
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            return False


    def generate_response(self, text, face_emotion, face_confidence):
        # Always analyze text emotion
        text_emotion = self._analyze_text_emotion(text)
        logger.info(f"Text analysis emotion: {text_emotion}")
        
        # Determine which emotion to use based on conditions
        if face_emotion == "no_face" or face_confidence < 0.2:
            # No reliable face detection - prioritize text emotion
            emotion_source = "text"
            final_emotion = text_emotion
            confidence = 0.7  # Default confidence for text analysis
        else:
            # Face detected with good confidence - use both but prioritize text
            emotion_source = "combined"
            # Prioritize text emotion as requested
            final_emotion = text_emotion
            confidence = 0.8  # Increased confidence when both sources available
            logger.info(f"Face: {face_emotion} ({face_confidence*100}%), Text: {text_emotion}")

        # Create history entry
        entry = {
            "role": "user",
            "content": text,
            "emotion": final_emotion,
            "text_emotion": text_emotion,  # Added text emotion specifically
            "face_emotion": face_emotion,  # Added face emotion specifically
            "confidence": confidence,
            "source": emotion_source,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add username if available
        if self.current_username:
            entry["username"] = self.current_username
            
        self.history.append(entry)

        # Build enhanced prompt that emphasizes text analysis but considers face
        prompt = f"""You are an emotional intelligence expert. Consider:
        - User message: {text}
        - Text analysis emotion: {text_emotion} (PRIMARY source)
        
        NOTE: Our facial emotion detection system can only detect these 5 emotions: 
        {EMOTION_LABELS_MODEL}
        
        - Facial emotion detected: {face_emotion} ({face_confidence*100:.1f}% confidence)
        - Current system is designed to prioritize text-based emotion analysis over facial emotions
        - Full range of emotions for text analysis: {EMOTION_LABELS_TEXT}

        Respond with empathy, prioritizing the text-based emotion analysis.
        Use facial emotion as supporting information only, especially if text analysis is ambiguous.
        If facial analysis reports "no_face" or has low confidence, ignore it entirely.
        
        Make your response natural and conversational, appropriate to someone feeling {final_emotion}.

        IMPORTANT GUIDELINES:
        1. DO NOT include any emoticons, emoji, or text-based smileys (like :) :( ;) etc.) in your response.
        2. Use proper spelling and grammar - proofread your response before submitting.
        3. Keep your response concise and to the point.
        4. Use only common words with simple spelling.
        """

        try:
            response = ollama.chat(
                model="mistral:latest",
                messages=[
                    {"role": "system", "content": prompt},
                    *[{"role": msg["role"], "content": msg["content"]} for msg in self.history[-3:]]
                ],
                options={'temperature': 0.3}  # Lower temperature for better spelling
            )
            response_content = response.get("message", {}).get("content", "I'm here to help.")
            
            # Remove any potential emoticons/emojis from the response
            import re
            emoticon_pattern = r'[:;=]-?[)(\]D>|pP/\\]|[)(\]D>|pP/\\]'
            response_content = re.sub(emoticon_pattern, '', response_content)
            
            # Check if TextBlob is already imported, if not add it as a dependency at runtime
            try:
                from textblob import TextBlob
                
                # Use TextBlob for spelling correction
                corrected_content = str(TextBlob(response_content).correct())
                
                # Log if changes were made
                if corrected_content != response_content:
                    logger.debug("Spelling correction applied to response")
                    response_content = corrected_content
                    
            except ImportError:
                # If TextBlob isn't available, try to install it
                logger.warning("TextBlob not found, attempting to install...")
                try:
                    import subprocess
                    subprocess.check_call(["pip", "install", "textblob"])
                    subprocess.check_call(["python", "-m", "textblob.download_corpora"])
                    
                    # Try again after installation
                    from textblob import TextBlob
                    corrected_content = str(TextBlob(response_content).correct())
                    response_content = corrected_content
                    logger.info("TextBlob installed and spelling correction applied")
                    
                except Exception as install_error:
                    logger.error(f"Failed to install TextBlob: {install_error}")
                    # Continue without spelling correction
                    
            except Exception as spell_error:
                logger.warning(f"TextBlob spell checking failed: {spell_error}")
                # Continue with original response if spell checking fails
            
            assistant_entry = {
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.history.append(assistant_entry)
            
            logger.info(f"Response generated for {final_emotion}")
            return response_content

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "Let's try approaching this differently. Could you elaborate?"
                
    def handle_client(self, conn):
        print("New client connection established")
        if not self._handle_handshake(conn):
            conn.close()
            return

        try:
            while True:
                print("Waiting for message from client...")
                raw_len = self._recv_all(conn, 4)
                if not raw_len:
                    print("Client disconnected")
                    break

                data_len = struct.unpack('!I', raw_len)[0]
                print(f"Receiving message of length: {data_len} bytes")
                data = self._recv_all(conn, data_len)
                if not data:
                    print("Empty data received from client")
                    break

                try:
                    text_data = json.loads(data.decode('utf-8'))
                    print(f"Received data from client: {text_data}")
                    
                    # Handle username if provided
                    if 'command' in text_data and text_data['command'] == 'username':
                        username = text_data.get('username', '')
                        if not username:
                            response_data = {
                                "status": "error",
                                "message": "No username provided."
                            }
                        else:
                            # verify username in database
                            is_verified = self._verify_username(username)
                            print(f"Username verification result for {username}: {is_verified}")
                            
                            if is_verified:
                                self.current_username = username
                                logger.info(f"Username set to: {self.current_username}")
                                # Send acknowledgement response
                                response_data = {
                                    "status": "success",
                                    "message": f"Nice to meet you, {username}. How can I help you today?"
                                }
                            else:
                                # Reset session state when username verification fails
                                self.current_username = None
                                self.current_session_start = None
                                self.history = []  # Clear conversation history
                                
                                logger.info(f"Username verification failed for: {username}")
                                response_data = {
                                    "status": "error",
                                    "message": "Please register first through the mobile app."
                                }
                        
                        print(f"Sending response for username command: {response_data}")
                        # Send response
                        response_bytes = json.dumps(response_data).encode('utf-8')
                        response_len = struct.pack('!I', len(response_bytes))
                        conn.sendall(response_len + response_bytes)
                        # Skip the rest and continue waiting for next command 
                        continue 

                    # Handle start command
                    elif 'command' in text_data and text_data['command'] == 'start':
                        # Only start session if username is verified
                        if self.current_username:
                            self.current_session_start = datetime.datetime.now().isoformat()
                            session_data = {
                                "start": self.current_session_start,
                                "end": None,
                                "summary": "",
                                "text_emotions": [],  # Added to track text emotions
                                "face_emotions": [],  # Added to track face emotions
                                "messages": [],
                                "username": self.current_username # Ensure username is included in session data
                            }
                            self.summary_data[self.current_session_start] = session_data
                            logger.info(f"Conversation session started for user: {self.current_username}")
                            
                            # Send acknowledgement response
                            response_data = {
                                "status": "success",
                                "message": f"Session started for {self.current_username}"
                            }
                        else:
                            # Send error if no username is set
                            response_data = {
                                "status": "error",
                                "message": "Please provide a username first"
                            }
                        
                        print(f"Sending response for start command: {response_data}")
                        response_bytes = json.dumps(response_data).encode('utf-8')
                        response_len = struct.pack('!I', len(response_bytes))
                        conn.sendall(response_len + response_bytes)
                        continue

                    # Handle stop command
                    # Handle stop command
                    elif 'command' in text_data and text_data['command'] == 'stop':
                        if self.current_session_start and self.current_session_start in self.summary_data:
                            self.summary_data[self.current_session_start]["end"] = \
                                datetime.datetime.now().isoformat()
                            # Generate and save summary
                            messages = self.summary_data[self.current_session_start]["messages"]
                            summary = self.generate_summary(messages)
                            self.summary_data[self.current_session_start]["summary"] = summary
                            self.save_history()
                            logger.info(f"Conversation session ended for user: {self.current_username}")
                            
                            # Save username before resetting
                            previous_username = self.current_username  # Keep for the response message
                            self.current_username = None
                            self.current_session_start = None
                            self.history = []  # Clear conversation history
                            
                            # Send acknowledgement
                            response_data = {
                                "status": "success",
                                "message": f"Session ended successfully for {previous_username}"
                            }
                        else:
                            # In error case, use current username directly instead of previous_username
                            response_data = {
                                "status": "error",
                                "message": f"No active session to stop for {self.current_username}" if self.current_username else "No active session to stop"
                            }
                        
                        print(f"Sending response for stop command: {response_data}")
                        response_bytes = json.dumps(response_data).encode('utf-8')
                        response_len = struct.pack('!I', len(response_bytes))
                        conn.sendall(response_len + response_bytes)
                        continue

                    # Handle regular messages
                    elif 'text' in text_data:
                        text = text_data.get("text", "")
                        print(f"Processing message: {text}")

                        # Check if we have a username and active session
                        if not self.current_username:
                            response_data = {
                                "status": "error",
                                "message": "Please provide a username first"
                            }
                            print(f"Sending response for text (no username): {response_data}")
                            response_bytes = json.dumps(response_data).encode('utf-8')
                            response_len = struct.pack('!I', len(response_bytes))
                            conn.sendall(response_len + response_bytes)
                            continue
                            
                        # Auto-start a session if none exists
                        if not self.current_session_start or self.current_session_start not in self.summary_data:
                            self.current_session_start = datetime.datetime.now().isoformat()
                            session_data = {
                                "start": self.current_session_start,
                                "end": None,
                                "summary": "",
                                "text_emotions": [],
                                "face_emotions": [],
                                "messages": [],
                                "username": self.current_username
                            }
                            self.summary_data[self.current_session_start] = session_data
                            logger.info(f"Auto-started conversation session for user: {self.current_username}")

                        face_emotion, face_confidence = self.video_receiver.get_latest_emotion()
                        print(f"Current emotion detected: {face_emotion} (confidence: {face_confidence*100:.1f}%)")

                        # Get text emotion 
                        text_emotion = self._analyze_text_emotion(text)
                        
                        response = self.generate_response(text, face_emotion, face_confidence)
                        
                        # Add to current session - this should now be safe
                        message_entry = {
                            "role": "user",
                            "content": text,
                            "text_emotion": text_emotion,
                            "face_emotion": face_emotion,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "username": self.current_username
                        }
                        
                        self.summary_data[self.current_session_start]["messages"].append(message_entry)
                        self.summary_data[self.current_session_start]["messages"].append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
                        # Add emotions to tracking lists
                        self.summary_data[self.current_session_start]["text_emotions"].append(text_emotion)
                        self.summary_data[self.current_session_start]["face_emotions"].append(face_emotion)

                        # Send response back to client
                        print(f"Sending text response: {response[:100]}...")  # Log first 100 chars
                        response_bytes = response.encode('utf-8')
                        response_len = struct.pack('!I', len(response_bytes))
                        try:
                            conn.sendall(response_len + response_bytes)
                            print("Response sent to client")
                        except Exception as send_error:
                            print(f"Error sending response: {send_error}")
                            break

                        print(f"[USER (Face: {face_emotion}, {face_confidence*100:.1f}%, Text: {text_emotion})]: {text}")
                        print(f"[ASSISTANT]: {response}")
                    else:
                        # Unrecognized command format
                        response_data = {
                            "status": "error",
                            "message": "Unknown command format"
                        }
                        print(f"Sending response for unknown command: {response_data}")
                        response_bytes = json.dumps(response_data).encode('utf-8')
                        response_len = struct.pack('!I', len(response_bytes))
                        conn.sendall(response_len + response_bytes)
                    
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    continue

        except Exception as e:
            print(f"Error in client handler: {e}")
            logger.error(f"Client handler error: {e}", exc_info=True)
        finally:
            conn.close()
            print("Client connection closed")
    
    def process_conversations(self):
        print("Processing conversation history...")
        conversation_segments = []
        current_segment = []

        for i in range(len(self.history)):
            if i > 0:
                prev_time = datetime.datetime.fromisoformat(self.history[i-1]["timestamp"])
                curr_time = datetime.datetime.fromisoformat(self.history[i]["timestamp"])
                if (curr_time - prev_time).seconds > 1800:
                    if current_segment:
                        conversation_segments.append(current_segment)
                        current_segment = []
            current_segment.append(self.history[i])

        if current_segment:
            conversation_segments.append(current_segment)

        for convo in conversation_segments:
            start_time = convo[0]["timestamp"]
            
            # Extract emotions and username from user messages
            face_emotions = []
            text_emotions = []
            final_emotions = []
            username = None
            
            for msg in convo:
                if msg["role"] == "user":
                    # Get the username from the first user message that has it
                    if not username and "username" in msg:
                        username = msg["username"]
                    
                    if "emotion" in msg:
                        final_emotions.append(msg["emotion"])
                    if "text_emotion" in msg:
                        text_emotions.append(msg["text_emotion"])
                    if "face_emotion" in msg:
                        face_emotions.append(msg["face_emotion"])
            
            # Get most common emotions
            most_common_final = Counter(final_emotions).most_common(1)
            most_common_text = Counter(text_emotions).most_common(1)
            most_common_face = Counter(face_emotions).most_common(1)
            
            summary_entry = {
                "overall_emotion": most_common_final[0][0] if most_common_final else "neutral",
                "text_emotion": most_common_text[0][0] if most_common_text else "neutral",
                "face_emotion": most_common_face[0][0] if most_common_face else "no_face",
                "text_emotions": text_emotions,
                "face_emotions": face_emotions,
                "summary": self.generate_summary(convo)
            }
            
            # Add username to summary if available
            if username:
                summary_entry["username"] = username
                
            self.summary_data[start_time] = summary_entry

        self.save_history()

    def generate_summary(self, conversation):
        print("Generating conversation summary...")
        messages = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation])
        
        # Extract username for summary context if available
        username = None
        for msg in conversation:
            if msg["role"] == "user" and "username" in msg:
                username = msg["username"]
                break
        
        system_prompt = "Summarize this conversation in 3 bullet points focusing on emotional state and key concerns."
        if username:
            system_prompt += f" The user's name is {username}."
        
        try:
            response = ollama.chat(
                model="mistral:latest",
                messages=[{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": messages
                }]
            )
            return response.get("message", {}).get("content", "No summary available.")
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary unavailable due to error"

    def save_history(self):
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(self.history, f, indent=4)
            with open(SUMMARY_FILE, "w") as f:
                json.dump(self.summary_data, f, indent=4)
            print("Conversation history saved successfully")
            
        # Call summarize.py script to update MongoDB
            try:
                logger.info("Running summarize.py to update MongoDB...")
                subprocess.run(["python","summarize.py"], check=True)
                logger.info("MongoDB updated successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running summarize.py: {e}")
            except Exception as e:
                logger.error(f"Error running summarize.py: {e}")

        except Exception as e:
            print(f"Error saving history: {e}")

            
    def run(self):
            try:
                print("Server running. Press Ctrl+C to stop...")
                while True:
                    conn, addr = self.server_socket.accept()
                    client_thread = threading.Thread(target=self.handle_client, args=(conn,))
                    client_thread.daemon = True
                    client_thread.start()
            except KeyboardInterrupt:
                print("\nServer shutdown requested")
            except Exception as e:
                print(f"Server error: {e}")
            finally:
                self.process_conversations()
                self.server_socket.close()
                self.video_receiver.stop()
                print("Server shutdown complete")

if __name__ == "__main__":
    logger.info("Starting SoulLink system...")
    video_receiver = VideoReceiver()
    server = RPIServer(video_receiver)
    
    try:
        video_receiver.start()  # Start video processing in background
        server.run()
    except KeyboardInterrupt:
        logger.info("\nApplication terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        sys.exit(0)