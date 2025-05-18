import os
import json
import datetime
from collections import Counter
import re
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

# Configure MongoDB connection
MONGODB_URI = "mongodb://localhost:27017/mood-tracker"
DB_NAME = "mood-tracker"
COLLECTION_NAME = "moods"

# File paths
SUMMARY_FILE = "conversation_summary.json"

# Try to use transformers for summarization and keyword extraction, fallback to simple approach if not available
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Initialize T5 model
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, max_length=15)
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import transformers library: {e}")
    print("Using fallback summarization method")
    TRANSFORMERS_AVAILABLE = False

def extract_tags_from_text(text, top_n=5):
    """Extract tags from text using simple NLP (keywords)."""
    if not text or len(text) < 10:
        return []
    tags = []
    if TRANSFORMERS_AVAILABLE:
        # Use a simple regex to split words, filter stopwords, and count frequency
        words = re.findall(r'\b\w{3,}\b', text.lower())
        # Basic stopword list
        stopwords = set([
            'the', 'and', 'for', 'that', 'with', 'this', 'from', 'you', 'are', 'but', 'not', 'all', 'can', 'was', 'have', 'has', 'had', 'will', 'would', 'there', 'their', 'what', 'your', 'about', 'which', 'when', 'where', 'who', 'how', 'why', 'she', 'him', 'her', 'his', 'they', 'them', 'our', 'out', 'get', 'got', 'just', 'now', 'one', 'like', 'been', 'did', 'too', 'very', 'more', 'some', 'any', 'than', 'then', 'also', 'because', 'into', 'over', 'after', 'before', 'such', 'only', 'other', 'most', 'could', 'should', 'may', 'might', 'each', 'every', 'many', 'much', 'own', 'same', 'see', 'use', 'used', 'using', 'well', 'make', 'made', 'does', 'doing', 'done', 'said', 'say', 'says', 'still', 'even', 'let', 'good', 'bad', 'yes', 'no', 'if', 'on', 'in', 'at', 'by', 'an', 'as', 'to', 'of', 'is', 'it', 'be', 'or', 'a', 'i', 'he', 'we', 'do', 'so', 'up', 'down'
        ])
        filtered = [w for w in words if w not in stopwords]
        freq = Counter(filtered)
        tags = [w for w, _ in freq.most_common(top_n)]
    else:
        # Fallback: split and take most frequent non-stopword words
        words = text.lower().split()
        freq = Counter(words)
        tags = [w for w, _ in freq.most_common(top_n)]
    return tags

def calculate_mood_score(emotions):
    """Calculate mood score based on emotions"""
    positive = {"energized", "excited", "happy", "hopeful", "inspired", "proud", 
               "balanced", "calm", "satisfied", "grateful", "loved", "relieved"}
    neutral = {"unmotivated", "surprised", "confused", "overwhelmed", "bored", "tired"}
    negative = {"angry", "annoyed", "frustrated", "nervous", "stressed", "worried",
               "disappointed", "hopeless", "lonely", "sad", "weak", "guilty"}
    
    if not emotions:
        return 3  # Default neutral
    
    # Count emotion types
    pos_count = sum(1 for e in emotions if e in positive)
    neu_count = sum(1 for e in emotions if e in neutral)
    neg_count = sum(1 for e in emotions if e in negative)
    
    total = pos_count + neu_count + neg_count
    if total == 0:
        return 3
    
    # Calculate weighted score (1-5 scale)
    score = 3 + 2 * (pos_count - neg_count) / total
    
    # Clamp to range 1-5
    return max(1, min(5, round(score)))

def generate_concise_summary(text):
    """Generate a concise summary"""
    if not text or len(text) < 50:
        return text
        
    try:
        if TRANSFORMERS_AVAILABLE:
            # Use transformers if available
            prefix = "summarize: "
            input_text = prefix + text
            
            # Generate summary
            summary = summarizer(input_text)[0]['summary_text']
            
            # Ensure it's not too long (max 15 words)
            words = summary.split()
            if len(words) > 15:
                summary = ' '.join(words[:15]) + '...'
                
            return summary
        else:
            # Simple fallback method: return first 100 chars
            return text[:100] + "..." if len(text) > 100 else text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return text[:100] + "..." if len(text) > 100 else text

def extract_date_only(iso_date_string):
    """Extract just the date part from an ISO date string"""
    try:
        # Parse ISO date string and convert to YYYY-MM-DD format
        date_obj = datetime.datetime.fromisoformat(iso_date_string)
        return date_obj.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        # Return original if parsing fails
        return iso_date_string

def map_emotion_to_standard(emotion):
    """Map detected emotions to standard emotion categories used by the app"""
    # Map to standard emotion categories: happy, sad, angry, fear, surprised, disgusted, neutral
    emotion_map = {
        # Happy category
        "energized": "happy",
        "excited": "happy",
        "happy": "happy",
        "hopeful": "happy",
        "inspired": "happy",
        "proud": "happy",
        "balanced": "happy",
        "calm": "happy",
        "satisfied": "happy",
        "grateful": "happy",
        "loved": "happy",
        "relieved": "happy",
        
        # Sad category
        "disappointed": "sad",
        "hopeless": "sad",
        "lonely": "sad",
        "sad": "sad",
        "weak": "sad",
        "guilty": "sad",
        
        # Angry category
        "angry": "angry",
        "annoyed": "angry",
        "frustrated": "angry",
        
        # Fear category
        "nervous": "fear",
        "stressed": "fear",
        "worried": "fear",
        
        # Surprised category
        "surprised": "surprised",
        
        # Neutral category
        "unmotivated": "neutral",
        "confused": "neutral",
        "overwhelmed": "neutral",
        "bored": "neutral",
        "tired": "neutral"
    }
    
    return emotion_map.get(emotion.lower(), "neutral")

def process_conversation_summary():
    """Process conversation summary and update MongoDB"""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        moods_collection = db[COLLECTION_NAME]
        users_collection = db["users"]
        
        # Load conversation summary
        if not os.path.exists(SUMMARY_FILE):
            print(f"Summary file {SUMMARY_FILE} not found")
            return
            
        with open(SUMMARY_FILE, 'r') as f:
            summary_data = json.load(f)
        
        # Process each conversation session
        for session_id, session in summary_data.items():
            try:
                # Skip if no username
                if "username" not in session:
                    print(f"Session {session_id} has no username, skipping")
                    continue
                    
                username = session["username"]
                
                # First, find the user document to get the userId
                user = users_collection.find_one({"username": username})
                if not user:
                    print(f"User {username} not found in the database, skipping")
                    continue
                
                user_id = user["_id"]
                print(f"Found user {username} with ID {user_id}")
                
                # Extract conversation text for summarization
                conversation_text = ""
                for message in session.get("messages", []):
                    if message["role"] == "user":
                        conversation_text += message["content"] + " "
                
                # Generate concise summary
                summary_text = generate_concise_summary(conversation_text)
                
                # DEBUG: Print conversation text
                print(f"[DEBUG] Conversation text for session {session_id}: {conversation_text}")
                
                # Extract tags from conversation text
                tags = extract_tags_from_text(conversation_text, top_n=5)
                # DEBUG: Print extracted tags
                print(f"[DEBUG] Extracted tags for session {session_id}: {tags}")

                # Ensure tags is not empty
                if not tags:
                    tags = ["general"]

                # Calculate mood score from emotions
                emotions = session.get("text_emotions", [])
                # Ensure emotions is not empty
                if not emotions:
                    emotions = ["neutral"]
                
                # Calculate mood score from emotions
                emotions = session.get("text_emotions", [])
                mood_score = calculate_mood_score(emotions)
                
                # Map emotions to standard categories
                mapped_emotions = [map_emotion_to_standard(emotion) for emotion in emotions]
                mapped_emotions = list(set(mapped_emotions))  # Remove duplicates
                # Ensure mapped_emotions is not empty
                if not mapped_emotions:
                    mapped_emotions = ["neutral"]
                
                # Determine primary emotion (most frequent)
                emotion_counter = Counter(mapped_emotions)
                primary_emotion = emotion_counter.most_common(1)[0][0] if mapped_emotions else "neutral"
                
                # Get session date
                session_date = session.get("start", datetime.datetime.now().isoformat())
                date_obj = datetime.datetime.fromisoformat(session_date.replace('Z', '+00:00') 
                                                         if session_date.endswith('Z') 
                                                         else session_date)
                
                # Get current timestamp
                current_time = datetime.datetime.now()
                
                # Check if entry already exists for this username and date with proper date comparison
                start_of_day = datetime.datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0)
                end_of_day = datetime.datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 59)
                
                existing = moods_collection.find_one({
                    "userId": user_id,
                    "date": {
                        "$gte": start_of_day,
                        "$lte": end_of_day
                    }
                })
                
                if existing:
                    # Average the mood score
                    existing_score = existing.get("moodScore", 3)
                    averaged_score = (existing_score + mood_score) / 2
                    
                    # Get existing emotions and combine with new ones (avoiding duplicates)
                    existing_emotions = existing.get("emotions", [])
                    combined_emotions = list(set(existing_emotions + mapped_emotions))
                    
                    # Keep most recent primary emotion
                    final_primary_emotion = primary_emotion if mapped_emotions else existing.get("primaryEmotion", "neutral")
                    
                    # Get existing tags and ensure no duplicates
                    existing_tags = existing.get("tags", [])
                    # Combine with new tags, avoiding duplicates
                    combined_tags = list(set(existing_tags + tags))
                    # Ensure combined_tags is not empty
                    if not combined_tags:
                        combined_tags = ["general"]
                    
                    # Combine notes if there is existing content
                    if existing.get("notes"):
                        # Keep existing notes and add new ones
                        notes = f"{existing['notes']} | {summary_text}"
                    else:
                        notes = summary_text
                    
                    # Increment version number
                    version = existing.get("__v", 0) + 1
                    
                    # Update existing entry
                    # Ensure combined_emotions is not empty
                    if not combined_emotions:
                        combined_emotions = ["neutral"]
                    moods_collection.update_one(
                        {"_id": existing["_id"]},
                        {"$set": {
                            "moodScore": averaged_score,
                            "emotions": combined_emotions,
                            "primaryEmotion": final_primary_emotion,
                            "notes": notes,
                            "tags": combined_tags,
                            "updatedAt": current_time,
                            "__v": version
                        }}
                    )
                    print(f"Updated mood entry for {username} on {date_obj.date()} (version {version})")
                else:
                    # Create mood entry for new document
                    mood_entry = {
                        "userId": user_id,  # Use the actual ObjectId
                        "username": username, 
                        "date": date_obj,  # Store as proper date object
                        "moodScore": mood_score,
                        "emotions": mapped_emotions if mapped_emotions else ["neutral"],
                        "primaryEmotion": primary_emotion,
                        "notes": summary_text,
                        "tags": tags if tags else ["general"],  # Fill with extracted tags or default
                        "createdAt": current_time,
                        "updatedAt": current_time,
                        "__v": 0  # Initialize version number to 0
                    }
                    
                    # Insert new entry
                    result = moods_collection.insert_one(mood_entry)
                    print(f"Added new mood entry for {username} on {date_obj.date()} (id: {result.inserted_id})")
                
            except Exception as e:
                print(f"Error processing session {session_id}: {e}")
                continue
                
        print("Database update completed successfully")
        
    except Exception as e:
        print(f"Error connecting to database: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    process_conversation_summary()