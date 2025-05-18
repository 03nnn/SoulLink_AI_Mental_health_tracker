// This file contains mood prediction data that will be synced to MongoDB
// Add new mood predictions to this array and run updateDb.js to sync them

module.exports = [
  {
    "username": "testuser",
    "date": "2025-03-31T16:42:59.230569",
    "moodScore": 3,
    "emotions": [
      "sad", "disheartened"
    ],
    "primaryEmotion": "sad",
    "notes": "Feeling disappointed, as the promised event was cancelled without notice.",
    "tags": ["event cancelled"]
  },
  {
    "username": "testuser",
    "date": "2025-03-31T16:43:06.073800",
    "moodScore": 3,
    "emotions": [
      "sad", "dispressed"
    ],
    "primaryEmotion": "sad",
    "notes": "You appeared uneasy due to feelings of self-doubt; however, you now feel much better. For improved mood, I recommend upbeat or relaxing music. Your chosen track is fantastic, and I'm glad you enjoy it!",
    "tags": ["self-doubt"]
  },
  // February 2025 moods
  {
    "username": "testuser",
    "date": "2025-02-11T10:00:00.000000",
    "moodScore": 4,
    "emotions": ["happy", "surprised"],
    "primaryEmotion": "happy",
    "notes": "Had a great meeting with the team",
    "tags": ["work", "team"]
  },
  {
    "username": "testuser",
    "date": "2025-02-14T12:30:00.000000",
    "moodScore": 5,
    "emotions": ["happy", "surprised"],
    "primaryEmotion": "happy",
    "notes": "Valentine's Day celebration",
    "tags": ["personal", "holiday"]
  },
  {
    "username": "testuser",
    "date": "2025-02-28T18:00:00.000000",
    "moodScore": 3,
    "emotions": ["neutral", "disgusted"],
    "primaryEmotion": "neutral",
    "notes": "Long day at work",
    "tags": ["work", "stress"]
  },
  // January 2025 moods
  {
    "username": "testuser",
    "date": "2025-01-02T09:00:00.000000",
    "moodScore": 2,
    "emotions": ["sad", "angry"],
    "primaryEmotion": "sad",
    "notes": "First Monday back after holidays",
    "tags": ["work", "stress"]
  },
  {
    "username": "testuser",
    "date": "2025-01-06T14:00:00.000000",
    "moodScore": 4,
    "emotions": ["happy", "surprised"],
    "primaryEmotion": "happy",
    "notes": "Weekend getaway",
    "tags": ["personal", "holiday"]
  },
  {
    "username": "testuser",
    "date": "2025-01-09T15:00:00.000000",
    "moodScore": 3,
    "emotions": ["neutral", "disgusted"],
    "primaryEmotion": "neutral",
    "notes": "Project deadline approaching",
    "tags": ["work", "stress"]
  },
  {
    "username": "testuser",
    "date": "2025-01-11T11:00:00.000000",
    "moodScore": 5,
    "emotions": ["happy", "surprised"],
    "primaryEmotion": "happy",
    "notes": "Received promotion",
    "tags": ["work", "achievement"]
  },
  {
    "username": "testuser",
    "date": "2025-01-14T13:00:00.000000",
    "moodScore": 2,
    "emotions": ["sad", "angry"],
    "primaryEmotion": "sad",
    "notes": "Argument with colleague",
    "tags": ["work", "conflict"]
  },
  {
    "username": "testuser",
    "date": "2025-01-26T17:00:00.000000",
    "moodScore": 4,
    "emotions": ["happy", "surprised"],
    "primaryEmotion": "happy",
    "notes": "Friend's birthday celebration",
    "tags": ["personal", "social"]
  },
  {
    "username": "testuser",
    "date": "2025-01-31T16:00:00.000000",
    "moodScore": 3,
    "emotions": ["neutral", "disgusted"],
    "primaryEmotion": "neutral",
    "notes": "Routine day at work",
    "tags": ["work", "routine"]
  }
  // Add new mood predictions here
  // Example:
  // {
  //   "username": "testuser",
  //   "date": "2025-04-01T10:30:00.000000",
  //   "moodScore": 4,
  //   "emotions": ["happy", "excited"],
  //   "primaryEmotion": "happy",
  //   "notes": "Feeling happy today because of the good news received.",
  //   "tags": ["good news", "work"]
  // }
];
