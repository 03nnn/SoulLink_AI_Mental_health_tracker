# MongoDB Schema for Mood Tracker App

## Collections

### Moods Collection
```javascript
{
  _id: ObjectId,
  userId: String,
  date: Date,
  moodScore: Number, // 1-5 scale (1: very negative, 3: neutral, 5: very positive)
  emotions: [String], // array of emotions felt (e.g., "happy", "anxious", "excited")
  primaryEmotion: String, // the strongest emotion felt
  notes: String, // daily summary or journal entry
  tags: [String], // contextual tags (e.g., "work", "family", "exercise")
  createdAt: Date,
  updatedAt: Date
}

