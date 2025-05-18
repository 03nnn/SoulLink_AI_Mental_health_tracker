const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

// Import models
const User = require('./models/User');
const Mood = require('./models/Mood');

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

// Generate sample moods for a full month of data
const generateAdditionalMoods = async (userId, username) => {
  const emotions = ['happy', 'sad', 'angry', 'fear', 'surprised', 'disgusted', 'neutral'];
  const today = new Date();
  const moods = [];

  // Generate moods for the past 30 days
  for (let i = 0; i < 30; i++) {
    const date = new Date();
    date.setDate(today.getDate() - i);

    const mood = {
      userId,
      username,
      date,
      moodScore: Math.floor(Math.random() * 5) + 1,
      primaryEmotion: emotions[Math.floor(Math.random() * emotions.length)],
      emotions: [emotions[Math.floor(Math.random() * emotions.length)]],
      notes: `Sample mood entry for ${date.toDateString()}`,
      tags: ['sample', 'test']
    };

    moods.push(mood);
  }

  return moods;
};

// Initialize the database
const initializeDb = async () => {
  try {
    // Create a test user
    const user = new User({
      username: 'testuser',
      email: 'test@example.com',
      password: 'password123',
      preferences: {
        theme: 'light',
        reminderTime: '20:00',
        weekStartsOn: 1
      }
    });

    await user.save();
    console.log('User created:', user._id);

    // Generate and save sample moods
    const sampleMoods = await generateAdditionalMoods(user._id, user.username);
    const moodPromises = sampleMoods.map(mood => new Mood(mood).save());
    await Promise.all(moodPromises);
    console.log('Sample moods created:', sampleMoods.length);

    console.log('Database initialization completed successfully');
  } catch (error) {
    console.error('Error initializing database:', error);
    process.exit(1);
  }
};

// Run the initialization
initializeDb();
