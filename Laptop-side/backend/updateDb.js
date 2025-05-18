const mongoose = require('mongoose');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

// Import models
const User = require('./models/User');
const Mood = require('./models/Mood');

// Import mood data
const newMoods = require('./moods_data');

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

// Function to update the database with new mood entries
const updateDatabase = async () => {
  try {
    // Process each new mood
    for (const moodData of newMoods) {
      try {
        // Find user by username
        const user = await User.findOne({ username: moodData.username });
        if (!user) {
          console.log(`User ${moodData.username} not found. Skipping mood.`);
          continue;
        }

        // Create new mood
        const mood = new Mood({
          userId: user._id,
          username: user.username,
          date: new Date(moodData.date),
          moodScore: moodData.moodScore,
          primaryEmotion: moodData.primaryEmotion,
          emotions: moodData.emotions || [],
          notes: moodData.notes || '',
          tags: moodData.tags || []
        });

        await mood.save();
        console.log(`Added mood for ${moodData.date} for user ${user.username}`);
      } catch (error) {
        console.error(`Error processing mood ${moodData.date}:`, error);
      }
    }

    console.log('Database update completed successfully');
  } catch (error) {
    console.error('Error updating database:', error);
    process.exit(1);
  }
};

// Run the update
updateDatabase();
