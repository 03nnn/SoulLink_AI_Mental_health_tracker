const mongoose = require('mongoose');

const moodSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  username: {
    type: String,
    required: true
  },
  date: {
    type: Date,
    required: true
  },
  moodScore: {
    type: Number,
    required: true,
    min: 1,
    max: 5
  },
  primaryEmotion: {
    type: String,
    required: true,
    enum: ['happy', 'sad', 'angry', 'fear', 'surprised', 'disgusted', 'neutral']
  },
  emotions: [{
    type: String,
    enum: ['happy', 'sad', 'angry', 'fear', 'surprised', 'disgusted', 'neutral']
  }],
  notes: {
    type: String,
    default: ''
  },
  tags: [{
    type: String
  }],
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Update updatedAt timestamp
moodSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Add static methods for common queries
moodSchema.statics = {
  // Get all moods for a user for a specific date
  findByDate: async function(userId, date) {
    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    return this.find({
      userId,
      date: {
        $gte: startOfDay,
        $lte: endOfDay
      }
    }).sort({ date: -1 }); // Sort by time in descending order
  },

  // Get the latest mood for a user for a specific date
  getLatestMood: async function(userId, date) {
    const moods = await this.findByDate(userId, date);
    return moods[0]; // Return the latest mood (first in sorted array)
  },

  // Get mood statistics for a month
  getMonthlyStats: async function(userId, startDate, endDate) {
    const moods = await this.find({
      userId,
      date: {
        $gte: new Date(startDate),
        $lte: new Date(endDate)
      }
    }).sort({ date: -1 });

    // Get unique dates
    const dates = [...new Set(moods.map(mood => 
      new Date(mood.date).setHours(0, 0, 0, 0)
    ))];

    // Get latest mood for each date
    const dailyMoods = dates.map(date => {
      const moodsOnDate = moods.filter(mood => 
        new Date(mood.date).setHours(0, 0, 0, 0) === date
      );
      return moodsOnDate[0]; // Latest mood for that date
    });

    // Calculate statistics
    const stats = {
      totalEntries: moods.length,
      uniqueDays: dates.length,
      averageScore: moods.length > 0 ? 
        moods.reduce((sum, mood) => sum + mood.moodScore, 0) / moods.length : 0,
      moodDistribution: {
        happy: 0,
        sad: 0,
        angry: 0,
        fear: 0,
        surprised: 0,
        disgusted: 0,
        neutral: 0
      }
    };

    // Count moods for each emotion
    moods.forEach(mood => {
      stats.moodDistribution[mood.primaryEmotion]++;
    });

    return {
      dailyMoods,
      stats
    };
  }
};

const Mood = mongoose.model('Mood', moodSchema);

module.exports = Mood;
