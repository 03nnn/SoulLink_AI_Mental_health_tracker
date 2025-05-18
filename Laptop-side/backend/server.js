const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const dotenv = require("dotenv");

// Load environment variables
dotenv.config();

// Import models
const User = require("./models/User");
const Mood = require("./models/Mood");

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'x-user-id']
}));
app.use(express.json());

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log("Connected to MongoDB"))
  .catch(err => {
    console.error("MongoDB connection error:", err);
    process.exit(1);
  });

// Simple authentication middleware (no JWT)
const authenticateUser = async (req, res, next) => {
  try {
    // Get user ID from query parameter or header
    const userId = req.query.userId || req.headers['x-user-id'];
    
    console.log("Authentication attempt with userId:", userId);
    
    if (!userId) {
      console.log("No userId provided");
      return res.status(401).json({ message: "Authentication required" });
    }
    
    // Find user by ID - use toString() to handle ObjectId comparison
    let user;
    try {
      if (mongoose.Types.ObjectId.isValid(userId)) {
        user = await User.findById(userId);
      } else {
        user = await User.findOne({ _id: userId });
      }
    } catch (err) {
      console.error("Error finding user:", err);
    }
    
    if (!user) {
      console.log("User not found with ID:", userId);
      return res.status(404).json({ message: "User not found" });
    }
    
    console.log("User authenticated:", user.username);
    
    // Attach user to request object
    req.user = user;
    next();
  } catch (error) {
    console.error("Authentication error:", error);
    return res.status(401).json({ message: "Authentication failed" });
  }
};

// Test route to check authentication
app.get("/api/test", authenticateUser, (req, res) => {
  res.json({ message: "Authentication successful", userId: req.user._id });
});

// Auth Routes
app.post("/api/auth/register", async (req, res) => {
  try {
    const { username, email, password } = req.body;
    
    // Validate input
    if (!username || !email || !password) {
      return res.status(400).json({ message: "All fields are required" });
    }
    
    // Check if user already exists
    const existingUser = await User.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(400).json({ message: "User already exists" });
    }
    
    // Create new user
    const user = new User({ username, email, password });
    await user.save();
    
    // Return user info (excluding password)
    const userObj = user.toObject();
    delete userObj.password;
    
    res.status(201).json(userObj);
  } catch (error) {
    console.error("Registration error:", error);
    res.status(500).json({ message: "Server error" });
  }
});

app.post("/api/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Validate input
    if (!email || !password) {
      return res.status(400).json({ message: "Email and password are required" });
    }
    
    // Find user
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ message: "Invalid credentials" });
    }
    
    // Check password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(400).json({ message: "Invalid credentials" });
    }
    
    // Return user info (excluding password)
    const userObj = user.toObject();
    delete userObj.password;
    
    res.json(userObj);
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ message: "Server error" });
  }
});

app.get("/api/auth/user", authenticateUser, (req, res) => {
  // Return user info (excluding password)
  const userObj = req.user.toObject();
  delete userObj.password;
  
  res.json(userObj);
});

// Mood Routes
app.get("/api/moods", authenticateUser, async (req, res) => {
  try {
    const { startDate, endDate } = req.query;
    const userId = req.user._id;
    
    console.log("Fetching moods for user:", userId);
    console.log("Date range:", startDate, "to", endDate);
    
    const query = { userId };
    
    if (startDate && endDate) {
      query.date = {
        $gte: new Date(startDate),
        $lte: new Date(endDate)
      };
    }
    
    // Get all moods in the date range
    const moods = await Mood.find(query).sort({ date: -1 });
    
    // Calculate statistics
    const stats = {
      totalEntries: moods.length,
      uniqueDays: new Set(moods.map(mood => 
        new Date(mood.date).setHours(0, 0, 0, 0)
      )).size,
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

    // Get latest mood for each date
    const dates = new Set(moods.map(mood => 
      new Date(mood.date).setHours(0, 0, 0, 0)
    ));

    const dailyMoods = Array.from(dates).map(date => {
      const moodsOnDate = moods.filter(mood => 
        new Date(mood.date).setHours(0, 0, 0, 0) === date
      );
      return moodsOnDate[0]; // Latest mood for that date
    });

    res.json({
      moods,
      dailyMoods,
      stats
    });
  } catch (error) {
    console.error("Error fetching moods:", error);
    res.status(500).json({ message: "Server error" });
  }
});

app.post("/api/moods", authenticateUser, async (req, res) => {
  try {
    const { date, moodScore, emotions, primaryEmotion, notes, tags } = req.body;
    const userId = req.user._id;
    const username = req.user.username;
    
    // Validate input
    if (!date || !moodScore || !primaryEmotion) {
      return res.status(400).json({ message: "Date, mood score, and primary emotion are required" });
    }
    
    // Create new mood
    const mood = new Mood({
      userId,
      username,
      date: new Date(date),
      moodScore,
      emotions: emotions || [],
      primaryEmotion,
      notes: notes || "",
      tags: tags || []
    });
    
    await mood.save();
    res.status(201).json(mood);
  } catch (error) {
    console.error("Error creating mood:", error);
    res.status(500).json({ message: "Server error" });
  }
});

// Start server
app.listen(PORT, () => {
  console.log('MongoDB API server running on port ${PORT}');
  console.log('Access the API at http://192.168.137.196:${PORT}/api');
});