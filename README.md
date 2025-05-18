# SoulLink - Emotional Well-being Companion

SoulLink is an innovative emotional well-being companion that helps track and understand your mental health over time. It combines a physical robot interface with advanced AI to provide emotional support and insights.

## 🌟 Features

- **Real-time Emotion Analysis**: Uses DeepFace and computer vision to analyze facial expressions
- **Conversational AI**: Powered by Ollama for natural, human-like interactions
- **Multi-modal Interaction**: Combines voice, touch, and visual feedback
- **Mobile Dashboard**: Track your emotional journey and gain insights
- **Physical Robot Interface**: Interactive robot with expressive face and touch response

## 🏗️ System Architecture

### 1. Laptop-side Backend
Location: `/Laptop-side`
- Handles core AI processing and database operations
- Implements DeepFace for emotion recognition
- Manages conversation history and emotional state tracking
- Generates AI responses using Ollama
- Maintains the central database for user data

### 2. Raspberry Pi Interface
Location: `/Pi-side`
- Manages the physical robot's hardware
- Handles wake word detection
- Processes speech-to-text and text-to-speech
- Controls robot movements and facial expressions
- Processes touch sensor inputs
- Communicates with the laptop backend

### 3. Mobile Application
Location: `/Mobile-App`
- React Native application for iOS and Android
- Visualizes emotional trends and history
- Provides insights and analytics
- Syncs with the laptop backend database

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+ (for mobile app)
- Raspberry Pi 4 (for robot interface)
- Ollama installed and configured
- DeepFace dependencies

### Installation

#### Laptop-side Setup
```bash
cd Laptop-side
pip install -r requirements.txt
python play.py
```

#### Pi-side Setup
```bash
cd Pi-side
pip install -r requirements.txt
python pi_final.py
```

#### Mobile App Setup
```bash
cd Mobile-App
yarn install
yarn start
```

## 🤖 Hardware Requirements

### Robot Components
- Raspberry Pi 4
- Touch sensors
- Servo motors for movement
- LCD display for facial expressions
- Microphone for voice input
- Speaker for audio output

## 📊 Database Schema

The system maintains several key data points:
- Emotion logs with timestamps
- Conversation history
- User interaction patterns
- Touch sensor data
- Response effectiveness metrics

## 📱 Mobile App Features

- Emotion timeline visualization
- Daily mood tracking
- Conversation history
- Insights and patterns

## 🔧 Troubleshooting

Common issues and solutions can be found in the respective component directories.
-When the entry in the Database is deleted manually, the mobile app may not show the updated data. 

## 📜 License

The 3D parts are taken from this repository: https://github.com/CodersCafeTech/Emo/tree/main/3D%20Design
-currently working for a license, applying for publication-18/may/2025.

## 🙏 Acknowledgments

- DeepFace for emotion recognition
- Ollama for conversational AI
- React Native for cross-platform mobile development
- Raspberry Pi community for hardware support

##  Website 
Visit our website for more information:- https://soul-link-two.vercel.app/
refer this to install piper ins raspberry pi- https://www.youtube.com/watch?v=Bw9C4rqO63c
