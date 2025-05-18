# Mood Tracker Backend

This is the backend server for the mood tracker application, built with Node.js, Express, and MongoDB.

## Prerequisites

- Node.js (v18 or higher)
- MongoDB (v6.0 or higher)
- npm or yarn

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the root directory with the following content:
   ```
   PORT=3001
   MONGODB_URI=mongodb://localhost:27017/mood-tracker
   ```

## Running the Server

1. Start the MongoDB server
2. Start the backend server:
   ```bash
   npm start
   ```
   
   or in development mode:
   ```bash
   npm run dev
   ```

## API Endpoints

### Authentication

- POST `/api/auth/register` - Register a new user
- POST `/api/auth/login` - Login existing user
- GET `/api/auth/user` - Get current user info

### Moods

- GET `/api/moods` - Get moods for a date range
- POST `/api/moods` - Create or update a mood

## Error Handling

The server includes comprehensive error handling and logging. All API responses include appropriate HTTP status codes and error messages.
