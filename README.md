# Business Coach FastAPI

FastAPI backend wired to MongoDB to persist chat sessions/messages compatible with the Next.js Mongoose schema.

## Env

Required vars in `.env`:

- MONGODB_URI=mongodb+srv://...
- MONGODB_DB=chatdb (optional; defaults to `chatdb`)
- GEMINI_API_KEY=...
- MODEL_NAME=gemini-1.5-flash
- CORS_ORIGIN=http://localhost:3001

## Run

- Install: `pip install -r requirements.txt`
- Start: `uvicorn main:app --reload`

## Endpoints

- POST /api/chat-history
  - body: { sessionId, userEmail, userName, userId }
  - returns: { chatHistory:[{type,content,timestamp}], hasHistory }

- POST /api/megachat
  - body: { message, sessionId, userEmail, userName, userId, chatHistory }
  - stores the user message and AI reply into MongoDB
