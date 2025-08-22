from dotenv import load_dotenv
import os
import datetime
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# In-memory store for chat messages (in a production app, use a proper database)
# Structure: {session_id: {user_email: [messages]}}
chat_store = {}

@app.get("/")
async def root():
    jwt_secret = os.getenv("JWT_SECRET")
    return {"message": "Hello, World!", "jwt_secret": jwt_secret}
    
@app.post("/api/chat-history")
async def get_chat_history(request: Request):
    try:
        # Parse request body
        body = await request.json()
        session_id = body.get("sessionId")
        user_email = body.get("userEmail")
        user_name = body.get("userName")
        user_id = body.get("userId")
        
        print("[FastAPI] Chat history request:", {
            "sessionId": session_id,
            "userEmail": user_email,
            "userName": user_name,
            "userId": user_id
        })
        
        # Default empty chat history
        chat_history = []
        
        # Check if we have all required fields to retrieve chat history
        if not session_id or not user_email:
            print("[FastAPI] Missing required fields for chat history retrieval")
            return JSONResponse(content={
                "chatHistory": [],
                "message": "Missing required fields for chat history retrieval"
            }, status_code=400)
        
        # Retrieve chat history from in-memory store if available
        try:
            if session_id in chat_store and user_email in chat_store[session_id]:
                chat_history = chat_store[session_id][user_email]
                print(f"[FastAPI] Found {len(chat_history)} messages for session {session_id}, user {user_email}")
            else:
                # If no history found, provide a welcome message
                chat_history = []
                print(f"[FastAPI] No chat history found for session {session_id}, user {user_email}.")
                
                # Initialize the store for this session and user for future messages
                if session_id not in chat_store:
                    chat_store[session_id] = {}
                if user_email not in chat_store[session_id]:
                    chat_store[session_id][user_email] = []
        except Exception as e:
            print(f"[FastAPI] Error accessing chat store: {str(e)}")
            chat_history = []
        
        # Prepare response message based on whether we found history or not
        response_message = f"Retrieved chat history for session {session_id}"
        if not chat_history:
            response_message = f"No chat history found for session {session_id}"
        
        return JSONResponse(content={
            "chatHistory": chat_history,
            "message": response_message,
            "status": "success",
            "hasHistory": len(chat_history) > 0
        }, status_code=200)
    except Exception as e:
        print("[FastAPI] Error getting chat history:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/megachat")
async def megachat_post(request: Request):
    try:
        body = await request.json()
        print("[FastAPI] Received data:", body)
        
        # Extract data from the request body
        message = body.get("message", "")
        chat_history = body.get("chatHistory", [])
        
        # Get user info from top-level fields first (new format)
        session_id = body.get("sessionId")
        user_email = body.get("userEmail")
        user_name = body.get("userName")
        user_id = body.get("userId")
        
        # Fall back to user object if top-level fields are not available (backward compatibility)
        user = body.get("user", None)
        if not user_email and user:
            user_email = user.get("email")
        if not user_name and user:
            user_name = user.get("name")
        if not user_id and user:
            user_id = user.get("userId")
            
        print("[FastAPI] message:", message)
        print("[FastAPI] chat_history length:", len(chat_history))
        print("[FastAPI] session_id:", session_id)
        print("[FastAPI] user_email:", user_email)
        print("[FastAPI] user_name:", user_name)
        print("[FastAPI] user_id:", user_id)

        # Build personalized prompt using LangChain
        history_str = "\n".join([
            f"{'User' if m.get('type') == 'user' else 'AI'}: {m.get('content', '')}" for m in chat_history
        ])
        prompt_text = f"""
You are a helpful AI assistant. Personalize your response for the following user:
Name: {user_name if user_name else 'Unknown'}
Email: {user_email if user_email else 'Unknown'}
Session ID: {session_id if session_id else 'No session'}
Chat history:
{history_str}
User: {message}
AI:
"""
        # Use LangChain to call Gemini
        chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
        prompt = ChatPromptTemplate.from_template("{input}")
        chain = prompt | chat
        response = chain.invoke({"input": prompt_text})
        ai_response = response.content
        print("[FastAPI] LangChain Gemini response:", ai_response)
        
        # Store the conversation in our in-memory chat store if we have all required info
        if session_id and user_email:
            # Initialize session if it doesn't exist
            if session_id not in chat_store:
                chat_store[session_id] = {}
            
            # Initialize user email if it doesn't exist
            if user_email not in chat_store[session_id]:
                chat_store[session_id][user_email] = []
            
            # Add the user's message to the history
            chat_store[session_id][user_email].append({
                "type": "user",
                "content": message,
                "timestamp": str(datetime.datetime.now())
            })
            
            # Add the AI's response to the history
            chat_store[session_id][user_email].append({
                "type": "ai",
                "content": ai_response,
                "timestamp": str(datetime.datetime.now())
            })
            
            print(f"[FastAPI] Stored messages for session {session_id}, user {user_email}. Total messages: {len(chat_store[session_id][user_email])}")
        
        return JSONResponse(content={"response": ai_response}, status_code=200)
    except Exception as e:
        print("[FastAPI] Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)