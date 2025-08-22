from dotenv import load_dotenv
import os
import datetime
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
from bson.objectid import ObjectId

load_dotenv()

app = FastAPI()

# Allow CORS for local dev / Next.js frontend
cors_origin = os.getenv("CORS_ORIGIN", "*")
origins = [o.strip() for o in cors_origin.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def _select_db_name_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    path_db = (parsed.path or "").lstrip("/")
    if path_db:
        return path_db
    env_db = os.getenv("MONGODB_DB")
    if env_db:
        return env_db
    return "test"


# Direct MongoDB connection using pymongo
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set in environment")

MONGODB_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", "10000"))
mongo_client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=MONGODB_TIMEOUT_MS,
    connectTimeoutMS=MONGODB_TIMEOUT_MS,
    socketTimeoutMS=MONGODB_TIMEOUT_MS,
)
mongo_db_name = _select_db_name_from_uri(MONGODB_URI)
mongo_db = mongo_client[mongo_db_name]
collection = mongo_db["chathistorysessions"]
reg_collection = mongo_db["registration_responses"]


@app.on_event("startup")
async def on_startup():
    try:
        mongo_client.admin.command("ping")
        print(f"[Mongo] Connected. Using db='{mongo_db_name}', coll='chathistorysessions'")
    except Exception as e:
        print("[Mongo] Ping failed:", str(e))


@app.on_event("shutdown")
async def on_shutdown():
    try:
        mongo_client.close()
    except Exception:
        pass


# ----------
# Pydantic IO
# ----------
class ChatMessageIO(BaseModel):
    type: str
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatHistoryRequest(BaseModel):
    sessionId: Optional[str] = None
    userEmail: Optional[str] = None
    userName: Optional[str] = None
    userId: Optional[str] = None


class MegaChatRequest(BaseModel):
    message: str
    chatHistory: List[Dict[str, Any]] = []
    sessionId: Optional[str] = None
    userEmail: Optional[str] = None
    userName: Optional[str] = None
    userId: Optional[str] = None
    user: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    jwt_secret = os.getenv("JWT_SECRET")
    return {"message": "Hello, World!", "jwt_secret": jwt_secret}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/db/health")
async def db_health():
    try:
        res = mongo_client.admin.command("ping")
        return {"ok": True, "ping": res}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    
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
        
        # Retrieve chat history from MongoDB
        try:
            # Optional: ensure the user/session exists so future writes succeed
            if user_email and user_id and user_name and session_id:
                _ensure_user(user_email, user_id, user_name)
                _ensure_session_for_user(user_email, session_id)

            # Fetch messages
            msgs = _get_session_messages(session_id) if session_id else None
            chat_history = []
            messages_payload = []
            if msgs:
                # Convert to frontend shape (type/content)
                for m in msgs:
                    role = m.get("role", "assistant")
                    created = m.get("createdAt")
                    ts = (
                        created.isoformat() if hasattr(created, "isoformat") else str(created)
                    ) if created else None
                    chat_history.append(
                        {
                            "type": "user" if role == "user" else "ai",
                            "content": m.get("content", ""),
                            "timestamp": ts,
                        }
                    )
                    messages_payload.append(
                        {
                            "role": "user" if role == "user" else "assistant",
                            "content": m.get("content", ""),
                            "timestamp": ts,
                        }
                    )
                print(
                    f"[FastAPI] Found {len(chat_history)} messages for session {session_id}, user {user_email}"
                )
            else:
                print(
                    f"[FastAPI] No chat history found for session {session_id}, user {user_email}."
                )
        except Exception as e:
            print(f"[FastAPI] Error accessing MongoDB: {str(e)}")
            chat_history = []
            messages_payload = []
        
        # Prepare response message based on whether we found history or not
        response_message = f"Retrieved chat history for session {session_id}"
        if not chat_history:
            response_message = f"No chat history found for session {session_id}"
        
        return JSONResponse(content={
            "sessionId": session_id,
            "chatHistory": chat_history,
            "messages": messages_payload,  # compatibility field
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

        # Fetch user's business info from registration_responses
        business_doc = _get_business_info(user_email, user_id)
        business_profile = _format_business_profile(business_doc) if business_doc else ""

        # Build personalized prompt using LangChain
        history_str = "\n".join([
            f"{'User' if m.get('type') == 'user' else 'AI'}: {m.get('content', '')}" for m in chat_history
        ])
        profile_block = f"\nBusiness Profile:\n{business_profile}" if business_profile else ""
        prompt_text = f"""
You are a helpful AI assistant. Personalize your response for the following user:
Name: {user_name if user_name else 'Unknown'}
Email: {user_email if user_email else 'Unknown'}
{profile_block}
Session ID: {session_id if session_id else 'No session'}
Chat history:
{history_str}
User: {message}
AI:
"""
        # Ensure user + session exist before storing
        if session_id and user_email:
            if not user_name or not user_id:
                user = body.get("user") or {}
                user_name = user_name or user.get("name")
                user_id = user_id or user.get("userId")
            if user_id and user_name:
                _ensure_user(user_email, user_id, user_name)
                _ensure_session_for_user(user_email, session_id)

        # Try to get AI response via LangChain + Gemini
        ai_response = None
        try:
            chat = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME", "gemini-1.5-flash"), google_api_key=GEMINI_API_KEY)
            prompt = ChatPromptTemplate.from_template("{input}")
            chain = prompt | chat
            response = chain.invoke({"input": prompt_text})
            ai_response = response.content
            print("[FastAPI] LangChain Gemini response:", ai_response)
        except Exception as gen_e:
            print("[FastAPI] Gemini generation failed:", str(gen_e))
        
        # Persist the conversation to MongoDB
        if session_id and user_email:
            user_msg = {
                "role": "user",
                "content": message,
                "createdAt": datetime.datetime.utcnow(),
                "metadata": {},
            }

            try:
                msgs_to_add = [user_msg]
                if ai_response is not None:
                    msgs_to_add.append(
                        {
                            "role": "assistant",
                            "content": ai_response,
                            "createdAt": datetime.datetime.utcnow(),
                            "metadata": {},
                        }
                    )
                _add_messages_to_session(
                    session_id,
                    msgs_to_add,
                    create_if_missing=True,
                    user_data={
                        "email": user_email,
                        "userId": user_id or "",
                        "username": user_name or (user_email.split("@")[0] if user_email else "user"),
                    },
                )
            except Exception as db_e:
                # Log but do not fail the request
                print("[FastAPI] Failed to persist messages:", str(db_e))

        return JSONResponse(content={"response": ai_response or ""}, status_code=200)
    except Exception as e:
        print("[FastAPI] Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


# -----------------------
# Direct pymongo helpers
# -----------------------
def _ensure_user(email: str, user_id: str, username: str):
    now = datetime.datetime.utcnow()
    collection.update_one(
        {"email": email.lower()},
        {
            "$setOnInsert": {
                "email": email.lower(),
                "userId": user_id,
                "username": username,
                "sessions": [],
                "createdAt": now,
            },
            "$set": {"updatedAt": now},
        },
        upsert=True,
    )


def _ensure_session_for_user(email: str, session_id: str, metadata: Optional[Dict[str, Any]] = None):
    now = datetime.datetime.utcnow()
    collection.update_one(
        {"email": email.lower(), "sessions.sessionId": {"$ne": session_id}},
        {
            "$push": {
                "sessions": {
                    "sessionId": session_id,
                    "createdAt": now,
                    "lastActive": now,
                    "metadata": metadata or {},
                    "messages": [],
                }
            },
            "$set": {"updatedAt": now},
        },
    )


def _add_messages_to_session(
    session_id: str,
    messages: List[Dict[str, Any]],
    create_if_missing: bool = False,
    user_data: Optional[Dict[str, Any]] = None,
):
    now = datetime.datetime.utcnow()
    res = collection.update_one(
        {"sessions.sessionId": session_id},
        {
            "$push": {"sessions.$.messages": {"$each": messages}},
            "$set": {"sessions.$.lastActive": now, "updatedAt": now},
        },
    )
    if res.modified_count == 0 and create_if_missing and user_data:
        _ensure_user(user_data["email"], user_data["userId"], user_data["username"])
        _ensure_session_for_user(user_data["email"], session_id)
        collection.update_one(
            {"sessions.sessionId": session_id},
            {
                "$push": {"sessions.$.messages": {"$each": messages}},
                "$set": {"sessions.$.lastActive": now, "updatedAt": now},
            },
        )


def _get_session_messages(session_id: str, limit: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    doc = collection.find_one(
        {"sessions.sessionId": session_id},
        {"sessions": {"$elemMatch": {"sessionId": session_id}}},
    )
    if not doc:
        return None
    session = (doc.get("sessions") or [{}])[0]
    msgs = session.get("messages", [])
    if limit and limit > 0:
        return msgs[-limit:]
    return msgs


def _get_business_info(user_email: Optional[str], user_id: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        # Prefer lookup by ObjectId if possible
        if user_id:
            try:
                oid = ObjectId(user_id)
                doc = reg_collection.find_one({"_id": oid})
                if doc:
                    return doc
            except Exception:
                pass

        # Fallback to email lookup
        if user_email:
            doc = reg_collection.find_one({"contact_info_email": user_email})
            if doc:
                return doc
    except Exception as e:
        print("[FastAPI] Business info lookup failed:", str(e))
    return None


def _format_business_profile(doc: Dict[str, Any]) -> str:
    keys = [
        ("Full Name", "contact_info_fullname"),
        ("Email", "contact_info_email"),
        ("Business Name", "contact_info_bizname"),
        ("City/State", "contact_info_city_state"),
        ("Ownership", "ownership"),
        ("Role", "role"),
        ("Business Age", "biz_age"),
        ("Revenue", "revenue"),
        ("Goals", "goals"),
        ("Social Media Use", "socialmedia_use"),
    ]
    lines: List[str] = []
    for label, field in keys:
        val = doc.get(field)
        if val is not None and str(val).strip() != "":
            lines.append(f"- {label}: {val}")
    return "\n".join(lines)