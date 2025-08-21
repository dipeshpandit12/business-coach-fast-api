from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

@app.get("/")
async def root():
    jwt_secret = os.getenv("JWT_SECRET")
    return {"message": "Hello, World!", "jwt_secret": jwt_secret}

@app.post("/api/megachat")
async def megachat_post(request: Request):
    try:
        body = await request.json()
        print("[FastAPI] Received data:", body)
        message = body.get("message", "")
        chat_history = body.get("chatHistory", [])
        user = body.get("user", None)
        print("[FastAPI] message:", message)
        print("[FastAPI] chat_history:", chat_history)
        print("[FastAPI] user:", user)

        # Build personalized prompt using LangChain
        user_email = user.get("email") if user else None
        user_name = user.get("name") if user else None
        history_str = "\n".join([
            f"{'User' if m.get('type') == 'user' else 'AI'}: {m.get('content', '')}" for m in chat_history
        ])
        prompt_text = f"""
You are a helpful AI assistant. Personalize your response for the following user:
Name: {user_name if user_name else 'Unknown'}
Email: {user_email if user_email else 'Unknown'}
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
        print("[FastAPI] LangChain Gemini response:", response.content)
        return JSONResponse(content={"response": response.content}, status_code=200)
    except Exception as e:
        print("[FastAPI] Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)