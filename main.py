import base64
import os
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# 1. THE CORS FIX: Explicitly allow local and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "*" # Keeps it open for other team members during testing
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    query: str

SYSTEM_PROMPT = """
You are the official, highly professional AI Assistant for Manipal Institute of Technology (MIT) Bengaluru.
Write in pure, conversational prose under 3 sentences. No bullets, no bolding, no emojis.
"""

# --- TM2 PINE CONE RAG SETUP (MEMORY OPTIMIZED) ---
_index = None

def _get_resources():
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index("mit-bengaluru")
    return _index

def search_campus_data(user_query: str, top_k: int = 3) -> str:
    index = _get_resources()
    
    # IMPORTANT: TM2 must replace this line with an API call (e.g., Pinecone Inference)
    # The server cannot handle local SentenceTransformer/PyTorch
    return "MIT Bengaluru offers B.Tech programs. Please check the official website for details."

# -------------------------------

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        context = search_campus_data(request.query)
        
        groq_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {request.query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=150,
        )
        
        bot_text = groq_response.choices[0].message.content
        
        # Audio generation
        tts = gTTS(text=bot_text, lang='en', tld='co.in') 
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_base64 = base64.b64encode(audio_fp.getvalue()).decode('utf-8')
        
        return {
            "text": bot_text,
            "audio_base64": audio_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
