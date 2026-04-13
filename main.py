import base64
import os
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# --- COLD START MITIGATION: LIFESPAN MANAGER ---
# This runs BEFORE the server starts accepting requests
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX", "mit-bengaluru-v2")
    resources["index"] = pc.Index(index_name)
    
    # Initialize Groq
    resources["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Optional: Send a tiny "warm-up" query to Pinecone
    try:
        resources["index"].query(vector=[0.0]*384, top_k=1)
        print("ARIA: Resources warmed up and ready.")
    except:
        print("ARIA: Warm-up query failed, but initialization is complete.")
        
    yield
    # Clean up on shutdown
    resources.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

SYSTEM_PROMPT = """You are ARIA, the official Placement Intelligence Voice AI for MIT Bengaluru. 
Limit responses to 1 to 2 short sentences. No markdown, no formatting. Plain text only."""

# --- UPDATED SEARCH FUNCTION ---
def search_campus_data(user_query: str, top_k: int = 3) -> str:
    # Use the pre-warmed index from global resources
    index = resources.get("index")
    if not index:
        return "System initializing."
        
    try:
        results = index.query(
            vector=[0.0] * 384, 
            top_k=top_k,
            include_metadata=True
        )
        context_parts = [m["metadata"].get("text", "") for m in results.get("matches", [])]
        return "\n\n".join(context_parts) if context_parts else "No specific context found."
    except Exception as e:
        print(f"RAG Error: {e}")
        return "MIT Bengaluru placement and campus information."

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        context = search_campus_data(request.query)
        groq_client = resources.get("groq")

        groq_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {request.query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        
        bot_text = groq_response.choices[0].message.content
        
        tts = gTTS(text=bot_text, lang='en', tld='co.in') 
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_base64 = base64.b64encode(audio_fp.getvalue()).decode('utf-8')
        
        return {
            "text": bot_text,
            "audio_base64": audio_base64
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
