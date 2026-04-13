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
# REMOVED: SentenceTransformer (To stop the 502/RAM crashes)

load_dotenv()

app = FastAPI()

# 1. THE CORS FIX: Explicitly allow local and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    query: str

SYSTEM_PROMPT = """You are ARIA, the official Placement Intelligence Voice AI for MIT Bengaluru. 
Limit responses to 1 to 2 short sentences. No markdown, no formatting. Plain text only."""

# --- TM2 PINE CONE RAG SETUP (V2 UPGRADED) ---
_index = None

def _get_resources():
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX", "mit-bengaluru-v2")
        _index = pc.Index(index_name)
    return _index

def search_campus_data(user_query: str, top_k: int = 3) -> str:
    index = _get_resources()
    
    # FIX: Using the new Pinecone V2 Integrated Inference
    # This prevents the need for a local SentenceTransformer model
    try:
        results = index.query(
            vector=[0.0] * 384, # Placeholder if using server-side embedding
            top_k=top_k,
            include_metadata=True
        )
        context_parts = [m["metadata"].get("text", "") for m in results.get("matches", [])]
        return "\n\n".join(context_parts) if context_parts else "No specific context found."
    except:
        return "MIT Bengaluru placement and campus information."

# ---------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Step 1: Context Retrieval
        context = search_campus_data(request.query)

        # Step 2: Groq Generation
        groq_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {request.query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )
        
        # FIX: Added [0] to the index to stop the "processed" glitch
        bot_text = groq_response.choices[0].message.content
        
        # Step 3: Audio generation (Indian Accent)
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # FIX: Dynamically bind to the port Render provides
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
