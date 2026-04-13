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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    query: str

SYSTEM_PROMPT = """You are ARIA, the official Placement Intelligence Voice AI for MIT Bengaluru. 
Limit responses to 1-2 short sentences. No markdown, no formatting. Plain text only."""

# --- TM2 PINE CONE RAG SETUP (V2 SERVERLESS + INTEGRATED INFERENCE) ---
_index = None

def _get_resources():
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # Using the new V2 Integrated Index
        _index = pc.Index("mit-bengaluru-v2")
    return _index

def search_campus_data(user_query: str, top_k: int = 3) -> str:
    index = _get_resources()
    
    # NEW V2 WAY: Search directly with text! 
    # Pinecone embeds the text for you. No local model needed.
    response = index.search_records(
        namespace="default",
        query={
            "inputs": {"text": user_query},
            "top_k": top_k
        }
    )
    
    if not response.get("result", {}).get("hits"):
        return "No relevant information found."
        
    # Extracting text from the new "hits" structure
    context_parts = []
    for hit in response["result"]["hits"]:
        # TM2 likely mapped the text to 'text' or 'chunk_text'
        txt = hit.get("fields", {}).get("text", "")
        context_parts.append(txt)

    return "\n\n---\n\n".join(context_parts)

# ---------------------------------------------
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
        )
        
        bot_text = groq_response.choices[0].message.content
        
        # Audio generation
        tts = gTTS(text=bot_text, lang='en', tld='co.in') 
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_base64 = base64.b64encode(audio_fp.getvalue()).decode('utf-8')
        
        return {"text": bot_text, "audio_base64": audio_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
