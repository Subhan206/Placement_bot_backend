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

# --- GUARDRAIL 1: THE STRICT SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are ARIA, the official Placement Intelligence Voice AI for MIT Bengaluru. 
Your primary goal is to assist students with campus and placement queries.

CRITICAL RULES FOR VOICE OUTPUT:
1. EXTREME BREVITY: Limit every response to a maximum of 1 to 2 short sentences. Do not exceed 25 words. 
2. NO FORMATTING: Do not use bullet points, bold text, markdown, or special characters. Use plain English words only (e.g., say "percent" instead of "%").
3. RAG DEPENDENCE: Only use provided context. If the answer is not in the context, say exactly: "I don't have that specific data right now. Please check the official placement portal."
4. NO FILLER: Never use introductory filler. Answer the core question immediately.
"""

# --- GUARDRAIL 2: INTENT DETECTION & QUERY EXPANSION ---
EXPANSIONS = {
    "cs faculty": "Computer Science faculty members list MIT Bengaluru names professors",
    "cse faculty": "Computer Science Engineering faculty members MIT Bengaluru",
    "hostel fee": "hostel room fee structure charges 2025 MIT Bengaluru annual",
    "hostel room": "hostel room types single double AC non-AC fee MIT Bengaluru",
}

def detect_query_intent(query: str):
    q = query.lower()
    if any(w in q for w in ["faculty", "professor", "teacher", "staff", "hod"]):
        return "faculty"
    if any(w in q for w in ["fee", "cost", "payment", "hostel charges"]):
        return "finance"
    return None

def expand_query(query: str) -> str:
    q_lower = query.lower()
    for shorthand, expansion in EXPANSIONS.items():
        if shorthand in q_lower: return expansion
    return query

# --- TM2 PINE CONE RAG SETUP ---
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
    category = detect_query_intent(user_query)
    expanded_text = expand_query(user_query)
    
    try:
        # Using Pinecone's server-side embedding (assumes index is set up for it)
        # If your index doesn't support 'model' param, you'll need the inference API call here
        results = index.query(
            vector=[0.0] * 384, # Placeholder: Replace with actual inference if not using Pinecone's integrated embedder
            top_k=top_k,
            include_metadata=True,
            filter={"category": {"$eq": category}} if category else None
        )
        
        context_parts = []
        seen_texts = set()
        for m in results.get("matches", []):
            text = m["metadata"].get("text", "")
            if text and text[:100] not in seen_texts:
                context_parts.append(text)
                seen_texts.add(text[:100])
                
        return "\n\n".join(context_parts) if context_parts else "No specific context found."
    except Exception as e:
        print(f"RAG Error: {e}")
        return "MIT Bengaluru placement and campus information."

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Retrieval with Guardrails
        context = search_campus_data(request.query)

        # 2. Generation with Strict Tone
        groq_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {request.query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2, # Lowered temperature for higher strictness
        )
        
        bot_text = groq_response.choices[0].message.content
        
        # 3. Audio generation (Indian Accent)
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
