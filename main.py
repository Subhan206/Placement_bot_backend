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
from sentence_transformers import SentenceTransformer

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


SYSTEM_PROMPT = """You are ARIA, the official Placement Intelligence Voice AI for MIT Bengaluru. Your primary goal is to assist students with campus and placement queries.

CRITICAL RULES FOR VOICE OUTPUT:
1. EXTREME BREVITY: You are speaking out loud. Limit every single response to a maximum of 1 to 2 short sentences. Do not exceed 25 words. 
2. NO FLUFF: Never use introductory filler like "At MIT Bengaluru, we offer..." or "I'd be happy to tell you about...". Answer the core question immediately.
3. NO FORMATTING: Do not use bullet points, bold text, markdown, asterisks, or special characters. Use plain English words only (e.g., say "percent" instead of "%").
4. RAG DEPENDENCE: Only use the provided context to answer. If the answer is not in the context, say exactly: "I don't have that specific data right now. Please check the official placement portal."
5. TONE: Professional, crisp, and directly helpful.
"""

# --- TM2 PINE CONE RAG SETUP (v2 UPGRADED) ---
_model = None
_index = None

def _get_resources():
    global _model, _index
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # Defaults to v2 if TM4 hasn't added the env var yet
        index_name = os.getenv("PINECONE_INDEX", "mit-bengaluru-v2")
        _index = pc.Index(index_name)
    return _model, _index

def detect_query_intent(query: str):
    q = query.lower()
    if any(w in q for w in ["faculty", "professor", "teacher", "staff", "who teaches", "hod", "head of department"]):
        if any(w in q for w in ["list", "all", "who are", "members", "names"]): return "faculty", "list"
        return "faculty", "profile"
    if any(w in q for w in ["hostel fee", "room fee", "fee structure", "room type", "room cost", "single ac", "double", "hostel charges", "hostel room"]): return "hostel", "table"
    if any(w in q for w in ["hostel", "accommodation", "room", "mess", "dining", "stay", "dormitory", "warden", "facilities"]): return "hostel", None
    if any(w in q for w in ["course", "program", "btech", "mtech", "degree", "curriculum", "specialization", "admission", "eligibility"]): return "programs", None
    if any(w in q for w in ["fee", "tuition", "scholarship", "loan", "cost", "payment", "financial"]): return "admissions", None
    if any(w in q for w in ["event", "news", "conference", "workshop", "seminar"]): return "news", None
    return None, None

EXPANSIONS = {
    "cs faculty":          "Computer Science faculty members list MIT Bengaluru names professors",
    "cse faculty":         "Computer Science Engineering faculty members MIT Bengaluru",
    "hostel fee":          "hostel room fee structure charges 2025 MIT Bengaluru annual",
    "hostel room":         "hostel room types single double AC non-AC fee MIT Bengaluru",
    "cs department":       "Computer Science department MIT Bengaluru faculty programs",
    "list all faculty":    "Computer Science faculty members list names MIT Bengaluru",
    "who are the faculty": "faculty members list names department MIT Bengaluru professors",
}

def expand_query(query: str) -> str:
    q_lower = query.lower()
    for shorthand, expansion in EXPANSIONS.items():
        if shorthand in q_lower: return expansion
    return query

def search_campus_data(user_query: str, top_k: int = 3) -> str:
    model, index = _get_resources()
    expanded = expand_query(user_query)
    query_vec = model.encode([expanded]).tolist()
    category, preferred_subtype = detect_query_intent(user_query)

    def run_query(filter_dict=None):
        kwargs = {"vector": query_vec, "top_k": top_k, "include_metadata": True, "namespace": "default"}
        if filter_dict: kwargs["filter"] = filter_dict
        return index.query(**kwargs)

    # Cascade Search
    matches = []
    if category and preferred_subtype:
        matches = run_query({"category": {"$eq": category}, "subtype": {"$eq": preferred_subtype}}).get("matches", [])
    if not matches and category:
        matches = run_query({"category": {"$eq": category}}).get("matches", [])
    if not matches:
        matches = run_query().get("matches", [])

    if not matches:
        return "No relevant information found in the MIT Bengaluru knowledge base."

    # De-duplicate and strip metadata for voice safety
    seen_texts = set()
    unique_matches = []
    for m in matches:
        text = m.get("metadata", {}).get("text", "")
        sig = text[:200]
        if sig not in seen_texts:
            seen_texts.add(sig)
            unique_matches.append(text) 

    return "\n\n---\n\n".join(unique_matches)

# ---------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Step 1: Get Context from TM2's RAG database
        context = search_campus_data(request.query)

        # Step 2: Call Groq
        groq_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {request.query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=150,
        )
        
        bot_text = groq_response.choices.message.content
        
        # Step 3: Audio generation
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