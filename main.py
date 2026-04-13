import base64
import os
import io
import re
import edge_tts 
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# IMPORT TM2's BRILLIANT SEARCH MODULE!
from search import search_campus_data, _get_resources as warmup_search

load_dotenv()

resources = {}

# --- FAST BOOT LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Pinecone inside TM2's file so it's ready instantly
    warmup_search()
    
    # Initialize Groq
    resources["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("ARIA: API & Search modules warmed up and ready.")
    yield
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

# --- THE STRICT BOUNCER SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are ARIA, the official Placement Intelligence Voice AI for MIT Bengaluru. 

CRITICAL SECURITY GUARDRAILS:
1. BOUNDARY ENFORCEMENT: You are strictly forbidden from answering general knowledge questions, math problems, coding tutorials, or anything outside of MIT Bengaluru campus, placements, and courses.
2. OUT-OF-BOUNDS PROTOCOL: If the user asks an unrelated question, you MUST reject it by saying exactly: "I am specifically trained for MIT Bengaluru campus queries. I cannot answer that."
3. RAG STRICTNESS: If the provided context says "No relevant information found", DO NOT use your internal knowledge to guess. Say exactly: "I don't have that specific data right now."
4. EXTREME BREVITY: Limit all responses to 1 or 2 short sentences (max 25 words). 
5. NO FORMATTING: Plain text only. No asterisks, bolding, or symbols.
"""


# --- TM2 PINE CONE RAG SETUP (1024d INFERENCE UPGRADED) ---
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
    pc = resources.get("pc")
    index = resources.get("index")

    if not pc or not index:
        return "System initializing."

    expanded = expand_query(user_query)
    
    # NEW FIX: Use Pinecone Inference API to get the correct 1024d vector
    try:
        embedding_response = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=[expanded],
            parameters={"input_type": "query"}
        )
        # FIX: Access the values from the first item in the inference list
        query_vec = embedding_response[0].values
    except Exception as e:
        print(f"Embedding failed: {e}")
        return "No relevant information found in the MIT Bengaluru knowledge base."

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
        # Step 1: Call TM2's highly advanced search module
        raw_context = search_campus_data(request.query)

        # Step 1.5: THE VOICE FIX (Strip out TM2's [Cosine: %] tracking tags so AI doesn't read them)
        context = re.sub(r"\[.*?\]", "", raw_context).strip()

        # Step 2: THE FIREWALL
        if "No relevant information found" in context or "No results found" in context:
            bot_text = "I don't have that specific data in my current database. Please check the official placement portal."
            
            communicate = edge_tts.Communicate(bot_text, "en-IN-NeerjaNeural")
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return {"text": bot_text, "audio_base64": audio_base64}

        # Step 3: MAIN GENERATION
        groq_client = resources.get("groq")
        groq_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {request.query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        

        # Fixed typo here!
        # FIX: Added [0] to access the actual text content
        bot_text = groq_response.choices[0].message.content

        
        # Step 4: AUDIO GENERATION
        communicate = edge_tts.Communicate(bot_text, "en-IN-NeerjaNeural")
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {"text": bot_text, "audio_base64": audio_base64}

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
