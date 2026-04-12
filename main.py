import base64
import os
from gtts import gTTS
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

load_dotenv()

app = FastAPI()

# 1. THE CORS FIX: Allow your Vercel frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API Clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Define the incoming request structure
class ChatRequest(BaseModel):
    query: str

# 2. THE GUARDRAIL PROMPT
SYSTEM_PROMPT = """
You are the official, highly professional AI Assistant for Manipal Institute of Technology (MIT) Bengaluru.

CORE DIRECTIVES:

Identity: You represent MIT Bengaluru. Be welcoming, concise, and helpful.

Strict Boundary: You ONLY provide information regarding the Bengaluru campus. If a user asks about the Manipal main campus, politely clarify that you only have access to Bengaluru campus data.

Factual Grounding: You will be provided with 'Context' retrieved from the official database. You must base your answers STRICTLY on this context. If the context does not contain the answer, say: 'I don't have that specific information right now, please check the official website.' Do not invent data.

Audio Optimization: Your output will be spoken aloud by a text-to-speech engine. Your responses MUST be short (under 3 sentences).

Formatting Ban: DO NOT use bullet points, asterisks, bolding, numbered lists, or emojis. Write in pure, conversational prose.
"""

# --- TM2 PINE CONE RAG SETUP ---
_model = None
_index = None

def _get_resources():
    global _model, _index
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index("mit-bengaluru")
    return _model, _index

def detect_category(query: str) -> str | None:
    q = query.lower()
    if any(w in q for w in ["course", "program", "btech", "mtech", "degree", "admission", "eligibility", "curriculum", "specialization"]): return "programs"
    if any(w in q for w in ["hostel", "accommodation", "room", "mess", "dining", "stay", "dormitory", "warden"]): return "hostel"
    if any(w in q for w in ["fee", "scholarship", "loan", "tuition", "cost", "payment", "financial"]): return "admissions"
    if any(w in q for w in ["faculty", "professor", "teacher", "who teaches", "hod", "head of department", "staff"]): return "faculty"
    if any(w in q for w in ["event", "news", "conference", "workshop", "seminar", "calendar"]): return "news"
    return None 

def search_campus_data(user_query: str, top_k: int = 3) -> str:
    model, index = _get_resources()
    query_embedding = model.encode([user_query]).tolist()
    category = detect_category(user_query)
    
    filter_dict = {"category": {"$eq": category}} if category else None
    
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=filter_dict)
    
    if not results["matches"] and filter_dict:
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
    if not results["matches"]:
        return "No relevant information found in the MIT Bengaluru knowledge base."
        
    context_parts = []
    for match in results["matches"]:
        text = match["metadata"].get("text", "")
        context_parts.append(text)
        
    return "\n".join(context_parts)
# -------------------------------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Step 1: Get Context from TM2's RAG database
        context = search_campus_data(request.query)
        
        # Step 2: Generate Text with Groq (Llama-3-70B for instant speed)
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
        
       # Step 3: Generate Audio with TTS (Using free gTTS to bypass billing)
        # Hack: using tld='co.in' gives the bot an Indian accent, perfect for MIT Bengaluru!
        tts = gTTS(text=bot_text, lang='en', tld='co.in') 
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        
        # Step 4: THE MAGIC TRICK - Convert audio to Base64 to bypass CORS
        audio_base64 = base64.b64encode(audio_fp.getvalue()).decode('utf-8')
        
        # Step 5: Return the packaged JSON
        return {
            "text": bot_text,
            "audio_base64": audio_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)