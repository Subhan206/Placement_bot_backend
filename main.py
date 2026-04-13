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
SYSTEM_PROMPT = """You are ARIA, the official conversational Voice AI for MIT Bengaluru placements and campus.

YOUR PERSONALITY & VOICE:
1. Speak naturally, warmly, and professionally, like a highly knowledgeable student advisor.
2. SYNTHESIZE THE DATA: Do not just read the context back like a robot. Use the provided context to write a smooth, natural spoken answer.
3. CONVERSATIONAL LENGTH: Aim for 2 to 4 short sentences. Give a complete, helpful answer, but keep it brief enough for voice output. 
4. NO FORMATTING: You are generating scripts for a text-to-speech engine. Plain text only. Never use asterisks, bullets, brackets, or special symbols. 

YOUR SECURITY GUARDRAILS:
1. STRICT DOMAIN: You only answer questions about MIT Bengaluru, its campus, faculty, courses, hostels, and placements.
2. OFF-TOPIC REJECTION: If a user asks about general knowledge, coding, math, or anything outside your domain, politely refuse: "I'm sorry, but I am specifically trained to help with MIT Bengaluru campus queries."
3. NO HALLUCINATIONS: If the provided context does not contain the answer, do not guess using your internal knowledge. Say: "I don't have that exact information on hand right now, please check the official portal."
"""

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Step 1: Call TM2's highly advanced search module
        raw_context = search_campus_data(request.query)

        # Step 1.5: THE VOICE FIX (Strip out TM2's [Cosine: %] tracking tags so AI doesn't read them)
        context = re.sub(r"\[.*?\]", "", raw_context).strip()
        # Step 2: THE FIREWALL
        if not context or "No relevant information found" in context:
            bot_text = "I don't have that exact information on hand right now, please check the official portal."
         
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
        
        bot_text = groq_response.choices.message.content
        
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
