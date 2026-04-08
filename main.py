import base64
import os
from gtts import gTTS
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

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

# Dummy function for TM2's RAG pipeline (They will replace this)
def get_tm2_context(query: str) -> str:
    # TM2 will hook this up to Pinecone/ChromaDB
    return "MIT Bengaluru offers B.Tech programs in Computer Science, ECE, and IT. The campus has state-of-the-art labs and a dedicated placement cell."

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Step 1: Get Context from TM2's RAG database
        context = get_tm2_context(request.query)
        
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