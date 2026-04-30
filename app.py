# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, List
import os
import re
import model
from huggingface_hub import InferenceClient

load_dotenv()

app = FastAPI()
# SECURITY PROTOCOL: Strict CORS Enforcement
# Restricted to official WISDOM domains to prevent cross-site request forgery
ALLOWED_ORIGINS = [
    "https://prithwishganguli.in",
    "https://wisdom-new-taupe.vercel.app",
    "https://www.wisdomj.in",
    "https://wisdomj.in",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []
    papers_context: Optional[str] = None
    site_context: Optional[str] = None

class PrepareRequest(BaseModel):
    pdf_url: Optional[str] = None
    text: Optional[str] = None

class ChunkRequest(BaseModel):
    chunk: str

def clean_text(text):
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.get("/")
def home(): return {"status": "Wisdom GPT Expert Engine Online"}

@app.post("/expand-query")
def expand_query(data: ChatRequest):
    try:
        api_token = os.getenv("HF_TOKEN")
        client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=api_token)
        
        prompt = (
            f"User Query: '{data.message}'\n\n"
            "Expand this query into 5 key scholarly search terms/concepts. "
            "Return only a comma-separated list of terms. Do not explain."
        )

        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        
        return {"expanded": response.choices[0].message.content.strip()}
        
    except Exception as e:
        return {"expanded": data.message}

@app.post("/semantic-search")
def semantic_search(data: ChatRequest):
    try:
        api_token = os.getenv("HF_TOKEN")
        client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=api_token)
        
        prompt = (
            f"USER SEARCH INTENT: '{data.message}'\n\n"
            "Evaluate these papers for DEEP CONCEPTUAL RELEVANCE. "
            "Ignore papers that only mention keywords in passing. Only keep papers where the core topic matches the user intent.\n"
            "Return format: ID|1-sentence relevance explanation (max 20 words)\n\n"
            f"PAPERS TO ANALYZE:\n{data.papers_context}"
        )

        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            max_tokens=800,
            temperature=0.1
        )
        
        return {"insights": response.choices[0].message.content.strip()}
        
    except Exception as e:
        print(f"❌ SEMANTIC ERROR: {str(e)}")
        return {"insights": ""}

@app.post("/chat")
def chat(data: ChatRequest):
    try:
        api_token = os.getenv("HF_TOKEN")
        client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=api_token)
        
        # --- THE EXPERT KNOWLEDGE BASE ---
        system_instr = (
            "You are Wisdom GPT, the official AI Expert for the WISDOM Journal. "
            "You provide high-level, accurate, and scholarly assistance by cross-referencing our static site info and our live MongoDB research repository.\n\n"
            "WORLD 1: OFFICIAL SITE RECORDS (STATIC)\n"
            f"{data.site_context or 'N/A'}\n\n"
            "WORLD 2: RESEARCH REPOSITORY (MONGODB)\n"
            f"{data.papers_context or 'No papers currently matched.'}\n\n"
            "EXPERT GUIDELINES:\n"
            "1. If asked about journal details (Editor, Submission, Contact), use WORLD 1.\n"
            "2. If asked about scholarly topics (Cybercrime, Law, Finance), synthesize findings from WORLD 2.\n"
            "3. Be conversational like ChatGPT but grounded like a Research Editor.\n"
            "4. NEVER guess. If info isn't in either world, offer to help the user find it on the main contact page."
        )

        messages = [{"role": "system", "content": system_instr}]
        
        # --- ROBUST ROLE ALTERNATION FILTER ---
        if data.history:
            last_role = "system"
            for msg in data.history[-4:]: # Only keep last 4 for efficiency
                if msg.role == last_role: continue
                if last_role == "system" and msg.role == "assistant": continue
                messages.append({"role": msg.role, "content": msg.content})
                last_role = msg.role
        
        # Add current query
        if messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n\nNEW QUERY: {data.message}"
        else:
            messages.append({"role": "user", "content": data.message})
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=300, # Concise but thorough
            temperature=0.3
        )
        
        return {"response": response.choices[0].message.content.strip()}
        
    except Exception as e:
        print(f"❌ ENGINE ERROR: {str(e)}")
        return {"response": "I am currently re-indexing the database records. Please try your request again in a moment."}

@app.post("/prepare")
def prepare(data: PrepareRequest):
    try:
        text = model.extract_text_from_pdf_url(data.pdf_url) if data.pdf_url else (data.text or "")
        return {"chunks": model.chunk_text(text, max_chars=4000)[:1]}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-chunk")
def summarize_chunk(data: ChunkRequest):
    try:
        summary = model.summarize_single_chunk(data.chunk, os.getenv("HF_TOKEN"))
        return {"summary": clean_text(summary)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
