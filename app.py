# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, List
import os
import re
import model

load_dotenv()

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PrepareRequest(BaseModel):
    pdf_url: Optional[str] = None
    text: Optional[str] = None

class ChunkRequest(BaseModel):
    chunk: str

def clean_text(text):
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    # Ensure space after punctuation (if not followed by space or end of string)
    text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.get("/")
def home():
    return {"message": "Wisdom Multi-Stage API is running"}

@app.post("/prepare")
def prepare(data: PrepareRequest):
    try:
        text = ""
        if data.pdf_url:
            text = model.extract_text_from_pdf_url(data.pdf_url)
        else:
            text = data.text or ""
            
        if not text.strip():
            raise ValueError("No text found to summarize")
            
        chunks = model.chunk_text(text)
        return {"chunks": chunks[:4]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-chunk")
def summarize_chunk(data: ChunkRequest):
    try:
        api_token = os.getenv("HF_TOKEN")
        summary = model.summarize_single_chunk(data.chunk, api_token)
        # Apply the punctuation fix
        cleaned_summary = clean_text(summary)
        return {"summary": cleaned_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
