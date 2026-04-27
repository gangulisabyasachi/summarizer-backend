# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import os
from model import generate_summary

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationParams(BaseModel):
    max_new_tokens: Optional[int] = 128
    num_beams: Optional[int] = 4

class InputText(BaseModel):
    text: str
    parameters: Optional[GenerationParams] = None

def trim_to_last_fullstop(text):
    last_dot_index = text.rfind('.')
    if last_dot_index != -1:
        return text[:last_dot_index + 1]
    return text

@app.get("/")
def home():
    return {"message": "Wisdom Summarizer API is running"}

@app.post("/predict")
def predict(data: InputText):
    try:
        api_token = os.getenv("HF_TOKEN")
        if not api_token:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set")
        
        summary = generate_summary(
            text=data.text,
            api_token=api_token
        )
        
        trimmed_summary = trim_to_last_fullstop(summary)
        return {"summary": trimmed_summary}

    except Exception as e:
        print("error" , e)
        raise HTTPException(status_code=500, detail=str(e))
