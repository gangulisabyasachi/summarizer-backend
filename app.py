# server.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from typing import Optional  # Add this import
import torch
import os
from model import generate_summary  # Ensure this is present
torch.set_num_threads(os.cpu_count())

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS setup - Allowing all for local network testing
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("./model/PnHLayman")
# model = AutoModelForSeq2SeqLM.from_pretrained("./model/PnHLayman")

# # Set device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

class GenerationParams(BaseModel):
    max_new_tokens: Optional[int] = 128
    num_beams: Optional[int] = 4
    length_penalty: Optional[float] = 0.8
    early_stopping: Optional[bool] = False
    temperature: Optional[float] = None  # Optional params
    no_repeat_ngram_size: Optional[int] = None

class InputText(BaseModel):
    text: str
    parameters: Optional[GenerationParams] = None  # Nested params
    

def trim_to_last_fullstop(text):
    last_dot_index = text.rfind('.')
    if last_dot_index != -1:
        return text[:last_dot_index + 1]
    return text  # Return original if no full stop found

@app.get("/")
def home():
    return {"message": "FastAPI server running"}

@app.post("/predict")
def predict(data: InputText):
    try:
        # Get token from environment
        api_token = os.getenv("HF_TOKEN")
        if not api_token:
            raise HTTPException(status_code=500, detail="HF_TOKEN not set in environment")
        
         # Use provided parameters or defaults
        params = data.parameters or GenerationParams()
        summary_params = {
            "max_length": 130,
            "min_length": 30,
        }
        
        # Generate summary via Public API
        summary = generate_summary(
            text=data.text,
            api_token=api_token,
            summary_params=summary_params
        )
        
        trimmed_summary = trim_to_last_fullstop(summary)
        return {"summary": trimmed_summary}

    except Exception as e:
        print("error" , e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: Please try again"
        )
