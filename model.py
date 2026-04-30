import os
import re
import requests
import io
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

def extract_text_from_pdf_url(pdf_url):
    print(f"📥 Downloading PDF: {pdf_url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # 🛡️ SECURITY: Add Vercel Bypass Secret if configured
    bypass_secret = os.getenv('VERCEL_BYPASS_SECRET')
    if bypass_secret:
        headers["x-vercel-protection-bypass"] = bypass_secret
        headers["x-vercel-set-bypass-cookie"] = "samesite-none; secure"
        print("🔑 Using Vercel Bypass Secret for authentication.")

    response = requests.get(pdf_url, headers=headers, timeout=20)
    response.raise_for_status()
    
    with io.BytesIO(response.content) as f:
        reader = PdfReader(f)
        text = ""
        # We process first 5 pages for a quick high-quality summary
        max_pages = min(5, len(reader.pages))
        for i in range(max_pages):
            text += reader.pages[i].extract_text() + "\n"
    return text

def simple_sentence_splitter(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, max_chars=2000): # Increased chunk size for better context
    sentences = simple_sentence_splitter(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk)
    return chunks

def summarize_single_chunk(chunk, api_token=None):
    token = api_token or os.getenv('HF_TOKEN')
    # Using Llama 3.1 for stability and high-quality scholarly summaries
    client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=token)
    
    messages = [
        {"role": "system", "content": "You are a scholarly assistant. Provide a single concise paragraph summarizing the research findings and objective of this paper."},
        {"role": "user", "content": chunk}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}")
        return "Summary generation paused. Please try again."
