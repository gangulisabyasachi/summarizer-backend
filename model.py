
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
    response = requests.get(pdf_url, timeout=15)
    response.raise_for_status()
    
    with io.BytesIO(response.content) as f:
        reader = PdfReader(f)
        text = ""
        max_pages = min(5, len(reader.pages))
        for i in range(max_pages):
            text += reader.pages[i].extract_text() + "\n"
    return text

def simple_sentence_splitter(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, max_chars=1200):
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
    client = InferenceClient(model="sshleifer/distilbart-cnn-12-6", token=token)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = client.summarization(chunk)
            return result if isinstance(result, str) else result.summary_text
        except Exception as e:
            if "504" in str(e) and attempt < max_retries - 1:
                import time
                time.sleep(2)
                continue
            raise e
