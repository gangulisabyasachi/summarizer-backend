
import os
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

def simple_sentence_splitter(text):
    # Basic sentence splitting without heavy libraries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, max_chars=3000):
    # Use larger chunks to reduce number of API calls (faster)
    sentences = simple_sentence_splitter(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def generate_summary(text, api_url=None, api_token=None, summary_params=None):
    """
    Summarizes input text using the FAST Public API model.
    """
    token = api_token or os.getenv('HF_TOKEN')
    if not token:
        raise ValueError("HF_TOKEN must be provided or set as an environment variable")
    
    # Using the distilled model for 3x faster response times
    model_id = "sshleifer/distilbart-cnn-12-6"
    client = InferenceClient(model=model_id, token=token)

    if summary_params is None:
        summary_params = {
            "max_length": 130,
            "min_length": 30,
            "do_sample": False
        }

    summaries = []
    chunks = chunk_text(text)
    
    for idx, chunk in enumerate(chunks):
        print(f"\n📤 Sending chunk {idx+1}/{len(chunks)} to Public API...")
        
        # Retry logic for 504 timeouts
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = client.summarization(chunk)
                summary_text = result if isinstance(result, str) else result.summary_text
                summaries.append(summary_text)
                print(f"✅ Chunk {idx+1} summarized.")
                break # Success!
            except Exception as e:
                if "504" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Timeout on attempt {attempt+1}. Retrying in 2s...")
                    import time
                    time.sleep(2)
                    continue
                print(f"❌ Error on chunk {idx+1}: {e}")
                raise RuntimeError(f"Public API summarization failed: {str(e)}")

    return " ".join(summaries)
