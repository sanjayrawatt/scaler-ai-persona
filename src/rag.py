import os
import json
import numpy as np
import fitz  # PyMuPDF for reading PDFs
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Setup Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# Initialize Gemini client for Embeddings
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def get_embeddings(texts):
    """Get embeddings from Gemini."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    return [e.values for e in result.embeddings]

def build_index():
    print("Loading documents...")
    all_chunks = []
    all_metadata = []
    
    # 1. Load Resume
    resume_path = os.path.join(DATA_DIR, "resume.pdf")
    if os.path.exists(resume_path):
        resume_text = extract_text_from_pdf(resume_path)
        chunks = chunk_text(resume_text)
        all_chunks.extend(chunks)
        all_metadata.extend([{"source": "resume"}] * len(chunks))
        print(f"Loaded resume: {len(chunks)} chunks.")
    
    # 2. Load GitHub Context
    github_path = os.path.join(DATA_DIR, "github_context.txt")
    if os.path.exists(github_path):
        with open(github_path, "r", encoding="utf-8") as f:
            github_text = f.read()
        chunks = chunk_text(github_text)
        all_chunks.extend(chunks)
        all_metadata.extend([{"source": "github"}] * len(chunks))
        print(f"Loaded github context: {len(chunks)} chunks.")
        
    if not all_chunks:
        print("No documents found in data directory.")
        return None

    # Build embeddings
    print(f"Building Gemini embeddings for {len(all_chunks)} chunks...")
    
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        embeddings = get_embeddings(batch)
        all_embeddings.extend(embeddings)
        print(f"  Processed {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunks")
    
    # Save index
    os.makedirs(DB_DIR, exist_ok=True)
    index_data = {
        "chunks": all_chunks,
        "metadata": all_metadata,
        "embeddings": all_embeddings
    }
    
    index_path = os.path.join(DB_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f)
    
    print(f"Index built and saved to {index_path} successfully!")
    return index_data

if __name__ == "__main__":
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set in .env")
    else:
        print(f"Using Gemini API for embeddings.")
        build_index()
