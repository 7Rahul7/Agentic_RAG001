import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from ddgs import DDGS
import re
from dotenv import load_dotenv

load_dotenv()

# === 1. Authenticate Gemini (load API key securely) ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # Use free model

# === 2. Embedding model and FAISS index ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
documents = []     # Chunked text content
doc_sources = []   # Metadata: file + page

