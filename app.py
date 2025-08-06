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

# === 3. Load and chunk PDFs ===
def load_pdfs_from_directory(data_dir="./data", chunk_size=200):
    global documents, doc_sources
    print(f"\n Loading PDFs from '{data_dir}'...\n")

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            doc = fitz.open(filepath)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                documents.extend(chunks)
                doc_sources.extend([f"{filename} - page {page_num+1}"] * len(chunks))

    if len(documents) == 0:
        print(" No PDF content found in './data'! Please add PDFs before running.")
        exit(1)

    print(f"Loaded {len(documents)} chunks from PDFs.\n")

    vectors = embedder.encode(documents)
    index.add(np.array(vectors))


# === 4. Retrieve context from FAISS ===
def retrieve_context(query, k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [(documents[i], doc_sources[i]) for i in I[0]]

# === 5. DuckDuckGo Search ===
def duckduckgo_search(query):
    if not query.strip():
        return "No result found (empty query)."
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=1))
        if results:
            return results[0].get("body", "No result found.")
        else:
            return "No result found."



# === 6. Validate PDF answer quality ===
def validate_answer_sufficiency(answer, query):
    validation_prompt = f"""
You are a helpful assistant evaluating your previous answer.

Question: {query}

Answer:
{answer}

Is this answer sufficient, accurate, and complete to fully answer the question? Respond with one word only: YES or NO.
"""
    validation_response = model.generate_content(validation_prompt).text.strip().upper()
    print(f"\n Gemini's validation response for PDF answer: {validation_response}\n")
    return validation_response == "YES"








# Why is RAG more accurate than traditional LLMs?
# What are the components of a RAG pipeline?

# Who is the CEO of Hugging Face in 2025?
# What are the top open-source LLMs in 2025?