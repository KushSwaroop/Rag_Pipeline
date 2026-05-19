import lancedb
from pypdf import PdfReader
import httpx
import os

# Initialize local LanceDB (This creates a 'rag_lancedb' folder in your project)
db = lancedb.connect("./rag_lancedb")
TABLE_NAME = "documents"

def get_embedding(text: str) -> list:
    """Calls local Ollama to convert text into a searchable vector array."""
    try:
        response = httpx.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Slices a massive text wall into overlapping readable chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_pdf(file_path: str, filename: str) -> int:
    """Reads a PDF, chunks it, and saves it to LanceDB."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
            
    chunks = chunk_text(full_text)
    if not chunks:
        return 0
    
    # Prepare data array for LanceDB (must contain a 'vector' key)
    data = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding: # Ensure the embedding generation didn't fail
            data.append({
                "id": f"{filename}_chunk_{i}",
                "vector": embedding,
                "text": chunk,
                "source": filename
            })
            
    if not data:
        return 0

    # Insert into LanceDB
    if TABLE_NAME in db.table_names():
        # Open existing table and append new data
        table = db.open_table(TABLE_NAME)
        table.add(data)
    else:
        # Create a new table automatically inferring the schema from the data
        db.create_table(TABLE_NAME, data=data)
        
    return len(data)

def retrieve_chunks(query: str, top_k: int = 6) -> list:
    """Searches the LanceDB table for chunks that match the user's question."""
    query_embedding = get_embedding(query)
    
    if not query_embedding:
        return []

    # Check if the table even exists yet (prevents crashes on empty DB)
    if TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TABLE_NAME)
    
    # Execute LanceDB vector search
    results = table.search(query_embedding).limit(top_k).to_list()
    
    # LanceDB returns the full row (id, vector, text, source). 
    # We just need to extract the raw text to feed back to our LLM.
    return [result["text"] for result in results]