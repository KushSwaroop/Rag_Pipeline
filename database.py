import lancedb
from pypdf import PdfReader
import pdfplumber
import httpx
import os
import csv
import time
import re
from pathlib import Path

# Initialize local LanceDB
db = lancedb.connect("./rag_lancedb")
TABLE_NAME = "documents"
DATA_DIR = Path(__file__).parent / "DATA"

MAX_CSV_ROWS = 5000
MAX_INGEST_SIZE_MB = 50

# -----------------------------------------------------------------
# EMBEDDINGS (BATCH API)
# -----------------------------------------------------------------
def get_embedding(text: str, client: httpx.Client = None) -> list:
    url = "http://127.0.0.1:11434/api/embeddings"
    json_data = {"model": "nomic-embed-text", "prompt": text}
    
    for attempt in range(5):
        try:
            if client:
                response = client.post(url, json=json_data, timeout=30.0)
            else:
                with httpx.Client(timeout=30.0) as c:
                    response = c.post(url, json=json_data)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            if attempt < 4:
                time.sleep(1.0 * (attempt + 1))
                continue
            else:
                print(f"Embedding error: {e}")
                return []

def get_embeddings_batch(texts: list, client: httpx.Client = None) -> list:
    """Calls local Ollama /api/embed to process up to 100 texts at once using GPU."""
    url = "http://127.0.0.1:11434/api/embed"
    inputs = [f"search_document: {t}" for t in texts]
    json_data = {"model": "nomic-embed-text", "input": inputs}
    
    for attempt in range(5):
        try:
            if client:
                response = client.post(url, json=json_data, timeout=300.0)
            else:
                with httpx.Client(timeout=300.0) as c:
                    response = c.post(url, json=json_data)
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            if attempt < 4:
                print(f"Batch embed retry {attempt+1}... {e}")
                time.sleep(2.0 * (attempt + 1))
                continue
            else:
                print(f"Batch embedding error: {e}")
                return []

# -----------------------------------------------------------------
# CHUNKING LOGIC
# -----------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> list:
    """Generic fallback string chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_gov_csv(file_path: str, filename: str) -> list:
    """Strategy: row-group by topic, 50-100 rows, overlap 10 rows."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers: return []
            
            rows = []
            for row in reader:
                if len(rows) > MAX_CSV_ROWS: break
                rows.append(row)
                
            chunk_size = 80  # Between 50-100
            overlap = 10
            start = 0
            while start < len(rows):
                end = start + chunk_size
                batch_rows = rows[start:end]
                
                chunk_text_str = f"Population data for Olpe, Germany (Source: {filename}):\n"
                for r in batch_rows:
                    pairs = [f"{h}: {v}" for h, v in zip(headers, r) if str(v).strip()]
                    chunk_text_str += " | ".join(pairs) + "\n"
                
                chunks.append(chunk_text_str)
                start += chunk_size - overlap
    except Exception as e:
        print(f"  CSV error: {e}")
    return chunks

def chunk_mixed_gov_pdf(file_path: str, filename: str) -> list:
    """Strategy: Extract tables to structured text markdown, 1 table per chunk."""
    chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            # Safety: skip table extraction on massive documents
            skip_tables = len(pdf.pages) > 100
            if skip_tables:
                print(f"    [WARN] Document has {len(pdf.pages)} pages. Skipping slow table extraction.")
                
            for page in pdf.pages:
                if not skip_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        md = f"Data table from city of Olpe (Source: {filename}):\n"
                        for i, row in enumerate(table):
                            clean_row = [str(c).replace('\n', ' ') if c else '' for c in row]
                            md += "| " + " | ".join(clean_row) + " |\n"
                            if i == 0:
                                md += "|" + "|".join(["---"] * len(clean_row)) + "|\n"
                        chunks.append(md)
                
                text = page.extract_text()
                if text and len(text) > 50:
                    chunks.extend(chunk_text(text, 1000, 200))
    except Exception as e:
        print(f"  PDF Table extraction error: {e}")
    return chunks

def chunk_presentation(file_path: str, filename: str) -> list:
    """Strategy: Slide-based, whole slide = 1 chunk, no overlap."""
    chunks = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 10:
                chunks.append(f"[Slide {i+1} | {filename}]\n{text}")
    except Exception as e:
        print(f"  Presentation extraction error: {e}")
    return chunks

def chunk_report(file_path: str, filename: str) -> list:
    """Strategy: Section-based, 1-3 paragraphs per section."""
    chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text: full_text += text + "\n\n"
                    
            paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 20]
            
            start = 0
            while start < len(paragraphs):
                group = paragraphs[start:start+3]
                chunks.append(f"Section from {filename}:\n" + "\n\n".join(group))
                start += 2 # Overlap by 1 paragraph
    except Exception as e:
        print(f"  Report extraction error: {e}")
    return chunks

def chunk_paper(file_path: str, filename: str) -> list:
    """Strategy: Semantic + structural. Preserve abstract, split by headers."""
    chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text: full_text += text + "\n"
                    
            # Keep abstract as single chunk
            abstract_match = re.search(r'(?i)\babstract\b.*?(?=\n1\.?\s+Introduction|\n\d+\.\s)', full_text, re.DOTALL)
            if abstract_match:
                chunks.append(f"[Abstract] {filename}:\n" + abstract_match.group(0).strip())
                full_text = full_text.replace(abstract_match.group(0), "")
                
            # Split sections using numbering e.g. "1. Introduction"
            sections = re.split(r'\n(?=\d+\.\s+[A-Z])', full_text)
            for sec in sections:
                if len(sec.strip()) < 50: continue
                # 800-1200 tokens roughly translates to ~4000 characters. 
                # Reduced to 2000 chars safely stay under local model token limit.
                if len(sec) > 2500:
                    subchunks = chunk_text(sec, chunk_size=2000, overlap=500)
                    chunks.extend([f"[Excerpt] {filename}:\n" + sc for sc in subchunks])
                else:
                    chunks.append(f"[Section] {filename}:\n" + sec.strip())
    except Exception as e:
        print(f"  Paper extraction error: {e}")
    return chunks

# -----------------------------------------------------------------
# ROUTING AND INGESTION
# -----------------------------------------------------------------
def extract_and_chunk(file_path: str, filename: str) -> list:
    ext = Path(filename).suffix.lower()
    fname_lower = filename.lower()
    
    if ext == ".csv" and "govdata" in fname_lower:
        return chunk_gov_csv(file_path, filename)
    elif ext == ".pdf" and "govdata" in fname_lower:
        return chunk_mixed_gov_pdf(file_path, filename)
    elif ext == ".pdf" and ("lecture" in fname_lower or "lab" in fname_lower or "slide" in fname_lower):
        return chunk_presentation(file_path, filename)
    elif ext == ".pdf" and ("project" in fname_lower or "system" in fname_lower):
        return chunk_report(file_path, filename)
    elif ext == ".pdf":
        return chunk_paper(file_path, filename)
    elif ext == ".csv":
        # Fallback csv read
        return chunk_gov_csv(file_path, filename) # generic fallback for csv
    else:
        # Fallback text
        with open(file_path, "r", errors="ignore") as f:
            return chunk_text(f.read())

def get_ingested_sources() -> set:
    if TABLE_NAME not in db.table_names():
        return set()
    table = db.open_table(TABLE_NAME)
    try:
        results = table.search().limit(10000).to_list()
        return set(r.get("source") for r in results if "source" in r)
    except Exception:
        return set()

def ingest_all(data_dir: Path = DATA_DIR, force: bool = False):
    supported_extensions = {".pdf", ".csv", ".txt"}
    
    files = sorted([f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions])
    
    if not files:
        print(f"No supported files found in {data_dir}")
        return
        
    already_ingested = set() if force else get_ingested_sources()
    
    print("=" * 60)
    print("  LANCEDB BATCH INGESTION (FILE-BY-FILE STREAMING)")
    print("=" * 60)
    print(f"  Data directory : {data_dir.resolve()}")
    print(f"  Files found    : {len(files)}")
    print(f"  Already in DB  : {len(already_ingested)}")
    print("=" * 60)
    
    total_inserted = 0
    start_time = time.time()
    
    with httpx.Client(timeout=300.0) as client:
        for filepath in files:
            filename = filepath.name
            
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > MAX_INGEST_SIZE_MB:
                print(f"\n[SKIP] File too large ({size_mb:.1f} MB > {MAX_INGEST_SIZE_MB} MB): {filename}")
                continue
                
            if filename in already_ingested:
                print(f"\n[SKIP] Already ingested: {filename}")
                continue
                
            print(f"\n[PROCESSING] Extracting & chunking: {filename} ({size_mb:.1f} MB)...")
            chunks = extract_and_chunk(str(filepath), filename)
            if not chunks:
                print(f"  [WARN] No chunks extracted.")
                continue
                
            print(f"  Extracted {len(chunks)} chunks. GPU embedding sequentially (extremely fast over local IP)...")
            
            # Fast sequential embedding (avoids Ollama parallel/batch panics and 500 errors)
            file_inserted = 0
            successful_data = []
            
            for i, chunk_text_str in enumerate(chunks):
                if i > 0 and i % 50 == 0:
                    print(f"    -> Embedded {i}/{len(chunks)}...", end="\r", flush=True)
                    
                emb = get_embedding(chunk_text_str, client=client)
                if not emb:
                    print(f"\n  [ERROR] Failed to embed chunk {i}. Stopping file.")
                    break
                    
                successful_data.append({
                    "id": f"{filename}_chunk_{i}",
                    "vector": emb,
                    "text": chunk_text_str,
                    "source": filename
                })
                
            if successful_data:
                if TABLE_NAME in db.table_names():
                    table = db.open_table(TABLE_NAME)
                    table.add(successful_data)
                else:
                    db.create_table(TABLE_NAME, data=successful_data)
                    
                file_inserted = len(successful_data)
                print(f"\n  [DONE] {file_inserted} chunks successfully inserted for {filename}.")
                total_inserted += file_inserted
            
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Total chunks inserted: {total_inserted}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-ingest files")
    args = parser.parse_args()
    
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        exit(1)
        
    ingest_all(force=args.force)