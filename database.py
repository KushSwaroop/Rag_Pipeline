import lancedb
from pypdf import PdfReader
import httpx
import os
import csv
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize local LanceDB (This creates a 'rag_lancedb' folder in your project)
db = lancedb.connect("./rag_lancedb")
TABLE_NAME = "documents"
DATA_DIR = Path(__file__).parent / "DATA"

# Maximum rows to ingest from large CSV/XLSX files (prevents memory issues)
MAX_CSV_ROWS = 5000
# Max file size to process (in MB) -- skip massive raw data files
MAX_INGEST_SIZE_MB = 50


def get_embedding(text: str, client: httpx.Client = None) -> list:
    """Calls local Ollama to convert text into a searchable vector array.
    
    Includes a retry mechanism to handle local server cold starts or resource queuing errors.
    """
    url = "http://127.0.0.1:11434/api/embeddings"
    json_data = {"model": "nomic-embed-text", "prompt": text}
    
    for attempt in range(5):
        try:
            if client:
                response = client.post(url, json=json_data)
            else:
                with httpx.Client(timeout=30.0) as c:
                    response = c.post(url, json=json_data)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            if attempt < 4:
                # Sleep and retry to allow Ollama server model runners to spin up
                time.sleep(1.0 * (attempt + 1))
                continue
            else:
                print(f"Embedding error: {e}")
                return []


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> list:
    """Slices a massive text wall into overlapping readable chunks.
    
    Defaults to 700 characters to keep token count safely under Ollama's 512-token batch limit 
    for nomic-embed-text (especially for non-English languages like German).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# -----------------------------------------------------------------
# TEXT EXTRACTORS (one per file type)
# -----------------------------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
    return full_text


def extract_text_from_csv(file_path: str) -> str:
    """Convert CSV rows into readable text for embedding.
    
    Strategy: convert each row into a 'Column: Value' format so the
    embeddings capture the semantic meaning of column headers + data.
    """
    text_parts = []
    try:
        # Try utf-8 first, fall back to latin-1
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding, errors="replace") as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    if not headers:
                        return ""
                    
                    row_count = 0
                    for row in reader:
                        if row_count >= MAX_CSV_ROWS:
                            text_parts.append(f"[... truncated at {MAX_CSV_ROWS} rows ...]")
                            break
                        # Format: "Column1: val1, Column2: val2, ..."
                        pairs = []
                        for h, v in zip(headers, row):
                            if v.strip():
                                pairs.append(f"{h}: {v}")
                        if pairs:
                            text_parts.append(", ".join(pairs))
                        row_count += 1
                    break  # encoding worked, stop trying
            except UnicodeDecodeError:
                continue
    except Exception as e:
        print(f"  CSV read error: {e}")
        return ""
    
    return "\n".join(text_parts)


def extract_text_from_xlsx(file_path: str) -> str:
    """Convert XLSX spreadsheet rows into readable text for embedding."""
    try:
        import openpyxl
    except ImportError:
        print("  openpyxl not installed -- skipping XLSX file")
        return ""
    
    text_parts = []
    try:
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            
            row_iter = ws.iter_rows(values_only=True)
            try:
                first_row = next(row_iter)
            except StopIteration:
                continue
            
            headers = [str(h) if h is not None else "" for h in first_row]
            text_parts.append(f"[Sheet: {sheet_name}]")
            
            row_count = 0
            for row in row_iter:
                if row_count >= MAX_CSV_ROWS:
                    text_parts.append(f"[... truncated at {MAX_CSV_ROWS} rows ...]")
                    break
                pairs = []
                for h, v in zip(headers, row):
                    if v is not None and str(v).strip():
                        pairs.append(f"{h}: {v}")
                if pairs:
                    text_parts.append(", ".join(pairs))
                row_count += 1
        wb.close()
    except Exception as e:
        print(f"  XLSX read error: {e}")
        return ""
    
    return "\n".join(text_parts)


# -----------------------------------------------------------------
# UNIFIED INGESTION
# -----------------------------------------------------------------
def extract_text(file_path: str, filename: str) -> str:
    """Route to the correct text extractor based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".csv":
        return extract_text_from_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return extract_text_from_xlsx(file_path)
    else:
        print(f"  Unsupported file type: {ext}")
        return ""


def ingest_file(file_path: str, filename: str, client: httpx.Client = None) -> int:
    """Reads any supported file, chunks it, embeds it, and saves to LanceDB."""
    # Check file size
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_INGEST_SIZE_MB:
        print(f"  [SKIP] File too large ({size_mb:.1f} MB > {MAX_INGEST_SIZE_MB} MB): {filename}")
        return 0
    
    full_text = extract_text(file_path, filename)
    if not full_text or len(full_text.strip()) < 50:
        print(f"  [SKIP] No meaningful text extracted from: {filename}")
        return 0
    
    chunks = chunk_text(full_text)
    if not chunks:
        return 0
    
    # Prepare data array for LanceDB (must contain a 'vector' key)
    successful_data = []
    
    # Process sequentially to avoid Ollama parallel batching panics.
    # Re-using the persistent client with direct IP (127.0.0.1) keeps network overhead under 1ms, 
    # achieving high speeds without triggering server failures.
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk, client=client)
        if not embedding:
            raise RuntimeError(f"Failed to embed chunk {i} for {filename} (empty embedding returned). skipping DB insertion to prevent partial ingestion.")
        
        successful_data.append({
            "id": f"{filename}_chunk_{i}",
            "vector": embedding,
            "text": chunk,
            "source": filename
        })

    # Insert into LanceDB
    if TABLE_NAME in db.table_names():
        # Open existing table and append new data
        table = db.open_table(TABLE_NAME)
        table.add(successful_data)
    else:
        # Create a new table automatically inferring the schema from the data
        db.create_table(TABLE_NAME, data=successful_data)
        
    return len(successful_data)


def ingest_pdf(file_path: str, filename: str) -> int:
    """Legacy wrapper -- kept for backward compatibility with app.py."""
    return ingest_file(file_path, filename)


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


# -----------------------------------------------------------------
# BULK INGESTION -- run with: python database.py
# -----------------------------------------------------------------
def get_ingested_sources() -> set:
    """Return the set of source filenames already in the DB."""
    if TABLE_NAME not in db.table_names():
        return set()
    table = db.open_table(TABLE_NAME)
    try:
        results = table.search().limit(10000).to_list()
        return set(r.get("source") for r in results if "source" in r)
    except Exception:
        return set()


def ingest_all(data_dir: Path = DATA_DIR, force: bool = False):
    """Walk the DATA directory and ingest every supported file into LanceDB.
    
    Args:
        data_dir: Path to folder containing documents.
        force: If True, re-ingest files that are already in the DB.
    """
    supported_extensions = {".pdf", ".csv", ".xlsx", ".xls"}
    
    files = sorted([
        f for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ])
    
    if not files:
        print(f"No supported files found in {data_dir}")
        return
    
    # Check what's already ingested
    already_ingested = set() if force else get_ingested_sources()
    
    print("=" * 60)
    print("  LANCEDB BULK INGESTION")
    print("=" * 60)
    print(f"  Data directory : {data_dir.resolve()}")
    print(f"  Files found    : {len(files)}")
    print(f"  Already in DB  : {len(already_ingested)}")
    print(f"  Force re-ingest: {force}")
    print("=" * 60)
    
    total_chunks = 0
    ingested_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.time()
    
    with httpx.Client(timeout=30.0) as client:
        for i, filepath in enumerate(files, 1):
            filename = filepath.name
            ext = filepath.suffix.lower()
            
            # Skip already-ingested files
            if filename in already_ingested:
                print(f"\n[{i}/{len(files)}] [SKIP] Already ingested: {filename}")
                skipped_count += 1
                continue
            
            print(f"\n[{i}/{len(files)}] [{ext.upper()[1:]}] Ingesting: {filename}", flush=True)
            
            try:
                chunk_count = ingest_file(str(filepath), filename, client=client)
                
                if chunk_count > 0:
                    print(f"  [OK] {chunk_count} chunks embedded and stored", flush=True)
                    total_chunks += chunk_count
                    ingested_count += 1
                else:
                    print(f"  [SKIP] No chunks produced", flush=True)
                    skipped_count += 1
                    
            except Exception as e:
                print(f"  [FAIL] {e}")
                failed_count += 1
    
    elapsed = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Ingested  : {ingested_count} files")
    print(f"  Skipped   : {skipped_count} files")
    print(f"  Failed    : {failed_count} files")
    print(f"  Total chunks in this run: {total_chunks}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    # Show DB stats
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        try:
            row_count = table.count_rows()
            print(f"  Total chunks in DB: {row_count}")
        except Exception:
            pass
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest all documents from DATA/ into LanceDB"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest files even if they are already in the database"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override the data directory path (default: ./DATA)"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        exit(1)
    
    ingest_all(data_dir=data_dir, force=args.force)