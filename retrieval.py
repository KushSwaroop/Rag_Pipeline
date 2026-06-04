import re
import httpx
import logging
import asyncio
from db import get_table

logger = logging.getLogger(__name__)

# Core stop words to filter out if fallback keyword search is used
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
    "with", "by", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "of", "up", "down", 
    "is", "are", "was", "were", "be", "been", "being", "have", "has", 
    "had", "do", "does", "did", "should", "would", "could", "this", "that"
}

async def get_embedding_async(text: str, client: httpx.AsyncClient = None) -> list:
    """Calls local Ollama to embed a query string.
    
    Prepends 'search_query: ' as required by nomic-embed-text.
    """
    url = "http://127.0.0.1:11434/api/embeddings"
    # Prepend search_query for search queries
    prompt = f"search_query: {text}"
    json_data = {"model": "nomic-embed-text", "prompt": prompt}
    
    for attempt in range(5):
        try:
            if client:
                response = await client.post(url, json=json_data)
            else:
                async with httpx.AsyncClient(timeout=30.0) as c:
                    response = await c.post(url, json=json_data)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            if attempt < 4:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            else:
                logger.error(f"Async embedding error: {e}")
                return []

async def get_embeddings_batch_async(texts: list, client: httpx.AsyncClient = None) -> list:
    """Calls local Ollama to embed a batch of texts.
    
    Uses the modern /api/embed endpoint.
    Prepends 'search_document: ' as required by nomic-embed-text.
    """
    url = "http://127.0.0.1:11434/api/embed"
    # Prepend search_document for ingestion chunks
    inputs = [f"search_document: {t}" for t in texts]
    json_data = {"model": "nomic-embed-text", "input": inputs}
    
    for attempt in range(5):
        try:
            if client:
                response = await client.post(url, json=json_data)
            else:
                async with httpx.AsyncClient(timeout=300.0) as c:
                    response = await c.post(url, json=json_data)
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            if attempt < 4:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            else:
                logger.error(f"Async batch embedding error: {e}")
                return []

async def retrieve_chunks_async(query: str, top_k: int = 6, client: httpx.AsyncClient = None) -> list:
    """Searches LanceDB using RRF hybrid search (Cosine Vector + native FTS / Fallback Keyword)."""
    table = get_table()
    if not table:
        return []
    
    # 1. Fetch query vector embedding
    query_embedding = await get_embedding_async(query, client=client)
    if not query_embedding:
        return []
        
    # 2. Vector search (Cosine)
    def run_vector_search():
        return table.search(query_embedding).metric("cosine").limit(top_k * 2).to_list()
        
    vector_results = await asyncio.to_thread(run_vector_search)
    
    # 3. Native Full-Text Search (BM25 keyword search)
    def run_fts_search():
        try:
            # Query LanceDB FTS index
            return table.search(query).limit(top_k * 2).to_list()
        except Exception as fts_error:
            logger.warning(f"Native FTS query failed: {fts_error}. Falling back to manual keyword matching.")
            # Fallback Python keyword matching
            try:
                all_records = table.search().limit(20000).to_list()
                query_words = [w for w in re.findall(r'\w+', query.lower()) if w not in STOP_WORDS and len(w) > 2]
                if not query_words:
                    query_words = [w for w in re.findall(r'\w+', query.lower())]
                    
                scored_records = []
                for r in all_records:
                    text_lower = r["text"].lower()
                    score = sum(text_lower.count(qw) for qw in query_words)
                    if score > 0:
                        scored_records.append((score, r))
                scored_records.sort(key=lambda x: x[0], reverse=True)
                return [item[1] for item in scored_records[:top_k * 2]]
            except Exception as fallback_error:
                logger.error(f"Fallback keyword search failed: {fallback_error}")
                return []
                
    keyword_results = await asyncio.to_thread(run_fts_search)
    
    # 4. Reciprocal Rank Fusion (RRF)
    k = 60
    scores = {}
    docs = {}
    
    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        docs[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        
    for rank, doc in enumerate(keyword_results):
        doc_id = doc["id"]
        docs[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        
    # Sort docs by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Return raw text of top_k results
    return [docs[doc_id]["text"] for doc_id in sorted_ids[:top_k]]
