import os
import json
import httpx
import logging
import asyncio
import uvicorn
from enum import Enum
from pathlib import Path
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db import add_to_table
from database import extract_and_chunk, get_embedding
from retrieval import get_embeddings_batch_async, retrieve_chunks_async

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Universal RAG Gateway API")
DATA_DIR = Path(__file__).parent / "DATA"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# PROVIDER ENUM
# ---------------------------------------------------------
class Provider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    ollama = "ollama"
    litellm = "litellm"

PROVIDER_DEFAULTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "ollama": "http://localhost:11434/v1/chat/completions",
    "litellm": "http://localhost:4000/v1/chat/completions"
}

# ---------------------------------------------------------
# CONFIG / BYOK LAYER
# ---------------------------------------------------------
class LLMConfig(BaseModel):
    provider: Provider = Provider.ollama
    model: str = "qwen2.5:3b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.5
    max_tokens: int = 1024
    top_p: float = 1.0

# ---------------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------------
class RAGPayload(BaseModel):
    query: str
    chunks: Optional[List[str]] = None
    num_ctx: int = 4096
    config: LLMConfig

# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are an intelligent, analytical AI assistant.
Your task is to answer the user's query thoughtfully, using the provided context chunks as your primary source of truth.

GUIDELINES:
1. Synthesize the information: Connect ideas across different context chunks instead of just copying text.
2. Reason through complex queries: If the user asks an analytical question, explain your thought process based on the context.
3. Be transparent: If the provided documents only partially answer the question, explain what is known from the context and what remains unclear.
4. Grounding: Do not invent facts outside the provided documents, but you may use your general knowledge to explain or interpret the context.
"""

# ---------------------------------------------------------
# CONTEXT BUILDER
# ---------------------------------------------------------
def build_context(request: RAGPayload) -> str:
    formatted_context = "\n\n---\n\n".join(
        [f"Document Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(request.chunks)]
    )

    return (
        f"CONTEXT DOCUMENTS:\n{formatted_context}\n\n"
        f"USER QUERY: {request.query}"
    )

# ---------------------------------------------------------
# NORMALIZED YIELD FORMATTER
# ---------------------------------------------------------
def format_yield(text: str, config: LLMConfig) -> str:
    payload = {
        "text": text,
        "provider": config.provider.value,
        "model": config.model
    }
    return json.dumps(payload) + "\n"

def format_error(error_msg: str, config: LLMConfig) -> str:
    payload = {
        "error": error_msg,
        "provider": config.provider.value,
        "model": config.model
    }
    return json.dumps(payload) + "\n"

def format_citations(chunks: List[str], config: LLMConfig) -> str:
    payload = {
        "citations": chunks,
        "provider": config.provider.value,
        "model": config.model
    }
    return json.dumps(payload) + "\n"

# ---------------------------------------------------------
# OPENAI-COMPATIBLE PAYLOAD
# ---------------------------------------------------------
def build_openai_payload(request: RAGPayload) -> dict:
    return {
        "model": request.config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_context(request)}
        ],
        "temperature": request.config.temperature,
        "max_tokens": request.config.max_tokens,
        "top_p": request.config.top_p,
        "stream": True
    }

# ---------------------------------------------------------
# PROVIDER ADAPTERS
# ---------------------------------------------------------
async def stream_openai_compatible(request: RAGPayload) -> AsyncGenerator[str, None]:
    payload = build_openai_payload(request)
    base_url = (
        request.config.base_url
        or PROVIDER_DEFAULTS.get(request.config.provider.value, "https://api.openai.com/v1/chat/completions")
    )
    headers = {"Content-Type": "application/json"}
    if request.config.api_key:
        headers["Authorization"] = f"Bearer {request.config.api_key}"

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", base_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                        delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                        if delta:
                            yield format_yield(delta, request.config)
                    except Exception:
                        continue
            yield format_citations(request.chunks, request.config)
    except httpx.HTTPStatusError as e:
        yield format_error(f"HTTP Error: {e.response.status_code} - {e.response.text}", request.config)
    except Exception as e:
        yield format_error(f"Stream Error: {str(e)}", request.config)

async def stream_anthropic(request: RAGPayload) -> AsyncGenerator[str, None]:
    headers = {
        "x-api-key": request.config.api_key or "",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": request.config.model,
        "max_tokens": request.config.max_tokens,
        "temperature": request.config.temperature,
        "top_p": request.config.top_p,
        "stream": True,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_context(request)}]
    }
    url = (
        request.config.base_url
        or PROVIDER_DEFAULTS.get(request.config.provider.value, "https://api.anthropic.com/v1/messages")
    )

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        data = json.loads(line[5:].strip())
                        if data.get("type") == "content_block_delta":
                            text = data.get("delta", {}).get("text")
                            if text:
                                yield format_yield(text, request.config)
                    except Exception:
                        continue
            yield format_citations(request.chunks, request.config)
    except httpx.HTTPStatusError as e:
        yield format_error(f"HTTP Error: {e.response.status_code} - {e.response.text}", request.config)
    except Exception as e:
        yield format_error(f"Stream Error: {str(e)}", request.config)

# ---------------------------------------------------------
# PROVIDER ROUTER
# ---------------------------------------------------------
def get_provider_streamer(provider: Provider):
    if provider in [Provider.openai, Provider.ollama, Provider.litellm]:
        return stream_openai_compatible
    if provider == Provider.anthropic:
        return stream_anthropic
    raise ValueError(f"Unsupported provider: {provider}")

# ---------------------------------------------------------
# BACKGROUND INGESTION WORKER
# ---------------------------------------------------------
async def run_ingestion_background(temp_file_path: Path, filename: str):
    try:
        logger.info(f"Starting background ingestion for: {filename}")
        # 1. Extraction and Chunking
        chunks = await asyncio.to_thread(extract_and_chunk, str(temp_file_path), filename)
        if not chunks:
            logger.warning(f"No chunks created for {filename}")
            if temp_file_path.exists():
                temp_file_path.unlink()
            return
            
        # 2. Sequential Embedding (To avoid Ollama 500 errors)
        ingest_data = []
        for idx, chunk_text_str in enumerate(chunks):
            embedding = await asyncio.to_thread(get_embedding, chunk_text_str)
            if not embedding:
                logger.error(f"Embedding failed for chunk {idx} of {filename}")
                continue
                
            ingest_data.append({
                "id": f"{filename}_chunk_{idx}",
                "vector": embedding,
                "text": chunk_text_str,
                "source": filename
            })
            
        # 5. Insert to LanceDB
        await asyncio.to_thread(add_to_table, ingest_data)
        logger.info(f"Successfully finished background ingestion for {filename} ({len(ingest_data)} chunks).")
    except Exception as e:
        logger.error(f"Error during background ingestion of {filename}: {e}")
    finally:
        # Keep the original in DATA but if we created any secondary temp files, we'd clean them up.
        pass

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "supported_providers": [p.value for p in Provider]
    }

@app.post("/api/upload")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Receives files, saves them to local DATA dir, and triggers background ingestion."""
    staged_files = []
    for file in files:
        target_path = DATA_DIR / file.filename
        
        # Save file to disk
        with open(target_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Queue the ingestion pipeline in the background
        background_tasks.add_task(run_ingestion_background, target_path, file.filename)
        staged_files.append(file.filename)
        
    return {
        "status": "staged",
        "message": f"Successfully queued {len(staged_files)} files for background processing.",
        "files": staged_files
    }

@app.post("/api/rag_generate")
async def generate_response(request: RAGPayload):
    if not request.chunks:
        # Asynchronous retrieval
        async with httpx.AsyncClient(timeout=30.0) as client:
            request.chunks = await retrieve_chunks_async(request.query, client=client)

    provider = request.config.provider

    # Check for API key requirements on hosted providers
    if provider == Provider.openai:
        api_key = request.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )
        request.config.api_key = api_key
    elif provider == Provider.anthropic:
        api_key = request.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Anthropic API key is missing. Please provide it in the config or set the ANTHROPIC_API_KEY environment variable."
            )
        request.config.api_key = api_key

    streamer = get_provider_streamer(request.config.provider)

    return StreamingResponse(
        streamer(request),
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )