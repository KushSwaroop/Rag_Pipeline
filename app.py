import streamlit as st
import httpx
import asyncio
import json

st.set_page_config(page_title="Multimodal RAG Chat", page_icon="🧠", layout="wide")

# Inject premium dark-mode custom styles
st.markdown(
    """
    <style>
    /* Premium visual styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }
    
    /* Heading typography */
    h1, h2, h3 {
        color: #f3f4f6 !important;
        font-weight: 700 !important;
    }
    
    /* Buttons custom HSL styling */
    div.stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #2563eb !important;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4) !important;
    }
    
    /* File uploader enhancements */
    [data-testid="stFileUploader"] {
        background-color: #1f2937;
        border: 2px dashed #4b5563;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar: UI Kit for File Uploads and LLM Config
# ---------------------------------------------------------
with st.sidebar:
    st.header("🗄️ Database Uploads")
    st.markdown("Upload files to stage and process them in the RAG Vector Database.")
    
    # Document Upload (PDF)
    pdf_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)
    
    # Image Upload (JPEG/PNG)
    img_files = st.file_uploader("Upload Images (JPEG/JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # Video/Audio Upload 
    media_files = st.file_uploader(
        "Upload Media (MP4, MP3, WAV)", 
        type=["mp4", "mp3", "wav"], 
        accept_multiple_files=True
    )
    
    # Collect all uploaded files
    all_uploads = []
    if pdf_files:
        all_uploads.extend(pdf_files)
    if img_files:
        all_uploads.extend(img_files)
    if media_files:
        all_uploads.extend(media_files)
        
    if all_uploads:
        st.success(f"{len(all_uploads)} file(s) staged.")
        if st.button("🔥 Process & Ingest Into DB"):
            with st.spinner("Uploading and indexing in progress..."):
                files_payload = []
                for file in all_uploads:
                    files_payload.append(
                        ("files", (file.name, file.getvalue(), file.type))
                    )
                try:
                    response = httpx.post(
                        "http://localhost:8000/api/upload",
                        files=files_payload,
                        timeout=300.0
                    )
                    if response.status_code == 200:
                        st.balloons()
                        st.success("Successfully queued for background ingestion!")
                        st.info(response.json().get("message", ""))
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

    st.markdown("---")
    st.header("⚙️ LLM Engine Settings")
    
    provider = st.selectbox(
        "Provider",
        options=["ollama", "openai", "anthropic", "litellm"],
        index=0
    )
    
    # Default model helper based on provider
    default_model = "qwen2.5:3b"
    if provider == "openai":
        default_model = "gpt-4o-mini"
    elif provider == "anthropic":
        default_model = "claude-3-5-sonnet-20241022"
    elif provider == "litellm":
        default_model = "gpt-4o-mini"
        
    model = st.text_input("Model Name", value=default_model)
    api_key = st.text_input("API Key (optional)", type="password", value="")
    base_url = st.text_input("Base URL (optional)", value="")
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    max_tokens = st.number_input("Max Tokens", min_value=1, max_value=8192, value=1024, step=64)

# Prepare LLM configuration payload
config_payload = {
    "provider": provider,
    "model": model,
    "api_key": api_key if api_key else None,
    "base_url": base_url if base_url else None,
    "temperature": temperature,
    "max_tokens": max_tokens
}

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.title("🧠 Multimodal Hybrid RAG Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Async function to consume the FastAPI stream
async def fetch_stream(query: str, config: dict):
    payload = {
        "query": query,
        "config": config,
        "num_ctx": 4096
    }
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", "http://localhost:8000/api/rag_generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield line
    except Exception as e:
        yield json.dumps({"error": f"Connection error: {str(e)}"})

# Chat Input Box
if prompt := st.chat_input("Ask a question about your documents, spreadsheet data, or transcribed media..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        async def process_async_generator():
            response_text = ""
            citations = []
            async for line in fetch_stream(prompt, config_payload):
                try:
                    data = json.loads(line)
                    if "text" in data:
                        response_text += data["text"]
                        message_placeholder.markdown(response_text + "▌")
                    elif "citations" in data:
                        citations = data["citations"]
                    elif "error" in data:
                        response_text += f"\n\n⚠️ **Error:** {data['error']}"
                        message_placeholder.markdown(response_text)
                except Exception:
                    pass
            
            # Format and display sources at the end
            if citations:
                response_text += "\n\n**Retrieved Sources:**\n"
                # Display unique sources
                unique_citations = list(set(citations))
                for i, cite in enumerate(unique_citations, 1):
                    # Clean/preview the source chunk
                    clean_cite = cite.replace("\n", " ").strip()
                    response_text += f"{i}. *{clean_cite[:140]}...*\n"
                    
            message_placeholder.markdown(response_text)
            return response_text

        # Run the async loop
        full_response = asyncio.run(process_async_generator())

    st.session_state.messages.append({"role": "assistant", "content": full_response})