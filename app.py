import streamlit as st
import httpx
import asyncio

st.set_page_config(page_title="Multimodal RAG Chat", page_icon="🧠", layout="wide")

# ---------------------------------------------------------
# Sidebar: UI Kit for File Uploads
# ---------------------------------------------------------
with st.sidebar:
    st.header("🗄️ Database Uploads")
    st.markdown("Upload files to be processed and chunked into the Vector DB.")
    
    # Document Upload (PDF)
    pdf_files = st.file_uploader("Upload Documents (PDF)", type=["pdf"], accept_multiple_files=True)
    
    # Image Upload (JPEG)
    img_files = st.file_uploader("Upload Images (JPEG/JPG)", type=["jpg", "jpeg"], accept_multiple_files=True)
    
    # Video/Audio Upload 
    media_files = st.file_uploader(
        "Upload Media (MP4, MP3, WAV)", 
        type=["mp4", "mp3", "wav"], 
        accept_multiple_files=True
    )
    
    if pdf_files or img_files or media_files:
        st.success("Files staged for processing.")
        st.caption("Note: You must implement an OCR/Transcription pipeline to convert images and media into text chunks before inserting them into your Vector DB.")

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.title("RAG Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Async function to consume the FastAPI stream
async def fetch_stream(query: str, chunks: list):
    payload = {
        "query": query,
        "chunks": chunks,
        "num_ctx": 4096
    }
    async with httpx.AsyncClient() as client:
        # Timeout is None to prevent closing connection during long generations
        async with client.stream("POST", "http://localhost:8000/api/rag_generate", json=payload, timeout=None) as response:
            async for text_chunk in response.aiter_text():
                yield text_chunk

# Chat Input Box
if prompt := st.chat_input("Ask a question about your files..."):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Show assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # ⚠️ PLACEHOLDER: In production, you will query your Vector DB here 
        # to retrieve the actual top 6-10 chunks based on the 'prompt'.
        mock_retrieved_chunks = [
            "This is dummy chunk 1 retrieved from the vector database.",
            "This is dummy chunk 2 containing relevant data."
        ]

        # FIX: Build the response inside the function and return it
        async def process_async_generator():
            response_text = ""
            # Consume the stream and update the UI in real-time
            async for chunk in fetch_stream(prompt, mock_retrieved_chunks):
                response_text += chunk
                message_placeholder.markdown(response_text + "▌") # Blinking cursor
            
            message_placeholder.markdown(response_text) # Final output
            return response_text

        # Run the async loop and capture the returned final string
        full_response = asyncio.run(process_async_generator())

    # 3. Save to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})