<div align="center">
  <h1>🧠 Universal Multimodal RAG Gateway</h1>
  <p><i>A high-performance, GPU-accelerated Retrieval-Augmented Generation pipeline powered by LanceDB, FastAPI, and Streamlit.</i></p>

  ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
  ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
  ![LanceDB](https://img.shields.io/badge/LanceDB-Vector_Search-purple)
  ![Ollama](https://img.shields.io/badge/Local_LLM-Ollama-black)
</div>

---

## 📖 Overview

The **Universal Multimodal RAG Gateway** is an enterprise-grade Retrieval-Augmented Generation system designed for seamless document processing and intelligent querying. It features a scalable FastAPI backend that handles asynchronous chunking and vector embeddings, paired with a sleek, dark-mode Streamlit frontend for an intuitive chatting experience.

Built to run entirely locally using **Ollama** (or optionally connect to cloud providers like OpenAI and Anthropic), the gateway dynamically processes large datasets securely on your local hardware.

## ✨ Key Features

- **🚀 GPU-Accelerated Batching:** Streams embeddings sequentially over a persistent local HTTP connection to prevent memory exhaustion, optimizing local GPU utilization.
- **📄 Multimodal Ingestion:** Dynamically routes file types. Supports PDFs, CSV datasets, Excel spreadsheets, and raw text files.
- **🧠 Adaptive Chunking Strategies:** Uses custom regex and structural chunking specifically tuned for academic papers, slide decks, and government reports.
- **⚡ Async FastAPI Backend:** Non-blocking endpoints for both background ingestion and asynchronous LLM streaming.
- **🎨 Premium UI:** A customized, responsive Streamlit interface with sidebar LLM configurations, file uploads, and streaming chat delivery.
- **🔌 Universal Provider Support:** Switch seamlessly between local models (Ollama) and cloud APIs (OpenAI, Anthropic) directly from the UI.

---

## 🏗️ Architecture Stack

1. **Frontend:** [Streamlit](https://streamlit.io/) (Graphical Chat Interface & File Staging)
2. **Backend API:** [FastAPI](https://fastapi.tiangolo.com/) (Routing, Uploads, LLM Proxy)
3. **Vector Database:** [LanceDB](https://lancedb.com/) (Embedded, high-speed vector & keyword search)
4. **Embeddings Engine:** Local Ollama (`nomic-embed-text`)
5. **Generation Engine:** Local Ollama (`qwen2.5:3b` by default)

---

## 🚀 Getting Started

You will need **two separate terminal windows** to run both the backend API and the frontend UI simultaneously. 

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.
- Required Ollama models pulled:
  ```bash
  ollama run qwen2.5:3b
  ollama pull nomic-embed-text
  ```

### 1. Environment Setup

Activate your virtual environment (Conda or venv):
```bash
# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

*(Note: Windows users may also need to install Tesseract OCR if processing raw images).*

### 2. Start the Backend API

In your first terminal, start the FastAPI server:
```bash
python main.py
```
*The API will be available at `http://localhost:8000`. You can view the swagger docs at `/docs`.*

### 3. Start the Frontend UI

In your second terminal (ensure the virtual environment is activated here too!), launch the Streamlit chat interface:
```bash
python -m streamlit run app.py
```
*The UI will launch in your default web browser at `http://localhost:8501`.*

---

## 💡 Usage Guide

1. **Upload Data:** Open the left sidebar in the web interface and upload your PDFs, CSVs, or text files.
2. **Ingest & Embed:** Click **"Process & Ingest Into DB"**. The frontend will securely transmit the files to the backend, which will asynchronously chunk and embed them using the GPU. You can check `task` logs or run `python check_progress.py` to monitor insertion.
3. **Configure LLM:** Select your preferred provider (Ollama, OpenAI, Anthropic) and adjust the temperature setting in the sidebar.
4. **Chat:** Ask questions in the chat box! The system will perform semantic retrieval against LanceDB and stream the synthesized response back to you in real-time, complete with source citations.

---

## 🛠️ Utilities

- **`database.py`:** Run `python database.py` to bulk-ingest the entire `DATA` folder completely headless, without using the web UI.
- **`check_progress.py`:** Run `python check_progress.py` to see exactly how many chunks and which files are currently indexed in your LanceDB instance.
