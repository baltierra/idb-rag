# FactBot — RAG Chatbot with Ollama & Deepseek R1

A Retrieval‑Augmented Generation (RAG) system that embeds your own PDF documents with Ollama + Deepseek R1, stores vectors in Chroma, and exposes a Streamlit chat UI.

---

## 🚀 Features

- **Embeddings & LLM** via Ollama (locally hosted models: `nomic-embed-text`, `deepseek-r1:14b`)
- **Vector store** backed by Chroma (persistent SQLite + blobs)
- **CLI tools** for ingestion (`load_documents.py`) and querying (`rag_query.py`)
- **Interactive UI** with Streamlit (`UI.py`)  
- **Pure Python & open‑source** — no external paid APIs

---

## 🔧 Prerequisites

- Ubuntu 22.04+ (or macOS/Linux)
- Python 3.12+  
- [Ollama CLI](https://ollama.com) installed & in `$PATH`  
- ≥ 16 GiB RAM for `deepseek‑r1:14b`  
- Git

---

## ⚙️ Setup

1. **Clone your repo**  
   ```bash
   cd ~
   git clone git@github.com:baltierra/idb-rag.git
   cd idb-rag
   ```
2. **Create & activate venv**
   ```bash
   python3 -m venv idb-rag
   source idb-rag/bin/activate
   ```
3. **Install Python deps**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **nstall Ollama & pull models**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull nomic-embed-text
   ollama pull deepseek-r1:14b
   ```
5. **5.	Start Ollama server**
   ```bash
   pkill ollama         # stop any running instance
   OLLAMA_HOST=127.0.0.1:11434 ollama serve & disown
   ```

---

## 📥 Ingest PDFs

1. Copy your .pdf files into `src/data/<folder>/` (e.g. `src/data/reports/`).
2. Run the loader:
   ```bash
   python src/load_documents.py --dir reports
   ```
   - `--dir` points at a subdirectory under `src/data/`.
   - Use `--reset` to clear and rebuild the vector store.

---

## ⁉️ CLI Query

```bash
python src/rag_query.py "Your natural‑language question here"
```
- Prints the model’s answer and source chunk IDs.

---

## 💬 Streamlit UI

```bash
streamlit run src/UI.py \
  --server.address 0.0.0.0 --server.port 8501
```
- Visit http://<VM‑IP>:8501 (or localhost:8501 via SSH tunnel).
- Chat in the sidebar; FactBot responds with sources.
