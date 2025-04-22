# 📚 IDB-RAG: Dual Chatbot System with OpenAI & Weaviate

A Retrieval-Augmented Generation (RAG) system built for the Inter-American Development Bank (IDB) to semantically query evaluation reports. This version features **two specialized chatbots**:

- **General Chatbot**: Queries all evaluation reports.
- **Barbados Chatbot**: Queries reports specific to Barbados.

Powered by **OpenAI GPT-4.1**, **Weaviate** vector database, and a clean **Streamlit UI**.

---

## 🚀 Features

- **Two Chatbots** with distinct document sets
- **Embeddings** and **responses** generated via OpenAI API
- **Vector search** using Weaviate with semantic retrieval
- **Interactive chat UI** built with Streamlit
- Fully dockerizable & cloud‑ready deployment

---

## 🧰 Prerequisites

- Python 3.10+
- A valid OpenAI API key
- Weaviate running locally or remotely
- Git

---

## ⚙️ Setup

1. **Clone the repository**

```bash
git clone https://github.com/baltierra/idb-rag.git
cd idb-rag
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file with the following:

```bash
OPENAI_API_KEY=your_openai_api_key
WEAVIATE_URL=your_weaviate_bd_url
WEAVIATE_API_KEY=optional_if_required_from_your_weaviate_db
```

---

## 📥 Ingest Reports

1. Place your PDF files in one of the following folders:
   - `src/data/general_reports/`
   - `src/data/barbados_reports/`

2. Run the document loader script:

```bash
python src/weaviate_setup.py --dir general_reports
python src/weaviate_setup.py --dir barbados_reports
```

Use `--reset` to clear and rebuild the index.

---

## 💬 Start the Streamlit App

```bash
streamlit run src/appeval.py --server.address 0.0.0.0 --server.port 8501

OR

streamlit run src/barbadosdata.py --server.address 0.0.0.0 --server.port 8501
```

Then open your browser at `http://localhost:8501` (or your VM IP).

You’ll be able to:
- Select **General Chatbot** or **Barbados Chatbot** from the sidebar.
- Enter natural language queries.
- See answers with **source citations**.

---

## 🧪 Notes

- This app is a **prototype** for showcasing search across evaluation reports using RAG.
- Additional datasets or countries can be added by repeating the ingestion process with new document folders.

---

## 📫 Contact

For access issues or bugs, please email **baltierra@me.com**