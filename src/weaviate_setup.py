#!/usr/bin/env python3
from pathlib import Path
import os

# load .env
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import weaviate
from weaviate import AuthApiKey

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate

# Your OpenAI‐based embedding factory
from embedding import get_embedding_function

# ————————————————————————————————————————————————
# CONFIGURATION
# ————————————————————————————————————————————————
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CLASS_NAME       = "CountryEval"
DATA_DIR         = Path(__file__).parent / "data" / "extended_reports"

# ————————————————————————————————————————————————
# CONNECT TO WEAVIATE CLOUD (v3 client)
# ————————————————————————————————————————————————
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
    additional_headers={"X-Cors-Header": "*"}  # if you need CORS
)

# ————————————————————————————————————————————————
# RESET SCHEMA
# ————————————————————————————————————————————————
if client.schema.exists(CLASS_NAME):
    print(f"🔄 Deleting existing class `{CLASS_NAME}`")
    client.schema.delete_class(CLASS_NAME)

print(f"➕ Creating class `{CLASS_NAME}`")
client.schema.create_class({
    "class": CLASS_NAME,
    "vectorizer": "none",      # we supply our own embeddings
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source",  "dataType": ["string"]}
    ]
})

# ————————————————————————————————————————————————
# EMBEDDING & VECTORSTORE SETUP
# ————————————————————————————————————————————————
EMBED_FN = get_embedding_function()
store    = Weaviate(
    client=client,
    index_name=CLASS_NAME,
    text_key="content",
    attributes=["source"],
    embedding=EMBED_FN,
    by_text=False,
)

# ————————————————————————————————————————————————
# INGEST PDFS
# ————————————————————————————————————————————————
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=80, length_function=len
)

pdfs = sorted(DATA_DIR.glob("*.pdf"))
print(f"📂 Found {len(pdfs)} PDFs in {DATA_DIR}")
for pdf in pdfs:
    print(f"  • Loading {pdf.name}")
    docs   = PyPDFLoader(str(pdf)).load()
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = pdf.name
    store.add_documents(chunks)

print("✅ Ingestion complete")

# ————————————————————————————————————————————————
# SIMPLE QUERY TEST
# ————————————————————————————————————————————————
print("\n🔍 Testing similarity search…")
hits = store.similarity_search_with_score(
    "For what countries do you handle the Extended Country Program Evaluation?",
    k=4
)

print("\n🔍 Top 4 matches with scores:\n")
for i, (doc, score) in enumerate(hits, start=1):
    snippet = doc.page_content.replace("\n", " ").strip()[:200]
    src     = doc.metadata.get("source", "unknown")
    print(f"{i}. (score={score:.4f}) [{src}] {snippet}…")