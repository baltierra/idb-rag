#!/usr/bin/env python3
import os
from pathlib import Path

import weaviate
from weaviate import AuthApiKey

# LangChain imports
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate  # updated import

# ————————————————————————————————————————————————
# CONFIGURATION
# ————————————————————————————————————————————————
WEAVIATE_URL     = os.getenv("WEAVIATE_URL", "ujkd968quupmdn4b9mzg.c0.us-west3.gcp.weaviate.cloud")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CLASS_NAME       = "CountryEval"
DATA_DIR         = Path(__file__).parent / "data" / "barbados_reports"

# ————————————————————————————————————————————————
# CONNECT TO WEAVIATE CLOUD (v3 client)
# ————————————————————————————————————————————————
client = weaviate.Client(
    url="https://ujkd968quupmdn4b9mzg.c0.us-west3.gcp.weaviate.cloud",
    auth_client_secret=AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
    additional_headers={"X-Cors-Header": "*"}  # if you need CORS
)

# OPTIONAL: Clear existing class (v3 style)
if client.schema.exists(CLASS_NAME):
    client.schema.delete_class(CLASS_NAME)

# Define a new class schema (v3 style)
client.schema.create_class({
    "class": CLASS_NAME,
    "vectorizer": "none",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source",  "dataType": ["string"]}
    ]
})

# ————————————————————————————————————————————————
# EMBEDDING & VECTORSTORE SETUP
# ————————————————————————————————————————————————
EMBED_FN = OllamaEmbeddings(model="nomic-embed-text")
store    = Weaviate(
    client=client,
    index_name=CLASS_NAME,
    text_key="content",
    attributes=["source"],
    embedding=EMBED_FN,
)

# ————————————————————————————————————————————————
# INGEST PDFS
# ————————————————————————————————————————————————
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len)

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
client.close()

# ————————————————————————————————————————————————
# SIMPLE QUERY TEST
# ————————————————————————————————————————————————
print("\n🔍 Testing similarity search…")
hits = store.similarity_search("What years did this evaluation cover?", k=1)
print("Top match snippet:", hits[0].page_content[:200], "…")