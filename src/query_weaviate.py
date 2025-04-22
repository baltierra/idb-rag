#!/usr/bin/env python3
import os
import argparse

import weaviate
from weaviate import AuthApiKey

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Weaviate

# ─── CONFIG ────────────────────────────────────────────────────
WEAVIATE_URL     = os.getenv("WEAVIATE_URL", "https://ujkd968quupmdn4b9mzg.c0.us-west3.gcp.weaviate.cloud")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CLASS_NAME       = "CountryEval"   # must match what you used in ingestion

# ─── PARSE ARGS ───────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Query your Weaviate‐backed RAG store"
)
parser.add_argument("question", help="Your natural‑language question")
parser.add_argument("-k", type=int, default=3, help="How many matches to return")
args = parser.parse_args()


# ─── CONNECT ─────────────────────────────────────────────────
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
)

# ─── BUILD VECTORSTORE ──────────────────────────────────────
emb = OllamaEmbeddings(model="nomic-embed-text")
store = Weaviate(
    client=client,
    index_name=CLASS_NAME,
    text_key="content",
    embedding=emb,
    attributes=["source"],
    by_text=False,
)


# ─── RUN QUERY ───────────────────────────────────────────────
results = store.similarity_search_with_score(args.question, k=args.k)

print(f"\nTop {len(results)} hits for: “{args.question}”\n")
for i, (doc, score) in enumerate(results, 1):
    src = doc.metadata.get("source", "unknown")
    snippet = doc.page_content.replace("\n", " ").strip()
    print(f"{i}. (score={score:.4f}, source={src})")
    print(f"   {snippet[:200]}…\n")