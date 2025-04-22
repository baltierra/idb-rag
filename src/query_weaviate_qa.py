#!/usr/bin/env python3
import os
import argparse
from weaviate import Client, AuthApiKey

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_community.llms.ollama import Ollama

# ─── CONFIG ────────────────────────────────────────────────────
WEAVIATE_URL     = os.getenv(
    "WEAVIATE_URL",
    "https://ujkd968quupmdn4b9mzg.c0.us-west3.gcp.weaviate.cloud"
)
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CLASS_NAME       = "CountryEval"

# ─── ARGS ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("question", help="Your free‑form question")
parser.add_argument("-k", type=int, default=3, help="How many chunks to retrieve")
args = parser.parse_args()

# ─── CONNECT & BUILD RETRIEVAL STORE ─────────────────────────
client = Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
)

emb = OllamaEmbeddings(model="nomic-embed-text")
store = Weaviate(
    client=client,
    index_name=CLASS_NAME,
    text_key="content",
    embedding=emb,
    by_text=False,        # use nearVector
)
# retrieve top‑k chunks
docs_and_scores = store.similarity_search_with_score(args.question, k=args.k)

# ─── BUILD A SINGLE “CONTEXT” & CALL THE LLM ────────────────
# join the retrieved chunks into one prompt
context = "\n\n---\n\n".join(doc.page_content for doc, _ in docs_and_scores)

prompt = f"""
Answer the question based **only** on the context below.  Cite each source by PDF name and page:
  
{context}

---

**Question:** {args.question}

**Answer (with sources):**
"""

# use your smaller LLM to stay fast
llm = Ollama(model="deepseek-r1:8b")  
response = llm.invoke(prompt)

# ─── OUTPUT ───────────────────────────────────────────────────
print("\n=== RAG Answer ===\n")
print(response)
print("\n=== Retrieved Chunks ===\n")
for i, (doc, score) in enumerate(docs_and_scores, 1):
    src = doc.metadata.get("source","?")
    print(f"{i}. [{src}] (score={score:.3f})")
    print(f"   {doc.page_content[:200]}…\n")