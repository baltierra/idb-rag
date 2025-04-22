#!/usr/bin/env python3
import os
from weaviate import Client, AuthApiKey

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_community.llms.ollama import Ollama

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# find the .env file in your project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ─── STREAMLIT PAGE CONFIG ───────────────────────────────────
st.set_page_config(page_title="IDB - FactBot")
with st.sidebar:
    st.title('IDB Evaluation Chatbot')
    st.markdown(
        """
        This chat interface lets you ask natural-language questions  
        againsts the Inter-American Development Bank's country-program  
        evaluation reports—and returns concise, sourced answers.
        """
    )

# ─── WEAVIATE + VECTORSTORE SETUP (run once on import) ─────────
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
CLASS_NAME       = "CountryEval"

client = Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
)
embedder = OllamaEmbeddings(model="nomic-embed-text")
store    = Weaviate(
    client=client,
    index_name=CLASS_NAME,
    text_key="content",
    embedding=embedder,
    by_text=False,         # use nearVector rather than nearText
)
llm = Ollama(model="deepseek-r1:8b")  # smaller, faster model

# ─── RESPONSE GENERATOR ────────────────────────────────────────
def generate_response(question: str) -> str:
    # 1) Retrieve top‑5 relevant chunks
    docs_and_scores = store.similarity_search_with_score(question, k=5)

    # 2) Build a single context block
    context = "\n\n---\n\n".join(
        doc.page_content for doc, _score in docs_and_scores
    )

    # 3) Prompt the LLM with context + question
    prompt = f"""
Answer the question based **only** on the context below. Cite each source by PDF name and page number.

{context}

---

**Question:** {question}

**Answer (with sources):**
"""
    answer = llm.invoke(prompt)

    # 4) Collect source file names for display
    sources = {doc.metadata.get("source", "unknown") for doc, _ in docs_and_scores}
    sources_md = "\n".join(f"- {s}" for s in sorted(sources))

    return f"{answer}\n\n**Sources:**\n{sources_md}"

# ─── STREAMLIT CHATLOG ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """👋 Welcome! Please ask me anything about IDB evaluations.

**Examples:**
- Summarize the main recommendations from the Costa Rica report.
- What years did the Paraguay evaluation cover?
- What is the latest GDP per capita recorded for the Dominican Republic?
- What are the key findings from the Barbados report?
"""
    }]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new user input
if user_input := st.chat_input("Type your question here…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            bot_response = generate_response(user_input)
            st.write(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})