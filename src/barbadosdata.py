#!/usr/bin/env python3
import os
import openai
import weaviate
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv
from weaviate import AuthApiKey, Client

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate

from embedding import get_embedding_function

# ─── LOAD ENV ─────────────────────────────────────────────────
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY_IDB_RAG")
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

openai.api_key = OPENAI_API_KEY

# ─── STREAMLIT PAGE CONFIG ───────────────────────────────────
st.set_page_config(page_title="FactBot Barbados (GPT-4.1+Weaviate)")
with st.sidebar:
    st.title("IDB FactBot: Barbados")
    st.markdown(
        """
        This chatbot provides information about Barbados, based on reports
        produced by the Inter-American Development Bank's Office of Evaluation
        and Oversight (OVE).

        You can ask detailed questions about IDB's Barbados-related reports and
        receive precise, cited answers—powered by OpenAI's GPT-4.1 and Weaviate.
        Please note: this is a prototype demo and may contain hallucinations or inaccuracies.
        
        The available reports to query from are:
        - Approach Paper: Extended Country Pogram Evaluation - Barbados (2015-2023).
        - Barbados, a Time for Change - Country Development Challenges.
        - Extended Country Program Evaluation - Barbados (2015-2023).
        - Index of Governance and Public Policy in Disaster Risk Management - Barbados.
        - Public Investment Profile for Disaster Risk Reduction - Barbados.
        """
    )

# ─── WEAVIATE + VECTORSTORE SETUP (once on import) ───────────
client = Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
)
embedder = get_embedding_function()
store    = Weaviate(
    client=client,
    index_name="BarbadosData",
    text_key="content",
    embedding=embedder,
    by_text=False,       # force nearVector
    attributes=["source"],
)

# ─── EXPERT‑CONSULTANT PROMPT TEMPLATE ───────────────────────
SYSTEM_PROMPT = """
You are an expert consultant that works for the Inter-American Development Bank (IDB).
Your job is to prepare evaluations for IDB member countries and expose that
information in reports, being able to answer questions about those reports.
You happen to be an expert in Barbados information given your background.
"""

# ─── RAG FUNCTION ─────────────────────────────────────────────
def generate_response(question: str, k: int = 5) -> str:
    # 1) Retrieve top‑k chunks
    docs_and_scores = store.similarity_search_with_score(question, k=k)

    # 2) Build the “context” block
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in docs_and_scores)

    # 3) Compose OpenAI Chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""
Context:
{context}

---

Question: {question}

Please answer *only* using the context above, in a concise paragraph, 
and cite each fact with the source PDF filename and page number.
"""}
    ]

    # 4) Call GPT‑4.1
    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    answer = resp.choices[0].message.content.strip()

    # 5) List unique sources
    sources = sorted({doc.metadata.get("source","unknown") for doc, _ in docs_and_scores})
    sources_md = "\n".join(f"- {s}" for s in sources)

    # 6) Return answer + source list
    return f"{answer}\n\n**Sources:**\n{sources_md}"

# ─── STREAMLIT CHAT UI ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """👋 Welcome to FactBot Barbados!  
Ask me anything about Barbados evaluation process or related information.

Some example questions that you can ask me:
- Can you summarize the main Barbados development challenges covered in the report "Barbados, a Time for Change"?
- What are the final recommendations in the Extended Country Program Evaluation (XCPE) report for Barbados?
- Give me a summary of the results of Barbados' Extended Country Program Evaluation report.
"""
    }]

# Render existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if user_q := st.chat_input("Type your question here…"):
    # Append user message
    st.session_state.messages.append({"role":"user","content":user_q})
    with st.chat_message("user"):
        st.write(user_q)

    # Generate & display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = generate_response(user_q, k=5)
            st.markdown(reply)

    # Save assistant message
    st.session_state.messages.append({"role":"assistant","content":reply})