from langchain_openai import OpenAIEmbeddings
import os

def get_embedding_function():
    # picks up OPENAI_API_KEY from env/.env
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY_IDB_RAG")
    )