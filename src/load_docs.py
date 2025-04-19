#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from embedding import get_embedding_function

# ——————————————————————————————————————————————————————————
#  CONFIGURATION
# ——————————————————————————————————————————————————————————
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma"

# ——————————————————————————————————————————————————————————
#  UTILITY FUNCTIONS
# ——————————————————————————————————————————————————————————
def load_documents(pdf_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_page_id = None
    chunk_index  = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page   = chunk.metadata.get("page")
        page_id = f"{source}:{page}"

        if page_id == last_page_id:
            chunk_index += 1
        else:
            chunk_index = 0

        chunk.metadata["id"] = f"{page_id}:{chunk_index}"
        last_page_id = page_id

    return chunks

def clear_database():
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

def add_to_db(chunks: list[Document]):
    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embedding_function()
    )
    CHROMA_DIR.mkdir(exist_ok=True)

    items      = db.get(include=[])
    existing   = set(items["ids"])
    new_chunks = [c for c in calculate_chunk_ids(chunks)
                  if c.metadata["id"] not in existing]

    if new_chunks:
        print(f"👉 Adding {len(new_chunks)} new chunks")
        ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=ids)
        db.persist()
    else:
        print("✅ All documents already in DB")

# ——————————————————————————————————————————————————————————
#  MAIN PROCESS
# ——————————————————————————————————————————————————————————
def main():
    parser = argparse.ArgumentParser(
        description="Load (or reset) your Chroma DB from all PDFs in a directory"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete and recreate the Chroma database."
    )
    parser.add_argument(
        "--dir", "-d", type=str, required=True,
        help="Directory (inside the data/ folder) containing PDF files to ingest."
    )
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing existing database…")
        clear_database()

    pdf_dir = DATA_DIR / args.dir
    if not pdf_dir.is_dir():
        print(f"❌ Directory not found: {pdf_dir}")
        return

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in: {pdf_dir}")
        return

    print(f"📂 Found {len(pdf_files)} PDF(s) in {pdf_dir}:")
    for p in pdf_files:
        print(f"  • {p.name}")

    # Process each file one by one
    for pdf_path in pdf_files:
        print(f"\n📄 Ingesting {pdf_path.name}")
        docs   = load_documents(pdf_path)
        chunks = split_documents(docs)
        add_to_db(chunks)

if __name__ == "__main__":
    main()