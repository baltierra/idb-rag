#!/usr/bin/env python3
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

from weaviate import Client, AuthApiKey

# ─── Load .env ─────────────────────────────────────────────────
# assumes .env lives one level above src/
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ─── Grab environment variables ─────────────────────────────────
WEAVIATE_URL     = os.getenv("WEAVIATE_URL", "").strip()
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "").strip()

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    print("❌ Missing WEAVIATE_URL or WEAVIATE_API_KEY in your environment.")
    print("   Make sure your .env has:")
    print("     WEAVIATE_URL=https://your‑cluster.weaviate.cloud")
    print("     WEAVIATE_API_KEY=your‑key‑here")
    sys.exit(1)

# ─── Connect to Weaviate ────────────────────────────────────────
client = Client(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
)

# ─── Fetch & display class names ───────────────────────────────
schema = client.schema.get()
classes = [c["class"] for c in schema.get("classes", [])]

print("\n📋  Classes in your Weaviate schema:")
for cls in classes:
    print(" •", cls)