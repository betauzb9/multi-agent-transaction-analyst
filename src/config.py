"""
F1 — Config: reads all keys/settings from .env. Never commit .env (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # optional — F4 skips gracefully without it
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")  # optional — F12
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")  # optional — F12
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")  # embedded, no signup needed
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fin01_docs")

# F10 — long-term memory needs its OWN persisted local path (separate from the docs
# vectorstore above), so past turns survive across separate process runs too.
MEMORY_QDRANT_PATH = os.getenv("MEMORY_QDRANT_PATH", "./qdrant_memory_data")

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./data/company.db")

RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "15"))  # F9 — guarantees termination
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))  # F8/F9 — caps the critic revise-loop


def require_google_key():
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key at aistudio.google.com/apikey "
            "and put it in your own .env (see .env.example)."
        )
