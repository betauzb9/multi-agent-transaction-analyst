"""
F10 — Long-term memory: a separate Qdrant collection of past (question, answer) turns, so
follow-ups like "and what about last quarter?" can be answered using earlier context.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from src import config
from src.llm import get_embeddings

MEMORY_COLLECTION = "fin01_memory"
_memory_store = None


def _get_memory_store() -> QdrantVectorStore:
    global _memory_store
    if _memory_store is None:
        embeddings = get_embeddings()
        client = QdrantClient(path=config.MEMORY_QDRANT_PATH)
        if not client.collection_exists(MEMORY_COLLECTION):
            dim = len(embeddings.embed_query("dimension probe"))
            client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        _memory_store = QdrantVectorStore(client=client, collection_name=MEMORY_COLLECTION, embedding=embeddings)
    return _memory_store


def remember_turn(question: str, answer: str) -> None:
    """Call after each completed turn."""
    _get_memory_store().add_texts([f"Q: {question}\nA: {answer}"])


def recall_relevant_turns(new_question: str, k: int = 3) -> list[str]:
    """Call before/inside the supervisor so it can use earlier context."""
    try:
        hits = _get_memory_store().similarity_search(new_question, k=k)
        return [h.page_content for h in hits]
    except Exception:
        return []  # no memory yet on the very first turn
