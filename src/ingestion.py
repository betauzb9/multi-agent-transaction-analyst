"""
F2 — Ingestion & vector store: load the docs/ folder (methodology, taxonomy, limitations) →
chunk → embed → store in Qdrant (embedded mode, no signup needed).

Usage:
    python -m src.ingestion
"""
import glob
import os

from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src import config
from src.llm import get_embeddings

DOCS_GLOB = os.path.join(os.path.dirname(__file__), "..", "docs", "*.md")


def load_docs() -> list[Document]:
    docs = []
    for path in glob.glob(DOCS_GLOB):
        with open(path, "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": os.path.basename(path)}))
    if not docs:
        raise FileNotFoundError(f"No documents found at {DOCS_GLOB}")
    return docs


def build_vectorstore() -> QdrantVectorStore:
    """Chunks the docs, embeds them, and (re)builds the Qdrant collection."""
    docs = load_docs()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)

    embeddings = get_embeddings()
    # embedding dimension must match the Qdrant collection size (watch-out in the guide) —
    # we probe it once instead of hardcoding a number that could drift with model changes.
    probe_dim = len(embeddings.embed_query("dimension probe"))

    client = QdrantClient(path=config.QDRANT_PATH)
    if not client.collection_exists(config.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=probe_dim, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(client=client, collection_name=config.QDRANT_COLLECTION, embedding=embeddings)
    vectorstore.add_documents(chunks)
    return vectorstore


def get_vectorstore() -> QdrantVectorStore:
    """Reconnects to an already-built Qdrant collection (used by the retriever agent)."""
    embeddings = get_embeddings()
    client = QdrantClient(path=config.QDRANT_PATH)
    if not client.collection_exists(config.QDRANT_COLLECTION):
        raise RuntimeError("Vector store not built yet. Run: python -m src.ingestion")
    return QdrantVectorStore(client=client, collection_name=config.QDRANT_COLLECTION, embedding=embeddings)


if __name__ == "__main__":
    vs = build_vectorstore()
    hits = vs.similarity_search("What happens with low-confidence merchant predictions?", k=3)
    print(f"Ingested docs into Qdrant collection '{config.QDRANT_COLLECTION}'.")
    print(f"Sanity check — top hit source: {hits[0].metadata.get('source')}")
