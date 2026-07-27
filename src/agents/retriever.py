"""
F3 — Retriever agent: RAG over docs/ (taxonomy, methodology, limitations).
Answers "why" / "how does the model decide" style questions.
"""
from src.state import AgentState
from src.ingestion import get_vectorstore

_vectorstore = None


def _vs():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    return _vectorstore


def retriever_agent(state: AgentState) -> dict:
    docs = _vs().as_retriever(search_kwargs={"k": 4}).invoke(state["question"])
    return {
        "documents": state["documents"] + [d.page_content for d in docs],
        "steps": state["steps"] + ["retriever"],
    }
