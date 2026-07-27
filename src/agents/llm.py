"""
Single place that builds the LLM and embeddings clients, so every agent uses the same
config (F1). Default provider is Gemini — free, no card (see docs/methodology.md / README).
"""
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src import config


def get_llm(temperature: float = 0.0):
    config.require_google_key()
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )


def get_embeddings():
    config.require_google_key()
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )


def get_text(response) -> str:
    """Normalizes an LLM response's .content into a plain string.

    Newer Gemini models (3.x) can return .content as a list of blocks
    (e.g. {"type": "thinking", ...} + {"type": "text", "text": "..."}) instead
    of a single string, which breaks any code calling .strip() directly on it.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)
