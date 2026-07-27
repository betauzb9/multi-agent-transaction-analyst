"""
F12 — Observability: Langfuse tracing of every agent step, tool call, token and cost.
Optional — the whole app works without it (see README's free-services table).
"""
from src import config


def get_callbacks() -> list:
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        # langfuse isn't installed — tracing is optional, so skip instead of crashing.
        return []

    # langfuse v3+: keys are configured on a Langfuse client once; CallbackHandler()
    # then picks up that active client automatically (no key args on the handler itself).
    Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
    )
    return [CallbackHandler()]
