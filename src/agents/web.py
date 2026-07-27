"""
F4 — Web agent: Tavily search for questions outside the local docs/database (e.g. "what do
other banks use as a spending category taxonomy"). Skips gracefully if TAVILY_API_KEY is unset
— this is optional (see docs/methodology.md and the README's free-services table).
"""
from src.state import AgentState
from src import config


def web_agent(state: AgentState) -> dict:
    if not config.TAVILY_API_KEY:
        # Graceful skip — no key configured. Done-when for F4 requires this exact behavior.
        return {"steps": state["steps"] + ["web(skipped-no-key)"]}

    from tavily import TavilyClient

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    hits = client.search(state["question"])["results"]
    return {
        "documents": state["documents"] + [h["content"] for h in hits],
        "steps": state["steps"] + ["web"],
    }
