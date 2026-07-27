"""
F9 — Supervisor graph: wires supervisor -> specialist -> back to supervisor -> ... -> generate
-> critic -> finish or revise. A recursion limit (config.RECURSION_LIMIT) plus a revisions cap
(config.MAX_REVISIONS) guarantee the graph always terminates (watch-out in the guide for F9).
"""
from langgraph.graph import StateGraph, END

from src.state import AgentState, initial_state
from src.agents.retriever import retriever_agent
from src.agents.web import web_agent
from src.agents.data_sql import data_agent
from src.agents.code_agent import code_agent
from src.agents.supervisor import supervisor, generate_answer
from src.agents.critic import critic
from src.memory import remember_turn
from src.observability import get_callbacks
from src import config


def route_after_supervisor(state: AgentState) -> str:
    return state["plan"]


def route_after_critic(state: AgentState) -> str:
    approved = state["steps"][-1].endswith("(ok)")
    if approved or state["revisions"] >= config.MAX_REVISIONS:
        return "finish"
    return "revise"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor)
    g.add_node("retriever", retriever_agent)
    g.add_node("web", web_agent)
    g.add_node("data", data_agent)
    g.add_node("code", code_agent)
    g.add_node("generate", generate_answer)
    g.add_node("critic", critic)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"retriever": "retriever", "web": "web", "data": "data", "code": "code", "finish": "generate"},
    )
    for agent_name in ["retriever", "web", "data", "code"]:
        g.add_edge(agent_name, "supervisor")

    g.add_edge("generate", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"finish": END, "revise": "supervisor"})

    return g.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def ask(question: str) -> AgentState:
    """Runs one full question through the multi-agent graph and stores the turn in memory."""
    app = get_app()
    result = app.invoke(
        initial_state(question),
        config={"recursion_limit": config.RECURSION_LIMIT, "callbacks": get_callbacks()},
    )
    remember_turn(question, result["answer"])  # F10 — so follow-ups can recall this turn
    return result


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How many transactions were categorized as Dining & Coffee?"
    result = ask(q)
    print("ANSWER:", result["answer"])
    print("TRACE:", result["steps"])
