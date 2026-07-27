"""
F7 — Supervisor / Router: the manager agent. Reads the question and what's been collected so
far, and decides which specialist runs next, or that the answer is ready ("finish").
"""
from pydantic import BaseModel, Field

from src.state import AgentState
from src.llm import get_llm, get_text
from src.memory import recall_relevant_turns


class Route(BaseModel):
    next: str = Field(description="one of: retriever, web, data, code, finish")


ROUTING_GUIDE = """You are the supervisor of a multi-agent transaction-categorization analyst.
Specialists:
- retriever: answers questions about the category taxonomy, methodology, or limitations (docs)
- web: answers questions needing information outside our docs/database
- data: answers "how many / what fraction / which category" questions via SQL over the
  transactions database (already-categorized transactions)
- code: answers exact-math/aggregation questions, or "what category would transaction X get"
  (calls the trained classifier directly)
Choose 'finish' once enough evidence has been collected to answer the question.
Never re-route to an agent that has already run for this question unless truly necessary.
"""


MAX_SUPERVISOR_HOPS = 6  # hard safety valve — guarantees termination even if the LLM never says "finish"


def supervisor(state: AgentState) -> dict:
    hops_so_far = sum(1 for s in state["steps"] if s.startswith("supervisor->"))
    if hops_so_far >= MAX_SUPERVISOR_HOPS:
        # Hard stop: answer with whatever evidence exists rather than looping forever
        # (see README's F9 "Watch out" — a mis-routing loop must never run unbounded).
        return {
            "plan": "finish",
            "steps": state["steps"] + ["supervisor->finish(hop-limit)"],
        }

    llm = get_llm()
    past_turns = recall_relevant_turns(state["question"])  # F10 — long-term memory
    visited_agents = sorted({
        s.split("->", 1)[1] for s in state["steps"]
        if s.startswith("supervisor->") and "->" in s
    })
    decision = llm.with_structured_output(Route).invoke(
        f"{ROUTING_GUIDE}\nRelevant past turns (may be empty): {past_turns}\n"
        f"Question: {state['question']}\nCollected so far: {state['steps']}\n"
        f"Agents already used for this question: {visited_agents or 'none yet'}\n"
        f"Next agent, or 'finish'."
    )
    return {
        "plan": decision.next,
        "steps": state["steps"] + [f"supervisor->{decision.next}"],
    }


def generate_answer(state: AgentState) -> dict:
    """Drafts the answer from whatever evidence has been collected, for the critic to check."""
    llm = get_llm()
    past_turns = recall_relevant_turns(state["question"])  # F10 — helps resolve follow-ups
    evidence = (
        f"Documents: {state['documents']}\n"
        f"SQL result: {state['sql_result']}\n"
        f"Code result: {state['code_result']}\n"
        f"Relevant past turns (may be empty): {past_turns}"
    )
    draft = get_text(llm.invoke(
        f"Question: {state['question']}\nEvidence:\n{evidence}\n"
        f"Write a concise, grounded answer using only this evidence. If evidence is missing "
        f"for part of the question, say so explicitly rather than guessing."
    ))
    return {"answer": draft, "steps": state["steps"] + ["generate"]}