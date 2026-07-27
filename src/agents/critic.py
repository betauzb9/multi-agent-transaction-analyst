"""
F8 — Critic / Verifier: checks the drafted answer against the gathered evidence.
Approves, or forces a revision (routed back to the supervisor by the graph, F9).
"""
from pydantic import BaseModel, Field

from src.state import AgentState
from src.llm import get_llm


class Verdict(BaseModel):
    ok: bool = Field(description="True only if the answer is correct AND fully supported by evidence")
    reason: str


def critic(state: AgentState) -> dict:
    llm = get_llm()
    verdict = llm.with_structured_output(Verdict).invoke(
        f"Question: {state['question']}\n"
        f"Evidence — documents: {state['documents']} | sql: {state['sql_result']} | "
        f"code: {state['code_result']}\n"
        f"Drafted answer: {state['answer']}\n"
        f"Is this answer correct AND fully supported by the evidence above? "
        f"Reject if it states a number not present in the SQL/code evidence, or claims a "
        f"policy not present in the documents."
    )
    return {
        "revisions": state["revisions"] + (0 if verdict.ok else 1),
        "steps": state["steps"] + [f"critic({'ok' if verdict.ok else 'revise: ' + verdict.reason})"],
    }
