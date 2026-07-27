"""
F1 — Shared state: one AgentState flows through every node in the graph.
"""
from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    question: str               # the user's question
    plan: str                   # supervisor's routing decision for this hop
    documents: List[str]        # chunks gathered by retriever/web agents
    sql_result: Optional[str]   # result from the data (SQL) agent
    code_result: Optional[str]  # result from the code agent
    answer: str                 # the drafted / final answer
    steps: List[str]            # trace of every node visited, e.g. ["supervisor->data", "data(sql)"]
    revisions: int              # how many times the critic has sent the answer back


def initial_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        plan="",
        documents=[],
        sql_result=None,
        code_result=None,
        answer="",
        steps=[],
        revisions=0,
    )
