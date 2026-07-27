"""
F6 — Code agent: writes and runs Python for calculations/aggregation the LLM itself is
unreliable at (exact math), and can call the trained categorizer directly for "what category
would this new transaction get" questions.

Sandboxed: restricted builtins, no imports beyond an explicit allow-list, and a hard runtime
cap via a subprocess timeout — never execute model-written code unrestricted on the bare
server (see README's "Watch out" for F6).
"""
import io
import contextlib
import multiprocessing

from src.state import AgentState
from src.llm import get_llm, get_text

RUNTIME_CAP_SECONDS = 5

# Only these names are reachable inside the sandboxed exec — the model cannot import os,
# subprocess, sockets, etc. `categorize` lets the code agent call the trained ML model
# (src/ml/predict.py) directly, which is the bridge between the multi-agent layer and the
# FIN-01 core deliverable.
def _safe_globals():
    from src.ml.predict import categorize
    return {"__builtins__": {"print": print, "len": len, "range": range, "round": round,
                              "sum": sum, "min": min, "max": max, "float": float, "int": int,
                              "str": str, "list": list, "dict": dict}, "categorize": categorize}


def _run_in_subprocess(code: str, queue: multiprocessing.Queue):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, _safe_globals())
        queue.put(buf.getvalue())
    except Exception as e:
        queue.put(f"Code execution error: {e}")


def run_sandboxed(code: str) -> str:
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run_in_subprocess, args=(code, queue))
    proc.start()
    proc.join(timeout=RUNTIME_CAP_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return f"Code execution error: exceeded {RUNTIME_CAP_SECONDS}s runtime cap."
    return queue.get() if not queue.empty() else "(no output)"


def code_agent(state: AgentState) -> dict:
    llm = get_llm()
    code = get_text(llm.invoke(
        "Write Python to answer this question. You may call the pre-defined function "
        "categorize(description: str, amount: float, txn_type: str) -> dict if the question "
        "is about classifying a new transaction. Only use print(), len(), range(), round(), "
        "sum(), min(), max(). No imports. print() the final result.\n"
        f"Question: {state['question']}"
    ))
    # strip markdown fences if the model added them
    code = code.strip().strip("`")
    if code.lower().startswith("python"):
        code = code[len("python"):].strip()

    output = run_sandboxed(code)
    return {
        "code_result": output,
        "steps": state["steps"] + ["code"],
    }
