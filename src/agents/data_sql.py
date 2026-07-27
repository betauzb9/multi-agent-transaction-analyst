"""
F5 — Data (SQL) agent: text-to-SQL over data/company.db, the SQLite database of transactions
already labeled by the trained categorizer (src/ml/train_categorizer.py). Answers "how many /
what fraction / which category" style questions.

Read-only guard: any statement that isn't a SELECT is rejected before it ever reaches
db.run() — never let the model-written query DROP/DELETE/UPDATE anything (see README's
"Watch out" for F5).
"""
from langchain_community.utilities import SQLDatabase

from src.state import AgentState
from src.llm import get_llm, get_text
from src import config

_db = None


def _get_db() -> SQLDatabase:
    global _db
    if _db is None:
        _db = SQLDatabase.from_uri(f"sqlite:///{config.SQLITE_DB_PATH}")
    return _db


def data_agent(state: AgentState) -> dict:
    db = _get_db()
    llm = get_llm()
    sql = get_text(llm.invoke(
        f"Schema:\n{db.get_table_info()}\n"
        f"Write ONE SQLite SELECT query (SQL only, no explanation, no markdown fences) "
        f"that answers: {state['question']}"
    )).strip()

    # strip accidental markdown fences before validating
    sql = sql.strip("`").removeprefix("sql").strip()

    if not sql.lower().startswith("select"):
        return {
            "sql_result": "Query rejected: only read-only SELECT statements are allowed.",
            "steps": state["steps"] + ["data(sql-rejected)"],
        }

    try:
        result = db.run(sql)
    except Exception as e:  # surfaced to the critic rather than crashing the graph
        result = f"SQL execution error: {e}"

    return {
        "sql_result": f"{sql}\n→ {result}",
        "steps": state["steps"] + ["data(sql)"],
    }
