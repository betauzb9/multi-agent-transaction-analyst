# FIN-01 — Multi-Agent Transaction Analyst

A **multi-agent AI analyst** (per the *Multi-Agent AI Analyst — Project Guide & Rubric*)
applied to the **FIN-01 FinTech capstone scenario**: automatic transaction categorization.

- The **ML core** (`src/ml/`) is the FIN-01 deliverable: a trained classifier that assigns a
  spending category to a raw transaction (merchant text + amount + type).
- The **multi-agent layer** (`src/agents/`, `src/graph.py`) is a supervisor-led team of agents
  that can answer natural-language analyst questions about those categorized transactions —
  e.g. *"How many transactions were categorized as Dining & Coffee, and why do users spend
  there?"* — by combining a **SQL query** (the count) with **retrieved methodology docs**
  (the "why") and a **critic** that verifies the combined answer before it's shown.

## How the two documents map together

| Guide feature | FIN-01 application |
|---|---|
| F2 Ingestion & vector store | Ingests `docs/` (taxonomy, methodology, limitations) |
| F3 Retriever agent | Answers "why/how does the model decide" questions |
| F4 Web agent | Answers questions outside our docs/database (optional, needs a Tavily key) |
| F5 Data (SQL) agent | Queries `data/company.db`, the categorized-transactions table |
| F6 Code agent | Runs exact math/aggregation, or calls the trained classifier directly on a new transaction |
| F7 Supervisor | Routes each question to the right specialist(s) |
| F8 Critic | Rejects an answer that states a number not in the SQL/code evidence |
| F10 Memory | Recalls earlier turns for follow-ups ("...and last quarter?") |
| F11 Evaluation | RAGAS + LLM-judge over the 12-question test set in `src/eval/testset.json` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # or your preferred env manager
pip install -r requirements.txt
cp .env.example .env                                   # then fill in YOUR OWN GOOGLE_API_KEY
```

**Get your own free keys** (never share one key across a group — rate limits are per account):
- `GOOGLE_API_KEY` — required. https://aistudio.google.com/apikey (no card)
- `TAVILY_API_KEY` — optional, web agent. https://tavily.com (no card)
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — optional, tracing. https://cloud.langfuse.com (no card)

### Build order (follow in this order — each phase depends on the last)

```bash
# Phase 1 — Foundation
python data/generate_synthetic_data.py     # or swap in a real public dataset first — see docs/methodology.md §1
python -m src.ml.train_categorizer         # trains + evaluates the categorizer (already tested: 99.7% held-out accuracy)
python -m src.ingestion                    # embeds docs/ into Qdrant (embedded, no signup)

# Phase 2 — Specialist agents (test each alone)
python -c "from src.ml.predict import categorize; print(categorize('SQ *STARBUCKS #4471', 5.75, 'pos'))"
python -c "from src.agents.retriever import retriever_agent; from src.state import initial_state; print(retriever_agent(initial_state('What happens with low-confidence predictions?')))"

# Phase 3 — Full multi-agent graph
python -m src.graph "How many transactions were categorized as Dining & Coffee?"

# Phase 4 — Evaluation
python -m src.eval.harness                 # RAGAS + LLM-judge over 12 questions

# Phase 5 — Frontend + deploy
python app.py                              # add share=True in app.py for a free public link (~72h, no card)
```

## Project structure

```
data/            synthetic dataset generator (swap for a real public dataset before submission)
docs/            taxonomy / methodology / limitations — ingested by the retriever AND the FIN-01 documentation deliverable
models/          trained categorizer + eval report (created by train_categorizer.py)
src/
  config.py      F1 — reads all keys from .env
  state.py       F1 — shared AgentState
  llm.py         LLM + embeddings factory (Gemini)
  ingestion.py   F2 — builds the Qdrant vector store from docs/
  ml/
    features.py            shared normalization/feature functions
    train_categorizer.py   trains + evaluates the FIN-01 classifier
    predict.py             inference + confidence-based fallback
  agents/
    retriever.py   F3
    web.py         F4 (optional, skips gracefully without a key)
    data_sql.py    F5 — read-only text-to-SQL over data/company.db
    code_agent.py  F6 — sandboxed Python execution, can call the trained classifier
    supervisor.py  F7 (+ generate_answer)
    critic.py      F8
  graph.py       F9 — LangGraph wiring; ask() runs one full question
  memory.py      F10 — long-term memory over past turns
  eval/
    testset.json   12 evaluation questions
    harness.py      F11 — RAGAS + LLM-judge
  observability.py F12 — optional Langfuse tracing
app.py           F13/F14 — Gradio streaming frontend + easiest no-card deploy path
```

## What's already verified to work (no API key needed)

- `data/generate_synthetic_data.py` — generates 1,440 synthetic transactions across 12 categories
- `src/ml/train_categorizer.py` — trains in seconds, **99.7% held-out accuracy, 99.7% macro-F1**
- `src/ml/predict.py` — correctly classifies known merchants with high confidence, and falls
  back to `Other / Uncategorized` on an unrecognized merchant instead of guessing
- `src/agents/code_agent.py`'s sandbox — runs normal code, **blocks disallowed imports**
  (`import os` fails), and **enforces the runtime cap** (an infinite loop is killed at 5s)

Everything under `src/agents/`, `src/graph.py`, `src/ingestion.py`, `src/memory.py`,
`src/eval/harness.py` and `app.py` depends on `langchain`/`langgraph`/`gradio`/`qdrant-client`,
which need `pip install -r requirements.txt` plus your own `GOOGLE_API_KEY` to run — wire them
up locally following the build order above.

## Before you submit

1. Replace the synthetic dataset with a real public dataset (see `docs/methodology.md` §1) and
   re-run `train_categorizer.py`.
2. Run the full build order above end-to-end and capture: the supervisor graph screenshot, a
   frontend trace screenshot, a Langfuse trace, and the eval metrics table (see the guide's
   "Required visuals" and "Submission checklist" sections).
3. Do the "3 wrong questions" error analysis (see the guide's "Error analysis" section) —
   for each, say which agent failed (mis-routed, wrong SQL, code error, missed retrieval, or
   critic let a bad answer through) and one fix.
4. Fill in `docs/limitations.md` and `docs/methodology.md` with your real dataset's specifics.
