# AI Transaction Analyst (Multi-Agent Transaction Analyst)

**Ask plain-language questions about your business transactions — a team of 4 specialist AI agents finds the answer, fact-checks it, and hands it back to you.**

🔗 **Live demo:** [multi-agent-transaction-analyst.onrender.com](https://multi-agent-transaction-analyst.onrender.com/)


📦 **Code (GitHub):** [github.com/betauzb9/multi-agent-transaction-analyst](https://github.com/betauzb9/multi-agent-transaction-analyst)

> ⏳ **Note:** this project runs on Render's free tier. If the service has been
> idle for a few minutes it "spins down" — the first request may take
> 30–60 seconds to wake it back up; requests after that are fast.

---

## What is this?

An AI analyst that lets a small/medium business owner ask **natural-language**
questions about their bank/Payme/Click transactions — in Uzbek, Russian, or
English, the system detects the language automatically. For example:

- *"How many transactions fall under 'Dining & Coffee'?"*
- *"What spending categories does this system use?"*
- *"What category would a transaction described as 'CLICK KORZINKA #4471' for 85,000 so'm get?"*

Unlike a plain chatbot, this isn't one big AI call — it's a team of specialist
agents, each handling its own part of the job, with a final answer that goes
through a dedicated **quality-check (Critic)** agent before it's ever shown
to the user.

## How it works

```
User question
        │
        ▼
   Supervisor — reads the question, decides which specialist(s) are needed
        │
        ├──▶ Knowledge Search (Retriever)  — pulls from methodology/category docs
        ├──▶ Database Query (SQL agent)    — counts and aggregates real transactions
        ├──▶ Calculation Engine (Code agent) — exact math, or classifies a new transaction
        └──▶ Web Research (Web agent)      — answers anything outside the local data
        │
        ▼
   Draft answer written from everything gathered so far
        │
        ▼
   Quality Check (Critic) — checks the draft against the evidence, rejects unsupported claims
        │
        ▼
   Final answer — in whichever language the question was asked in
```

The Supervisor can call more than one specialist in sequence when needed (e.g.
pull a number from the database, then add a "why" from the knowledge base), and
if the Critic rejects a draft, the loop runs again before anything is shown.

## Key features

- **Multilingual:** ask in Uzbek, Russian, or English — the answer comes back
  in the same language, even though the underlying documentation is in Uzbek.
- **Localized to the Uzbek market:** the sample transaction data mirrors
  Payme/Click/Uzcard/Humo/P2P statement formats and real Uzbek merchants
  (Korzinka, Yandex Go, Beeline, etc.), with amounts in so'm (UZS).
- **Automatic categorization model:** a trained ML model classifies new/unseen
  transaction text into one of 12 categories (or "Other/Uncategorized" when
  confidence is too low).
- **Every answer is fact-checked:** a separate Critic agent verifies the final
  answer against the gathered evidence before it's shown.
- **Live agent trace:** the UI shows which agent is working and what it's doing
  in real time — not a black box.
- **Read-only by design:** the SQL agent only ever runs `SELECT` queries — there
  is no way for it to write to or modify the database.

## Tech stack

| Layer | Technology |
|---|---|
| Multi-agent orchestration | LangGraph |
| LLM | Google Gemini |
| Vector search (RAG) | Qdrant |
| ML categorization | scikit-learn (TF-IDF + Logistic Regression) |
| Database | SQLite |
| Frontend | Gradio |
| Observability | Langfuse |
| Deployment | Render |

## Project structure

```
├── app.py                        # Gradio frontend (chat UI, live agent trace)
├── data/
│   ├── generate_synthetic_data.py   # Uzbek-market-realistic sample data generator
│   ├── transactions.csv             # Transactions used for ML training
│   └── company.db                   # SQLite DB the SQL agent queries
├── models/
│   ├── categorizer.joblib           # Trained categorization model
│   └── eval_report.json             # Model accuracy report
├── docs/
│   ├── category_taxonomy.md         # The 12 categories and the reasoning behind them
│   ├── methodology.md               # How the model was trained, normalization approach
│   └── limitations.md               # Known limitations and next steps
├── src/
│   ├── graph.py                     # LangGraph agent graph
│   ├── agents/                      # Supervisor, Retriever, SQL, Code, Web, Critic
│   ├── ml/                          # Features, training, prediction
│   └── ingestion.py                 # Loads docs into Qdrant (for RAG)
└── requirements.txt
```

## Running locally

```bash
# 1. Set up the environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Open .env and fill in GOOGLE_API_KEY (required), plus TAVILY_API_KEY and
# LANGFUSE_* (optional)

# 3. Prepare the data and model
python data/generate_synthetic_data.py
python -m src.ml.train_categorizer
python -m src.ingestion

# 4. Run the app
python app.py
```

Open `http://localhost:7860` in your browser.

## Plugging in real business transactions

By default the system runs on synthetic (but realistic-looking) data. To
connect your own real transactions instead:

1. Export a CSV from your bank/Payme/Click account.
2. Map the columns to `date, description, amount, txn_type, category`
   (`category` can be left blank — the model will predict it).
3. Replace `data/transactions.csv` and `data/company.db` with the real data.
4. Re-run `python -m src.ml.train_categorizer` and check the accuracy in
   `models/eval_report.json`.
5. Push the changes to GitHub — Render will redeploy automatically (see below).

Details: `docs/methodology.md` §1.

## Deploying on Render

The project is hosted on [Render](https://render.com). To redeploy:

**Build Command:**
```bash
pip install -r requirements.txt && python data/generate_synthetic_data.py && python -m src.ml.train_categorizer && python -m src.ingestion
```

**Start Command:**
```bash
python app.py
```

**Environment Variables (Render dashboard → Environment):**
| Variable | Required | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Used for Gemini LLM and embedding calls |
| `TAVILY_API_KEY` | Optional | Used by the Web agent for internet search |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Optional | Used for observability |
| `PYTHONUNBUFFERED` | Recommended | Set to `1` so build logs stream live — otherwise `print()` output gets buffered, making it much harder to see where a stuck build actually is |

> 💡 If a deploy freezes for a long time, check `GOOGLE_API_KEY` first — the
> `src.ingestion` step calls the Gemini embedding API, and an invalid or
> expired key can cause it to hang for a long time without ever printing
> an error.

## Limitations

Known limitations of this system (synthetic-data characteristics, handling of
out-of-vocabulary merchants, etc.) are documented in detail in `docs/limitations.md`.
