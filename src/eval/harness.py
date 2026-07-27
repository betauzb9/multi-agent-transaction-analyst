"""
F11 — Evaluation harness: runs the multi-agent graph over a fixed test set
(src/eval/testset.json, >=10 questions) and scores it with RAGAS metrics + an LLM judge.

Usage:
    python -m src.eval.harness
Outputs:
    src/eval/eval_results.json  -- per-question scores + a summary metrics table
"""
import json
import os

from src.graph import ask
from src.llm import get_llm

HERE = os.path.dirname(__file__)
TESTSET_PATH = os.path.join(HERE, "testset.json")
RESULTS_PATH = os.path.join(HERE, "eval_results.json")


def llm_judge_score(question: str, reference: str, answer: str) -> dict:
    """Scores 1-5 vs a reference answer, since a general-purpose LLM judge is more portable
    than RAGAS's retrieval-context-shaped metrics for a mixed SQL+doc+code answer."""
    llm = get_llm()
    from pydantic import BaseModel, Field

    class Score(BaseModel):
        score: int = Field(description="1-5, how well the answer matches the reference in substance")
        justification: str

    result = llm.with_structured_output(Score).invoke(
        f"Question: {question}\nReference answer: {reference}\nSystem answer: {answer}\n"
        f"Score 1-5 how well the system answer matches the reference in substance "
        f"(ignore wording differences)."
    )
    return {"score": result.score, "justification": result.justification}


def run_ragas(questions, answers, contexts, references):
    """Optional — only runs if `ragas` is installed; the harness still works without it."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers, "contexts": contexts, "reference": references,
        })
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        return scores.to_pandas().mean(numeric_only=True).to_dict()
    except ImportError:
        return {"note": "ragas not installed — run `pip install ragas datasets` to enable this metric"}


def main():
    with open(TESTSET_PATH) as f:
        testset = json.load(f)

    per_question = []
    questions, answers, contexts, references = [], [], [], []

    for i, item in enumerate(testset, 1):
        print(f"[{i}/{len(testset)}] {item['question']}")
        try:
            result = ask(item["question"])
            judge = llm_judge_score(item["question"], item["reference"], result["answer"])
            per_question.append({
                "question": item["question"],
                "answer": result["answer"],
                "trace": result["steps"],
                "llm_judge": judge,
            })
            questions.append(item["question"])
            answers.append(result["answer"])
            contexts.append(result["documents"] or [""])
            references.append(item["reference"])
        except Exception as e:
            # One bad question (e.g. a recursion-limit hit) must not lose the other 11 results.
            print(f"  -> FAILED: {e}")
            per_question.append({
                "question": item["question"],
                "answer": None,
                "trace": None,
                "llm_judge": {"score": 0, "justification": f"Run failed: {e}"},
                "error": str(e),
            })

        # Save after every question so a later crash never loses earlier progress.
        with open(RESULTS_PATH, "w") as f:
            json.dump({"per_question": per_question}, f, indent=2)

    scored = [p for p in per_question if p["llm_judge"]["score"] > 0 or "error" not in p]
    n_failed = sum(1 for p in per_question if "error" in p)
    avg_judge_score = (
        sum(p["llm_judge"]["score"] for p in per_question) / len(per_question)
        if per_question else 0
    )
    ragas_scores = run_ragas(questions, answers, contexts, references) if questions else {"note": "skipped — no successful runs"}

    summary = {
        "n_questions": len(testset),
        "n_failed": n_failed,
        "avg_llm_judge_score_out_of_5": round(avg_judge_score, 2),
        "ragas": ragas_scores,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "per_question": per_question}, f, indent=2)

    print("\n=== Evaluation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
