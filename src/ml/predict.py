"""
Inference interface for the transaction categorizer (FIN-01 functional requirement §7:
"accept a new transaction record and return a spending category").

Also implements the confidence-based fallback (docs/methodology.md §4): low-confidence
predictions are returned as "Other / Uncategorized" instead of a possibly-wrong label.

CLI usage:
    python -m src.ml.predict "SQ *STARBUCKS #4471" 5.75 pos
Library usage:
    from src.ml.predict import categorize
    categorize("SQ *STARBUCKS #4471", 5.75, "pos")
"""
import os
import sys
import joblib
import pandas as pd

from src.ml.features import normalize_description

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "..", "models", "categorizer.joblib")

CONFIDENCE_THRESHOLD = 0.55
FALLBACK_LABEL = "Other / Uncategorized"

_model = None


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Run: python -m src.ml.train_categorizer"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def categorize(description: str, amount: float, txn_type: str) -> dict:
    """Returns {"category": str, "confidence": float, "candidates": [(label, prob), ...]}."""
    model = _load_model()
    row = pd.DataFrame([{
        "description_norm": normalize_description(description),
        "amount": amount,
        "txn_type": txn_type,
    }])
    probs = model.predict_proba(row)[0]
    labels = model.named_steps["clf"].classes_
    ranked = sorted(zip(labels, probs), key=lambda x: -x[1])
    top_label, top_prob = ranked[0]

    if top_prob < CONFIDENCE_THRESHOLD:
        return {
            "category": FALLBACK_LABEL,
            "confidence": float(top_prob),
            "candidates": [(l, float(p)) for l, p in ranked[:3]],
        }
    return {
        "category": top_label,
        "confidence": float(top_prob),
        "candidates": [(l, float(p)) for l, p in ranked[:3]],
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python -m src.ml.predict "<description>" <amount> <txn_type>')
        sys.exit(1)
    desc, amount, txn_type = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    result = categorize(desc, amount, txn_type)
    print(result)
