"""
Trains the transaction categorization model (FIN-01 core ML deliverable).

Text: char n-gram TF-IDF over the normalized merchant description (handles unseen merchants
reasonably, see docs/methodology.md §3).
Numeric: log-scaled amount + one-hot txn_type, concatenated with the text features.
Model: multinomial Logistic Regression, class_weight="balanced" (helps rare categories).

Usage:
    python -m src.ml.train_categorizer
Outputs:
    models/categorizer.joblib   -- the trained pipeline (load with src/ml/predict.py)
    models/eval_report.json     -- held-out evaluation metrics
"""
import json
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from src.ml.features import normalize_description, log_amount

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "..", "data", "transactions.csv")
MODEL_DIR = os.path.join(HERE, "..", "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def build_pipeline() -> Pipeline:
    text_features = ColumnTransformer(
        transformers=[
            ("desc_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2),
             "description_norm"),
            ("amount", FunctionTransformer(log_amount, validate=False), ["amount"]),
            ("txn_type", OneHotEncoder(handle_unknown="ignore"), ["txn_type"]),
        ]
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("features", text_features), ("clf", clf)])


def main():
    df = pd.read_csv(DATA_PATH)
    df["description_norm"] = df["description"].apply(normalize_description)

    X = df[["description_norm", "amount", "txn_type"]]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    eval_report = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_test": len(y_test),
        "per_class": report,
    }

    joblib.dump(pipe, os.path.join(MODEL_DIR, "categorizer.joblib"))
    with open(os.path.join(MODEL_DIR, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"Held-out accuracy: {acc:.3f} | macro-F1: {macro_f1:.3f} | n_test={len(y_test)}")
    print("Saved model -> models/categorizer.joblib")
    print("Saved eval report -> models/eval_report.json")


if __name__ == "__main__":
    main()
