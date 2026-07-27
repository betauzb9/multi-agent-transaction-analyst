# Methodology — Automatic Transaction Categorization (FIN-01)

## 1. Dataset

The reference implementation ships with a **synthetic transaction dataset**
(`data/generate_synthetic_data.py`) that mimics the structure of public transaction/merchant
categorization datasets (raw merchant text, amount, transaction type, category label). This
is a stand-in so the whole pipeline runs end-to-end with zero downloads and zero cost.

**Before submission, replace this with a real public dataset**, for example:
- Kaggle "Bank Transaction Categorization" style datasets
- Kaggle "Credit Card / Merchant Category Code" datasets
- Any public financial-text or merchant-classification dataset

Document in this file: the source URL, license, size, class balance, and any cleaning you did.

## 2. Preprocessing / normalization

Raw merchant text (e.g. `SQ *COFFEE HOUSE #4471 SEATTLE WA`) is normalized before modeling:

1. Uppercase → lowercase.
2. Strip POS prefixes (`SQ *`, `TST*`, `POS `), trailing store numbers, and state/city codes
   **without deleting the merchant brand token itself** — over-aggressive stripping is a common
   failure mode that destroys the signal the model needs (this directly answers Question #2 in
   the FIN-01 brief: normalize noise, keep signal).
3. Collapse repeated whitespace/punctuation.
4. Keep transaction amount and type as separate numeric/categorical features — they are cheap,
   available at prediction time, and help disambiguate merchants that span categories
   (Question #4 in the brief).

## 3. Model

- **Text features:** TF-IDF over character n-grams (3–5) on the normalized merchant string.
  Character n-grams (rather than word n-grams) are deliberately used because merchant strings
  are abbreviated, misspelled, and often not real words — this is what lets the model handle
  **previously unseen merchants** reasonably (Question #3): a new merchant that shares
  sub-string patterns with known ones ("STARBUCKS #4471" vs "STARBUCKS #9981") still gets a
  useful representation.
- **Numeric features:** log-scaled transaction amount, one-hot transaction type.
- **Classifier:** multinomial Logistic Regression (`class_weight="balanced"` to address rare
  categories — Question #5) with calibrated probabilities, chosen over a heavier model because
  it trains in seconds on CPU, is easy to explain/defend, and calibrated probabilities are what
  the confidence-based fallback (below) needs.

## 4. Low-confidence handling (fallback)

If `max(predicted_probabilities) < CONFIDENCE_THRESHOLD` (default `0.55`, tunable in
`src/ml/predict.py`), the prediction is returned as **`Other / Uncategorized`** with the
top-2 candidate categories attached, instead of forcing a possibly-wrong confident label.
This directly answers Question #6 in the FIN-01 brief.

## 5. Evaluation

- Stratified train/test split (held-out, never seen during training).
- Metrics: accuracy, macro-F1 (treats rare categories fairly — not swamped by "Groceries"),
  and per-class precision/recall/F1 (see `models/eval_report.json` after running
  `train_categorizer.py`).
- Rare categories are evaluated separately with macro-F1 rather than only overall accuracy,
  since accuracy alone hides poor performance on small classes.

## 6. How this connects to the multi-agent analyst layer

The trained classifier is the "ground truth" that populates the transactions database used by
the **Data (SQL) agent** (`src/agents/data_sql.py`) — so a business question like *"how many
transactions were categorized as Dining last month, and why do users spend there?"* is answered
by: SQL agent (the count) + Retriever agent (this methodology / taxonomy for the "why") +
Critic (checks the combined answer is grounded in both).
