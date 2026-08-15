"""
Shared feature-engineering functions, imported (never run as __main__) by both
train_categorizer.py and predict.py, so pickled sklearn objects resolve consistently.
"""
import re
import numpy as np
import pandas as pd

# Payment-system prefixes as they appear in real Payme/Click/Uzcard/Humo/P2P statement text.
POS_PREFIXES = re.compile(r"^(PAYME \*|CLICK |UZCARD POS |HUMO |P2P |SQ \*|TST\* |POS )")
# Branch/city/reference noise commonly appended by local payment systems, plus generic
# store-number/reference noise. Stripped so the same merchant normalizes the same way
# regardless of which branch or reference code was attached.
STORE_NOISE = re.compile(
    r"(#\d+|STORE \d+|REF\d+|\bFILIALI\b|\bTOSHKENT\b|\bCHILONZOR\b|\bYUNUSOBOD\b|\b[A-Z]{2}\b$)"
)


def normalize_description(text: str) -> str:
    """Normalize merchant text while preserving the brand token (see docs/methodology.md §2)."""
    t = text.strip().upper()
    t = POS_PREFIXES.sub("", t)
    t = STORE_NOISE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def log_amount(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[["amount"]].values)
