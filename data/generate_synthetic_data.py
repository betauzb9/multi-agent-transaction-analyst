"""
Generates a synthetic transaction dataset for FIN-01 (Automatic Transaction Categorization).

This is a STAND-IN so the whole pipeline (ML training + the multi-agent analyst's SQL agent)
runs end-to-end with zero downloads and zero cost. Before final submission, swap this for a
real public dataset (see docs/methodology.md, section 1) and re-run
`src/ml/train_categorizer.py`.

Usage:
    python data/generate_synthetic_data.py
Produces:
    data/transactions.csv   -- for ML training (text, amount, type, category)
    data/company.db         -- SQLite DB for the Data (SQL) agent, read-only in the app
"""
import random
import sqlite3
import csv
import os
from datetime import datetime, timedelta

random.seed(42)

MERCHANTS = {
    "Groceries": ["WALMART", "KROGER", "WHOLE FOODS", "ALDI", "TRADER JOES", "SAFEWAY"],
    "Dining & Coffee": ["STARBUCKS", "MCDONALDS", "CHIPOTLE", "SUBWAY", "LOCAL DINER", "PANERA"],
    "Transport & Fuel": ["UBER", "LYFT", "SHELL OIL", "CHEVRON", "CITY TRANSIT", "EXXON"],
    "Utilities & Bills": ["CITY ELECTRIC CO", "AT&T WIRELESS", "COMCAST CABLE", "WATER AUTHORITY"],
    "Rent & Housing": ["GREENFIELD PROPERTY MGMT", "PARKSIDE APARTMENTS", "MORTGAGE SERVICING CO"],
    "Entertainment & Subscriptions": ["NETFLIX", "SPOTIFY", "AMC THEATERS", "DISNEY PLUS"],
    "Shopping & Retail": ["AMAZON", "TARGET", "BEST BUY", "H&M", "IKEA"],
    "Health & Pharmacy": ["CVS PHARMACY", "WALGREENS", "CITY MEDICAL CLINIC", "AETNA COPAY"],
    "Travel": ["DELTA AIRLINES", "MARRIOTT HOTELS", "AIRBNB", "EXPEDIA"],
    "Financial Services": ["BANK ATM WITHDRAWAL", "WIRE TRANSFER FEE", "LOAN PAYMENT CO"],
    "Income & Transfers": ["PAYROLL DEPOSIT ACME CORP", "ZELLE TRANSFER RECEIVED", "VENMO CASHOUT"],
    "Education": ["STATE UNIVERSITY TUITION", "CAMPUS BOOKSTORE", "COURSERA INC"],
}

TXN_TYPES = ["debit", "credit", "pos", "ach", "transfer"]

NOISE_PREFIXES = ["SQ *", "TST* ", "POS ", ""]
NOISE_SUFFIXES = [" #4471", " #9981 SEATTLE WA", " STORE 0231", " REF9834", ""]

AMOUNT_RANGES = {
    "Groceries": (15, 180), "Dining & Coffee": (4, 65), "Transport & Fuel": (8, 90),
    "Utilities & Bills": (30, 220), "Rent & Housing": (700, 2200), "Entertainment & Subscriptions": (5, 60),
    "Shopping & Retail": (10, 400), "Health & Pharmacy": (8, 300), "Travel": (60, 1800),
    "Financial Services": (2, 50), "Income & Transfers": (200, 4000), "Education": (50, 6000),
}


def make_description(merchant: str) -> str:
    return f"{random.choice(NOISE_PREFIXES)}{merchant}{random.choice(NOISE_SUFFIXES)}"


def generate(n_per_category: int = 120):
    rows = []
    start = datetime(2025, 1, 1)
    for category, merchants in MERCHANTS.items():
        lo, hi = AMOUNT_RANGES[category]
        for _ in range(n_per_category):
            merchant = random.choice(merchants)
            desc = make_description(merchant)
            amount = round(random.uniform(lo, hi), 2)
            txn_type = "credit" if category == "Income & Transfers" else random.choice(TXN_TYPES[:-1])
            date = start + timedelta(days=random.randint(0, 364))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "description": desc,
                "amount": amount,
                "txn_type": txn_type,
                "category": category,
            })
    random.shuffle(rows)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "description", "amount", "txn_type", "category"])
        w.writeheader()
        w.writerows(rows)


def write_sqlite(rows, path):
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            txn_type TEXT NOT NULL,
            category TEXT NOT NULL
        )
    """)
    cur.executemany(
        "INSERT INTO transactions (date, description, amount, txn_type, category) VALUES (?,?,?,?,?)",
        [(r["date"], r["description"], r["amount"], r["txn_type"], r["category"]) for r in rows],
    )
    # A dedicated read-only role is created for the app user at deploy time (see README);
    # SQLite itself has no per-user GRANT system, so the app enforces read-only at the query layer
    # (src/agents/data_sql.py rejects any statement that isn't a SELECT).
    conn.commit()
    conn.close()


if __name__ == "__main__":
    rows = generate()
    here = os.path.dirname(__file__)
    write_csv(rows, os.path.join(here, "transactions.csv"))
    write_sqlite(rows, os.path.join(here, "company.db"))
    print(f"Generated {len(rows)} synthetic transactions -> data/transactions.csv, data/company.db")
