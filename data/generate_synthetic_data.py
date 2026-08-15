"""
Generates an O'zbekiston (Uzbekistan) market-realistic transaction dataset for FIN-01
(Automatic Transaction Categorization).

This mimics real local payment-system SMS/statement text (Payme, Click, Uzcard, Humo, P2P
transfers) with amounts in UZS (so'm), so the whole pipeline (ML training + the multi-agent
analyst's SQL agent) runs end-to-end with zero downloads and zero cost, while already looking
like a real Uzbek business's transaction feed.

If you have an ACTUAL bank/Payme/Click export (CSV) for a real business, prefer that over this
generator — see docs/methodology.md §1 for the required columns (date, description, amount,
txn_type, category) and how to plug it in instead of running this script.

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

# Category names are in Uzbek since this is the language the business owner (and the deployed
# app's users) will see in SQL results, taxonomy docs, and generated answers.
MERCHANTS = {
    "Oziq-ovqat": [
        "KORZINKA", "MAKRO", "HAVAS SUPERMARKET", "UZUM MARKET", "CARREFOUR",
        "MEGA PLANET OZIQ-OVQAT", "DEHQON BOZORI",
    ],
    "Kafe va restoranlar": [
        "EVOS", "WIMPY", "CHOPAR PIZZA", "KFC", "CAFE CHINARA", "ANDIJON OSHXONASI",
        "COFFEE BEAN", "BRO CAFE",
    ],
    "Transport": [
        "YANDEX GO", "MYTAXI", "UZAUTO YO'L SOLIG'I", "GAZPROMNEFT AZS",
        "METRO TOSHKENT", "UZBEKISTON TEMIR YOLLARI", "UZGASOIL AZS",
    ],
    "Kommunal to'lovlar": [
        "TOSHKENT ELEKTR TARMOQLARI", "HUDUDGAZ TA'MINOT", "SUV TA'MINOTI MCHJ",
        "BEELINE", "UCELL", "UZMOBILE", "UZONLINE INTERNET",
    ],
    "Ijara va ko'chmas mulk": [
        "UY IJARASI TO'LOVI", "OFIS IJARASI TO'LOVI", "KO'CHMAS MULK AGENTLIGI",
    ],
    "Ko'ngilochar va obuna": [
        "UZUM TV", "NETFLIX", "SPOTIFY", "IMAX KINOTEATR", "YOUTUBE PREMIUM",
        "PLAYSTATION STORE",
    ],
    "Xarid va do'konlar": [
        "ZARA", "LC WAIKIKI", "MEGA PLANET SAVDO", "TASHKENT CITY MALL",
        "SAMSUNG STORE", "TEXNOMART",
    ],
    "Salomatlik va dorixona": [
        "OILAVIY DORIXONA", "ORIENT PHARM", "TIBBIYOT MARKAZI SOG'LOM AVLOD",
        "INVIVO LABORATORIYASI",
    ],
    "Sayohat": [
        "UZBEKISTON HAVO YO'LLARI", "HOTEL WYNDHAM TASHKENT", "BOOKING COM",
        "AVIACHIPTA KASSASI",
    ],
    "Moliyaviy xizmatlar": [
        "KOMISSIYA TO'LOVI", "KREDIT TO'LOVI", "BANKOMAT NAQD PUL YECHISH",
        "PLASTIK KARTA XIZMAT HAQI",
    ],
    "Ish haqi va o'tkazmalar": [
        "ISH HAQI TO'LOVI", "P2P PLASTIKDAN PLASTIKKA", "ZARABOTNAYA PLATA",
        "FREELANCE TO'LOVI",
    ],
    "Ta'lim": [
        "TDIU TO'LOV BO'LIMI", "IT PARK KURSLARI", "UDEMY ONLINE KURS",
        "MAKTABGACHA TA'LIM MARKAZI",
    ],
}

TXN_TYPES = ["click", "payme", "uzcard", "humo", "p2p"]

# Prefixes/suffixes mimic real Payme/Click/Uzcard SMS and statement noise.
NOISE_PREFIXES = ["PAYME *", "CLICK ", "UZCARD POS ", "HUMO ", "P2P ", ""]
NOISE_SUFFIXES = [
    " #4471", " TOSHKENT", " CHILONZOR FILIALI", " REF9834", " YUNUSOBOD", "",
]

# Amounts in UZS (so'm) — realistic ranges for a small/medium Uzbek business or household.
AMOUNT_RANGES = {
    "Oziq-ovqat": (20_000, 450_000),
    "Kafe va restoranlar": (15_000, 150_000),
    "Transport": (5_000, 120_000),
    "Kommunal to'lovlar": (50_000, 900_000),
    "Ijara va ko'chmas mulk": (1_500_000, 9_000_000),
    "Ko'ngilochar va obuna": (15_000, 200_000),
    "Xarid va do'konlar": (50_000, 3_000_000),
    "Salomatlik va dorixona": (20_000, 800_000),
    "Sayohat": (200_000, 8_000_000),
    "Moliyaviy xizmatlar": (3_000, 100_000),
    "Ish haqi va o'tkazmalar": (2_000_000, 20_000_000),
    "Ta'lim": (100_000, 15_000_000),
}


def make_description(merchant: str) -> str:
    return f"{random.choice(NOISE_PREFIXES)}{merchant}{random.choice(NOISE_SUFFIXES)}"


def generate(n_per_category: int = 150):
    rows = []
    start = datetime(2025, 1, 1)
    for category, merchants in MERCHANTS.items():
        lo, hi = AMOUNT_RANGES[category]
        for _ in range(n_per_category):
            merchant = random.choice(merchants)
            desc = make_description(merchant)
            amount = round(random.uniform(lo, hi), -2)  # round to nearest 100 so'm, like real receipts
            txn_type = "p2p" if category == "Ish haqi va o'tkazmalar" else random.choice(TXN_TYPES[:-1])
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
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    print(f"Generated {len(rows)} synthetic transactions (UZS, Payme/Click/Uzcard style) -> "
          f"data/transactions.csv, data/company.db")
