# Spending Category Taxonomy — FIN-01

This taxonomy answers "Questions You Must Resolve" #1 from the FIN-01 brief: a taxonomy
useful for user-facing spending summaries without being too granular for a text classifier
trained on short, noisy merchant strings.

## Top-level categories (12 + fallback)

| Category | Examples of merchant text |
|---|---|
| Groceries | WALMART, KROGER, WHOLE FOODS, ALDI |
| Dining & Coffee | MCDONALDS, STARBUCKS, CHIPOTLE, local restaurant names |
| Transport & Fuel | UBER, LYFT, SHELL, CHEVRON, transit authorities |
| Utilities & Bills | ELECTRIC CO, AT&T, COMCAST, water/sewer authorities |
| Rent & Housing | property management companies, mortgage servicers |
| Entertainment & Subscriptions | NETFLIX, SPOTIFY, cinemas, streaming services |
| Shopping & Retail | AMAZON, TARGET, clothing and electronics stores |
| Health & Pharmacy | CVS, WALGREENS, clinics, insurance co-pays |
| Travel | airlines, hotels, AIRBNB, booking platforms |
| Financial Services | bank fees, ATM withdrawals, transfers, loan payments |
| Income & Transfers | payroll deposits, P2P transfers received |
| Education | tuition, bookstores, online course platforms |
| **Other / Uncategorized** | low-confidence or unmatched merchant text (fallback bucket, see §3) |

## Design rationale

1. **Granularity.** 12 categories was chosen because it is coarse enough that a TF‑IDF /
   linear model trained on short merchant strings can achieve reasonable per-class recall,
   while still being useful for spending summaries (a 40+ category taxonomy fragments the
   training data per class and confuses end users).
2. **Mutually exclusive, mostly.** A few merchants are ambiguous by nature (e.g. Amazon sells
   groceries and electronics). These are resolved by amount-based heuristics and, where
   available, MCC (merchant category code) metadata rather than by the text model alone.
3. **Fallback bucket is mandatory.** "Other / Uncategorized" is not a modeling failure — it is
   a deliberate design decision so the system never forces a confident label onto an unseen or
   ambiguous merchant (see `limitations.md`, §"Low-confidence handling").

## Extending the taxonomy

New categories should only be added when there is enough labeled training volume (rule of
thumb: at least ~50 examples) to support them without starving existing classes.
