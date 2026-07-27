# Limitations, Risks & Next Steps — FIN-01

## Limitations

- **Synthetic reference data.** The bundled dataset is generated, not real; class balance and
  merchant-name diversity are simplified relative to real-world data. A real public dataset
  must be swapped in before this is a genuine capstone submission (see `methodology.md` §1).
- **No true out-of-vocabulary guarantee.** Character n-grams help with unseen merchants that
  share sub-strings with training data, but a genuinely novel merchant chain with no overlap
  will still fall back to `Other / Uncategorized` rather than being correctly labeled.
- **Amount/type features are weak signals alone.** They help disambiguate but are not
  sufficient by themselves to categorize; text remains the primary signal.
- **No personally identifiable or credential data is used or stored** — only merchant text,
  amount, and transaction type, consistent with FIN-01 §7's constraint.
- **Multi-agent layer adds latency and cost.** Every question may involve several LLM calls
  (supervisor → agent → critic, possibly a revision loop). A step/recursion budget bounds this,
  but it is materially slower than a single classifier call.

## Risks

- **Silent category drift.** New merchants or spending patterns over time can degrade accuracy
  without visible errors; the evaluation harness (F11) should be re-run periodically.
- **Critic false-approval.** The critic is itself an LLM and can approve an ungrounded answer;
  it reduces but does not eliminate incorrect answers.
- **Fallback bucket overuse.** If the confidence threshold is set too high, too many
  transactions land in `Other / Uncategorized`, reducing the usefulness of spending summaries.

## Recommended next steps

1. Replace the synthetic dataset with a real public dataset and re-run `train_categorizer.py`.
2. Add MCC (merchant category code) metadata where available — it is a much stronger signal
   than text alone when present.
3. Track macro-F1 on rare categories over time as new merchants appear.
4. Add a human-in-the-loop review queue for transactions repeatedly falling into
   `Other / Uncategorized`, to seed future training data.
