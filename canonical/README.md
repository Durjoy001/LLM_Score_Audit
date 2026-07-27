# Canonical AI score sets — provenance & handling

**Verified: 2026-07-27**

This folder is the **pinned, authoritative** source of AI scores for the paper.
It exists because `src/data/` is **gitignored** (untracked) and the scorer is
**not deterministic**, which together caused a silent score-drift incident (see
below). These files are checked into git so they cannot drift again.

## Files

- `canonical_ai_scores_dataset1.json` — the 12 biotech/semiconductor proposals
  (p1–p8, pA–pD). Per proposal: 5 dimension scores, overall, confidence,
  OR-logic verdict, and the raw `invest_score_echo` / `qa_score_echo`.
- `canonical_kickstarter_scores.json` — the 141 Kickstarter proposals. Per
  proposal: 5 dimensions, overall, confidence, verdict, and `funded_pct`.
- `CANONICAL_RESULTS_SUMMARY.md` — the reconciled OLD-vs-NEW analysis the paper
  is written from.

## Provenance

- **Dataset1 source:** the **Jun-16 Stage 6/7 run** (`INVEST_WEIGHT=0.90`), captured in
  `results/Dataset1/rerun_stage6_7_iw90.log` and
  `results/Dataset1/evaluation/evaluation_report.xlsx`. The two sources were
  **cross-validated to a max per-dimension difference of 0.0005**, and the set
  **reproduces every published number** in
  `results/Dataset1/evaluation/evaluation_summary.json` (ρ=0.5894, verdict
  accuracy=0.8333, all per-dimension κ/ρ/ICC).
- **Confidence** comes from `src/data/refined_answers/<pid>/postproc/metrics.json`
  (Stage-5 output, mtime Jun-16 — never regenerated, so it did not drift).
- **Verdict rule (canonical):** `GO iff overall ≥ 0.725 OR confidence ≥ 0.811`
  (the OR-logic rule recorded in `evaluation_summary.json`). GO→`Y`, HOLD/NO-GO→`N`.
  The `verdict_ai` column in `evaluation_report.xlsx` holds *stale raw report
  verdicts* and must NOT be used.
- **Kickstarter source:** the `src/data/reports` Kickstarter cache (2026-07-15),
  verified consistent with the frozen `results/Kickstarter` eval (no post-eval
  re-run → no drift). Max AI overall score = **0.640** (kt49).

## Handling rules going forward

1. **`src/data/reports` is NOT authoritative.** It was re-run on Jun-18 and its
   scores differ from the paper's frozen Jun-16 set due to LLM non-determinism
   (same code, `temperature=0`, `seed=42`, identical input → different scores).
   See `../resultsNew/pA_fix_data_drift.md` and `CANONICAL_RESULTS_SUMMARY.md`
   §"Data integrity".
2. Cite **these files** for any AI score used in the paper.
3. If the pipeline is ever re-run, treat the output as a *new sample*, not a
   correction of these numbers, unless the model snapshot is pinned and
   determinism is independently re-verified.
