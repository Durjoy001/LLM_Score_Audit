# LLM Score Audit — Canonical Results Summary

**Compiled 2026-07-27. Paper-of-record numbers.** All AI scores come from the
canonical Jun-16 set (`canonical_ai_scores_dataset1.json`,
`canonical_kickstarter_scores.json`), which reproduces the published paper numbers
exactly (ρ=0.5894, verdict accuracy=0.8333). The stale Jun-18 `src/data` cache is
**not** used. Ground truth: OLD = single expert rater (as published); NEW = median/
mean/majority consensus of **3** expert raters.

> **Overarching caveat (applies to every number below):** the Dataset1 panel is
> **n = 12 proposals**. All per-dimension κ/ρ and the overall ρ are small-sample
> statistics with wide, overlapping confidence intervals; treat directional
> patterns as suggestive, not significant. The robust, defensible claims are the
> aggregate ones (§3 human ceiling, §5 Kickstarter). Individual per-dimension
> deltas should not be over-interpreted.

---

## 1. OLD (single-rater, published) vs NEW (3-rater consensus)

Non-unanimous consensus Funded verdicts (majority-voted): **p5, pD**.

### Per-dimension AI-vs-human agreement

| Dimension | Weighted κ OLD | Weighted κ NEW | Spearman ρ OLD | Spearman ρ NEW |
|---|---|---|---|---|
| team | 0.3239 | 0.3846 | 0.3982 | 0.3508 |
| objective | −0.1379 | 0.0 | −0.2748 | *n/a (constant)* |
| strategy | −0.125 | 0.3333 | 0.3236 | 0.3862 |
| advantages | −0.08 | 0.2258 | 0.0612 | 0.1472 |
| feasibility | 0.25 | 0.05 | 0.1561 | 0.042 |

`objective` NEW ρ is undefined: the median-consensus objective score is **4 for all
12 proposals** (zero variance), so Spearman cannot be computed — itself a
by-product of the raters' tight agreement on that dimension.

### Headline scalars

| Metric | OLD (paper) | NEW consensus |
|---|---|---|
| Overall Spearman ρ | **0.5894** | **0.1747** (rounded ranks) / 0.050 (unrounded) |
| Overall ICC(2,1) | 0.1188 | 0.1357 |
| Pairwise rank concordance | 0.7586 | 0.5769 |
| Verdict accuracy (OR-logic) | 0.8333 | 0.8333 |

Interpretation: against the 3-rater panel the AI's rank agreement **collapses from
ρ≈0.59 to ρ≈0.05–0.17** — i.e. the single-rater 0.589 was optimistic; a more
reliable panel shows the AI barely tracks the human ranking. Verdict accuracy is
identical (0.833) but this is **coincidental** — the OR-logic AI GO-set {p3, p7}
happens to sit inside the consensus GO-set; the two misses are pA and pD. It is not
evidence the AI matches human go/no-go judgement.

---

## 2. Human–human reliability (the new inter-rater baseline)

| Target | ICC(2,1) [single] | ICC(2,k) [panel] | mean pairwise wκ | Gwet's AC1 |
|---|---|---|---|---|
| team | 0.875 | 0.9545 | 0.8671 | 0.8436 |
| objective | −0.0476 | −0.1579 | −0.0303 | 0.8759 |
| strategy | 0.7944 | 0.9206 | 0.7808 | 0.7778 |
| advantages | 0.7885 | 0.9179 | 0.7745 | 0.7838 |
| feasibility | 0.8553 | 0.9466 | 0.8421 | 0.8476 |
| overall ranking | 0.8586 | 0.9480 | — | — |

The three experts agree strongly (ICC(2,1) ≈ 0.79–0.88 on most dimensions;
ICC(2,k) ≈ 0.92–0.95). **`objective` is the kappa/ICC paradox case:** near-zero
ICC/κ but **AC1 = 0.876**, because the marginals are highly skewed (nearly all
scores are 4) — low variance deflates κ/ICC even though raters overtly agree. This
is exactly why AC1 is reported alongside κ, and it is the honest reading of that
dimension.

---

## 3. AI vs the human leave-one-rater-out (LORO) band — **headline new finding**

Each human rater's agreement with the panel of the *other two* defines a human
band; the AI's agreement with the full panel is placed against it.

| Target | AI-vs-consensus ρ | Human LORO band | AI position |
|---|---|---|---|
| team | 0.3508 | [0.8625, 0.927] | **below** human floor |
| objective | *undefined* | [−0.091, −0.091] | undefined |
| strategy | 0.3862 | [0.8173, 0.9129] | **below** human floor |
| advantages | 0.1472 | [0.7977, 0.9087] | **below** human floor |
| feasibility | 0.042 | [0.8926, 0.94] | **below** human floor |
| overall ranking | 0.050 | [0.9184, 0.9521] | **below** human floor |

**On every scorable target the AI's agreement with the panel falls below the
weakest human's agreement with the panel** — a clean, robust statement of the
model's ceiling relative to human experts. (`objective` is degenerate for both AI
and humans because consensus is constant.)

---

## 4. Corrected INVEST_WEIGHT calibration sweep

Spearman ρ (AI overall vs human rank) as INVEST_WEIGHT sweeps 0→1, on canonical
invest/qa echoes:

| INVEST_WEIGHT | ρ OLD (single-rater) | ρ NEW (consensus) |
|---|---|---|
| … | … | … |
| 0.85 | 0.5958 | 0.0213 |
| **0.90 (production)** | **0.5894** | 0.0496 |
| 0.95 | 0.5716 | **0.0956 ← NEW peak** |
| **1.00** | **0.6001 ← OLD peak** | 0.0673 |

- **Production INVEST_WEIGHT = 0.90 does NOT tie the peak.** For the single rater
  the peak is at IW=1.0 (0.6001); 0.90 gives 0.5894 and 0.85 gives 0.5958 — 0.90 is
  ~0.01 below the best. The earlier "0.90 ties the peak" claim was an artifact of
  the stale cache. The empirical justification for 0.90 is therefore **weak**, not
  a tie.
- For the consensus panel the whole curve is near zero (peak 0.096 at IW=0.95); no
  weight makes the AI track the panel well.

Verdict-threshold search (max accuracy, OR-logic): OLD smallest τ_r = 0.6624,
NEW = 0.6881, τ_c = 0.8112 in both. **Full table:** `resultsNew/Dataset1/calibration_sweep_consensus.json`.

---

## 5. Kickstarter "threshold transfers with 0% recall" — **confirmed, unaffected**

- Max AI overall score across all **141** Kickstarter proposals = **0.640** (kt49);
  max confidence = 0.661.
- Canonical consensus decision threshold τ_r ∈ **[0.688, 0.731]**, τ_c ≈ 0.811 —
  **every value stays above the 0.640 score ceiling.**
- Under the production threshold **and** both consensus thresholds: **0 GO
  predictions, 0% recall** on the 24 real successes.
- **The finding holds unchanged.** The Kickstarter cache did **not** drift (all 141
  reports pre-date the frozen eval; no post-eval re-run), so this result is on
  canonical scores throughout. Detail: `resultsNew/Kickstarter/threshold_propagation.json`.

---

## 6. Data integrity — two issues found and fixed

Two defects were discovered while rebuilding against the 3-rater ground truth. Both
are **fixed**; documented here so you can decide footnote vs. appendix.

**(a) Proposal-identity mis-join (fixed in code).** `evaluate_cohens_kappa._find_record`
matched proposal A (`A_安海半导体…v1.4-YF.pdf`) to `kt4__openai` — a *Kickstarter*
proposal whose source file `4.pdf` produced a degenerate 1-character match key
`"4"` that substring-matched `v1.**4**`. Fix: reject degenerate keys, prefer exact
matches, warn on any fuzzy fallback. All 12 proposals now resolve correctly (pA =
0.681). This bug only bit the *current* cache, not the published `results/Dataset1`.
Diagnostic: `resultsNew/pA_fix_data_drift.md`; fix in `src/tools/evaluate_cohens_kappa.py`.

**(b) Score-cache drift (root-caused; canonical set pinned).** `src/data/reports`
is gitignored and the scorer is non-deterministic. A Jun-18 re-run of Stage 6/7
produced different invest scores for 11/12 proposals vs the Jun-16 run that the
paper froze — **same code, `temperature=0`, `seed=42`, byte-identical input**
(OpenAI's seed is best-effort; likely a `gpt-4o-mini` snapshot change). Separately,
the verdict rule changed in git (commit `3e84844`, AND→OR logic) between the runs.
Net: the live cache is a *different sample*, not a correction. Canonical Jun-16
scores are now frozen and git-tracked in this folder (`README.md` for provenance).

---

## 7. Things I am NOT fully confident in (read before citing)

1. **p7's GO is razor-thin (0.726 vs 0.725, margin 0.001).** Given the scorer
   non-determinism, that single GO could flip to HOLD on any re-run. Any claim that
   leans on p7 being GO is fragile.
2. **Small n (=12).** Every per-dimension κ/ρ and the overall ρ have wide CIs.
   The sign of individual dimensions (e.g. strategy NEW ρ=0.386 vs OLD 0.324) is not
   robust. Only aggregate patterns (§3, §5) should carry weight.
3. **NEW overall ρ has two values** (0.175 with integer-rounded consensus ranks per
   the pipeline convention; 0.050 with un-rounded mean ranks). Both are weakly
   positive; do not quote a precise magnitude — quote "≈0.05–0.17, far below human."
4. **`objective` is unusable for AI-vs-human ρ** (constant consensus). Report it via
   AC1 only, and note the zero-variance reason.
5. **Verdict-accuracy parity (0.833 = 0.833) is coincidental**, not a sign the AI
   matches human go/no-go. It rides on the base rate and a 2-proposal GO set.
6. The **0.640 vs 0.725 gap** and the **human-ceiling finding (§3)** are the two
   results I am most confident in and would build the paper's claims around.

---

### File map
- Dataset1 canonical scores: `canonical/canonical_ai_scores_dataset1.json`
- Kickstarter canonical scores: `canonical/canonical_kickstarter_scores.json`
- Consensus eval: `resultsNew/Dataset1/evaluation/evaluation_summary_consensus.json`
- LORO: `resultsNew/Dataset1/evaluation/loro_spearman.json`
- Reliability: `resultsNew/Dataset1/rater_reliability.json`
- Calibration: `resultsNew/Dataset1/calibration_sweep_consensus.json`
- Kickstarter threshold: `resultsNew/Kickstarter/threshold_propagation.json`
- OLD-vs-NEW table: `resultsNew/summary_old_vs_new.md`
- Drift diagnostic: `resultsNew/pA_fix_data_drift.md`
