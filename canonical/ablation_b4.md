# B4 — Ablating the heuristic rules

**Date:** 2026-07-29 | **LLM calls:** 96 (cap ablation only; components 2–4 are pure re-derivation)  
**Prompt variants:** canonical/b4_prompt_variants.json (A sha 182a50db…, B sha 27977547…, C sha 926d894b…)  
**Ground truth:** Dataset1 = 3-rater panel consensus (n=12); Kickstarter = `funded_pct ≥ 100`, a real outcome (n=141)

---

## Component 1 — Remove the "AI-powered" innovation cap

Three conditions, both providers, 2 runs per cell:

| condition | prompt | meaning |
|---|---|---|
| **A** | sha `182a50db…` (9197 ch) | production prompt, cap present |
| **B** | sha `27977547…` (8633 ch) | pre-built variant — removes RULE 1 + HARD CAP line only |
| **C** | sha `926d894b…` (8485 ch) | **genuine ablation** — also removes the worked example restating the cap, the `(INCLUDES most AI proposals)` steer, and the dangling `DO NOT apply the I-C cap` clause |

> **Why C exists.** B's own provenance records that cap language survives in it — most
> critically the EXAMPLES bullet `→ AI heart failure decision support system (AI IS the
> product) → I-C, score 0.42-0.56`, which restates the cap verbatim *and* describes p1's
> exact archetype. B ablates the rule's statement while leaving an instruction that applies
> the same rule to the test case. A null result under B would be uninterpretable.

### Cap-bound cohort under condition A

| provider | I-C any run | cap actually fires (I-C **and** score in 0.42–0.56) |
|---|---|---|
| openai | pC | **pC** (1/12) |
| gemini | p1, p2, p6, p8, pB, pC | **p1, p2, p6, p8, pC** (5/12) |

### Effect of removing the cap (mean Δ vs condition A)

| provider | group | n | Δ innovation (B−A) | Δ innovation (C−A) | Δ overall (C−A) |
|---|---|---|---|---|---|
| openai | cap-bound | 1 | 0.1000 | **0.1000** | 0.0561 |
| openai | not cap-bound | 11 | 0.0182 | **0.0205** | 0.0044 |
| gemini | cap-bound | 5 | 0.0070 | **0.0490** | -0.0004 |
| gemini | not cap-bound | 7 | -0.0121 | **-0.0336** | -0.0226 |

Noise floor for reference: within-provider innovation max range **0.045**, overall MAD **0.0056**, overall max range **0.0374**.

### Per-proposal innovation score, cap-bound proposals only

| provider | pid | A | B | C | Δ C−A | category A → C |
|---|---|---|---|---|---|---|
| openai | pC | 0.500 | 0.600 | 0.600 | **+0.100** | I-C → I-C/I-C |
| gemini | p1 | 0.480 | 0.475 | 0.535 | **+0.055** | I-C/I-C → I-C/I-C |
| gemini | p2 | 0.520 | 0.520 | 0.520 | **+0.000** | I-C/I-C → I-C/I-C |
| gemini | p6 | 0.520 | 0.520 | 0.665 | **+0.145** | I-C/I-C → I-C/I-B |
| gemini | p8 | 0.480 | 0.520 | 0.525 | **+0.045** | I-C/I-C → I-C/I-C |
| gemini | pC | 0.450 | 0.450 | 0.450 | **+0.000** | I-C/I-C → I-C/I-C |

**Findings.**

1. **The cap binds for exactly one OpenAI proposal (pC) and five Gemini proposals.** The
   B4 pre-flight that reported an empty cohort was measuring OpenAI's Jun-16 run, where
   only pC reaches I-C. The rule is not inert — it is *rarely triggered* under gpt-4o-mini.
2. **Removing the cap raises the capped score, and the effect clears the noise floor.**
   OpenAI pC: 0.500 → 0.600 (+0.100 vs 0.045 floor), overall +0.0561 vs 0.0374 floor.
3. **B and C diverge for Gemini, exactly as predicted.** On the cap-bound cohort B moves
   innovation by +0.007 (nothing) while C moves it by +0.049. p1 is the clearest case:
   0.480 → 0.475 under B, → 0.535 under C. **The pre-built variant would have produced a
   false null.** For OpenAI B and C agree, because pC is not the archetype named in the
   surviving example.
4. **Caveat — the ablation is not perfectly surgical.** Gemini p3 (not cap-bound) swings
   −0.255 on innovation under C, dropping I-A → I-C/I-B. Editing the INNOVATION section
   perturbs anchoring for non-AI proposals too, so the causal claim should be limited to
   the cap-bound cohort, where the direction is consistent.

---

## Component 2 — Verdict rule ablation

Thresholds held at production values: τ_r = 0.725, τ_c = 0.811.

### Dataset1 (n=12, truth = 3-rater majority verdict, base rate 0.333)

| arm | rule | accuracy | precision | recall | F1 | n predicted GO |
|---|---|---|---|---|---|---|
| openai_canonical | `score_only` | 0.750 | 1.000 | 0.250 | 0.400 | 1 |
| openai_canonical | `confidence_only` | 0.750 | 1.000 | 0.250 | 0.400 | 1 |
| openai_canonical | `score_AND_confidence` | 0.667 | n/a | 0.000 | n/a | 0 |
| openai_canonical | `score_OR_confidence` | 0.833 | 1.000 | 0.500 | 0.667 | 2 |
| gemini_run1 | `score_only` | 0.583 | 0.333 | 0.250 | 0.286 | 3 |
| gemini_run1 | `confidence_only` | 0.750 | 1.000 | 0.250 | 0.400 | 1 |
| gemini_run1 | `score_AND_confidence` | 0.667 | n/a | 0.000 | n/a | 0 |
| gemini_run1 | `score_OR_confidence` | 0.667 | 0.500 | 0.500 | 0.500 | 4 |
| gemini_run2 | `score_only` | 0.750 | 0.667 | 0.500 | 0.571 | 3 |
| gemini_run2 | `confidence_only` | 0.750 | 1.000 | 0.250 | 0.400 | 1 |
| gemini_run2 | `score_AND_confidence` | 0.750 | 1.000 | 0.250 | 0.400 | 1 |
| gemini_run2 | `score_OR_confidence` | 0.750 | 0.667 | 0.500 | 0.571 | 3 |

### Kickstarter (n=141, truth = funded ≥ 100%, base rate 0.1702)

Max AI overall score = **0.64**, below τ_r = 0.725.

| rule | accuracy | recall | n predicted GO | TP/FP/FN/TN |
|---|---|---|---|---|
| `score_only` | 0.8298 | 0.000 | 0 | 0/0/24/117 |
| `confidence_only` | 0.8298 | 0.000 | 0 | 0/0/24/117 |
| `score_AND_confidence` | 0.8298 | 0.000 | 0 | 0/0/24/117 |
| `score_OR_confidence` | 0.8298 | 0.000 | 0 | 0/0/24/117 |

**Findings.**

1. On Dataset1 the production OR rule is the best of the four for OpenAI (0.833) but
   **not** for Gemini, where score-only and confidence-only both beat it in run 1. The
   OR rule's advantage is scorer-specific, not structural.
2. The AND rule collapses to zero predicted GO for OpenAI — it is strictly worse than
   either conjunct alone. The AND→OR change in commit `3e84844` was load-bearing.
3. **On Kickstarter all four rules are numerically identical** — accuracy 0.8298, recall
   0.000, zero predicted GO. Every rule is inert because no proposal clears either
   threshold. Four structurally different decision rules are indistinguishable from
   thresholded accuracy. This is the setup for component 4.

---

## Component 3 — Vary the 0.90/0.10 blend

### Dataset1

| arm | ρ @ IW=0.90 | peak ρ | peak IW | production ties peak | distinct IW across LOO folds | production picked |
|---|---|---|---|---|---|---|
| openai_canonical | 0.1747 | 0.2184 | [0.75] | False | [0.15, 0.6, 0.65, 0.75, 0.95] | 0% |
| gemini_run1 | 0.5604 | 0.5823 | [0.8] | False | [0.5, 0.6, 0.8] | 0% |
| gemini_run2 | 0.6187 | 0.6187 | [0.9, 0.95, 1.0] | True | [0.6, 0.9] | 83% |

### Kickstarter (n=125, 16 excluded; weights: equal (1.0) — per-proposal Stage-5 weights not recorded for Kickstarter)

ρ @ IW=0.90 = **0.3686**, AUC @ IW=0.90 = **0.8186**; peak ρ = 0.3741 @ IW=[1.0].

Nested 5-fold CV (inner loop selects IW on train, outer evaluates on held-out fold):

| fold | inner-selected IW | train ρ | test ρ @ selected | test ρ @ 0.90 | test AUC @ selected | test AUC @ 0.90 |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 0.3381 | 0.5375 | 0.5340 | 0.8788 | 0.8636 |
| 1 | 1.0 | 0.3548 | 0.3860 | 0.3787 | 0.8631 | 0.8571 |
| 2 | 1.0 | 0.4301 | 0.0291 | 0.0245 | 0.7065 | 0.6957 |
| 3 | 1.0 | 0.3115 | 0.5718 | 0.5708 | 0.7262 | 0.7262 |
| 4 | 1.0 | 0.4296 | 0.2367 | 0.2585 | 0.8690 | 0.8810 |

Mean held-out ρ at inner-selected IW = **0.3522**; at production IW=0.90 = **0.3533**.

**Findings.**

1. **IW=0.90 is not an optimum and is not stable.** On Dataset1 it never ties the peak for
   OpenAI, and LOO resampling selects five different IW values across twelve folds —
   production is chosen in 0% of them. At n=12 the blend weight is unidentifiable.
2. **On Kickstarter the objective is flat.** ρ moves from 0.348 to 0.374 across IW ∈
   [0.6, 1.0] and AUC from 0.809 to 0.820. Any weight in that range is statistically
   indistinguishable.
3. **Nested CV shows tuning buys nothing.** The inner loop selects IW=1.00 in all five
   folds, yet held-out ρ at the selected weight (0.3522) is *marginally worse* than at the
   untuned production weight (0.3533). Selecting the blend on training data does not
   generalise — it is fitting a flat surface.
4. **The QA component carries almost no signal.** At IW=0.00 (pure QA) ρ = −0.038 and
   AUC = 0.521 — chance. All discriminative content is in the investment score. The
   0.90/0.10 blend is therefore a presentational choice, not a tuned parameter.

---

## Component 4 — Score validity vs threshold validity

### Dataset1: the two orderings invert

| arm | score validity ρ | score validity AUC | thresholded accuracy | recall |
|---|---|---|---|---|
| openai_canonical | 0.1747 | 0.6250 | 0.8333 | 0.5000 |
| gemini_run1 | 0.5604 | 0.8125 | 0.6667 | 0.5000 |
| gemini_run2 | 0.6187 | 0.8438 | 0.7500 | 0.5000 |

- Ranking by **score validity**: gemini_run2 > gemini_run1 > openai_canonical
- Ranking by **thresholded accuracy**: openai_canonical > gemini_run2 > gemini_run1
- Orderings agree: **False** — they are exactly reversed.

### Kickstarter (n=141, real outcome): the sharpest case

| quantity | value |
|---|---|
| score validity — ρ vs `funded_pct` | **0.3659** |
| score validity — AUC (funded vs not) | **0.7644** |
| max AI overall score | 0.64 (τ_r = 0.725) |
| thresholded accuracy, **all four rules** | 0.8298 |
| recall, all four rules | 0.000 |
| negative base rate | 0.8298 |

**The finding.** The AI score ranks Kickstarter outcomes with **AUC 0.764** — genuinely
informative. Thresholded at the production τ_r it predicts **zero** GO, yielding accuracy
0.8298 that is *exactly* the negative base rate and recall 0.000. A reader looking only at
83% accuracy would conclude the system works; a reader looking at recall would conclude it
is useless; both would miss that the underlying score has real discriminative power the
threshold discards.

On Dataset1 the same distinction appears as an inversion: OpenAI has the **worst** score
validity (ρ=0.175, AUC=0.625) but the **best** thresholded accuracy (0.833), while Gemini
run 2 has the best score validity (ρ=0.619, AUC=0.844) and lower accuracy (0.750).

**Thresholded accuracy is not a proxy for score validity — on these data it is
anti-correlated with it.** Any evaluation that reports only verdict accuracy is blind to
whether the scorer ranks proposals well, and that is precisely the distinction the paper
argues for.

---

## Consolidated verdict on the heuristics

| heuristic | status after B4 |
|---|---|
| "AI-powered" innovation cap | **Fires rarely (1/12 OpenAI, 5/12 Gemini) and its effect is scorer-dependent.** Real but small; demote to a documented heuristic. |
| OR-logic verdict rule | **Best of four for OpenAI only.** Not structurally justified; scorer-specific. |
| τ_r = 0.725 / τ_c = 0.811 | **Inert on Kickstarter** — no proposal reaches either. Not transferable across datasets. |
| INVEST_WEIGHT = 0.90 | **Not an optimum, not stable, and tuning it does not generalise.** Any IW ∈ [0.6, 1.0] is equivalent. Demote to a presentational default. |

All four should be reported as **current heuristic settings**, not tuned parameters.
The evidence for demotion is now quantitative rather than an admission of time pressure.

---

## Files

- `canonical/ablation_b4.json` — all four components, full detail
- `canonical/b4_prompt_variants.json` — A/B/C prompt texts, shas, edit log, residual-marker check
- `canonical/b4_cap_ablation_raw.json` — 96 raw Call B invocations
- `canonical/b4_kickstarter_invest_qa.json` — recovered invest/qa echoes + validation
- Scripts: `build_b4_prompt_c.py`, `run_b4_cap_ablation.py`, `parse_kickstarter_invest_qa.py`, `analyze_b4.py`

Not modified: existing `canonical/` files, `results/`, `resultsNew/`, `src/data/`.
