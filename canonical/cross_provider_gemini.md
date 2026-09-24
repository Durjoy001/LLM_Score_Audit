# Cross-provider replication check (paper fix B3) — OpenAI vs Gemini, Dataset1

**Model:** `gemini-3.5-flash-lite` (Flash-Lite (GA) -- tier match for gpt-4o-mini; not a Pro/flagship model)  
**Temperature:** 0.0 | **Thinking:** minimal | **Seed:** requested 42, reached provider: **False**  
**Date:** 2026-07-29 13:05:00 | **Runs:** 2 × 12 proposals = 24 Call B invocations  
**Evidence:** existing src/data/extracted/<pid>__openai/dimensions_v2.json (Jun-16); no re-extraction performed  
**Prompt SHA256:** `182a50db8e224a77c8856add2c9c6323…` (unmodified production prompt)

> **Seed caveat.** Gemini exposes no seed parameter. The OpenAI arm is nominally
> seeded (best-effort); the Gemini arm is **unseeded**. Run-to-run spread within
> Gemini is therefore an upper bound on what a seeded Gemini would show.

---

## 1. Per-proposal overall score

| pid | OpenAI | Gemini r1 | Δ r1 | Gemini r2 | Δ r2 | OA verdict | G r1 | G r2 |
|---|---|---|---|---|---|---|---|---|
| p1 | 0.6800 | 0.6725 | -0.0075 | 0.6674 | -0.0126 | N | N | N |
| p2 | 0.6900 | 0.6341 | -0.0559 | 0.6279 | -0.0621 | N | N | N |
| p3 | 0.7090 | 0.7202 | +0.0112 | 0.7472 | +0.0382 | Y | Y | Y |
| p4 | 0.6910 | 0.6653 | -0.0257 | 0.6795 | -0.0115 | N | N | N |
| p5 | 0.6580 | 0.7524 | +0.0944 | 0.7144 | +0.0564 | N | Y | N |
| p6 | 0.6850 | 0.6433 | -0.0417 | 0.6270 | -0.0580 | N | N | N |
| p7 | 0.7260 | 0.8007 | +0.0747 | 0.8182 | +0.0922 | Y | Y | Y |
| p8 | 0.6510 | 0.5671 | -0.0839 | 0.5966 | -0.0544 | N | N | N |
| pA | 0.6810 | 0.6801 | -0.0009 | 0.7054 | +0.0244 | N | N | N |
| pB | 0.7140 | 0.7918 | +0.0778 | 0.7908 | +0.0768 | N | Y | Y |
| pC | 0.6260 | 0.5585 | -0.0675 | 0.5602 | -0.0658 | N | N | N |
| pD | 0.6470 | 0.6967 | +0.0497 | 0.6933 | +0.0463 | N | N | N |

Mean overall: OpenAI **0.6798**, Gemini 1 **0.6819**, Gemini 2 **0.6857**.

## 1b. Per-dimension blended scores — gemini_run1

| pid | team (OA → Gem, Δ) | objective (OA → Gem, Δ) | strategy (OA → Gem, Δ) | advantages (OA → Gem, Δ) | feasibility (OA → Gem, Δ) |
|---|---|---|---|---|---|
| p1 | 0.644 → 0.734 (+0.090) | 0.685 → 0.820 (+0.135) | 0.643 → 0.643 (+0.000) | 0.735 → 0.492 (-0.243) | 0.689 → 0.689 (-0.000) |
| p2 | 0.728 → 0.701 (-0.027) | 0.631 → 0.631 (+0.000) | 0.686 → 0.668 (-0.018) | 0.674 → 0.512 (-0.162) | 0.727 → 0.664 (-0.063) |
| p3 | 0.824 → 0.824 (-0.000) | 0.666 → 0.639 (-0.027) | 0.691 → 0.646 (-0.045) | 0.735 → 0.825 (+0.090) | 0.640 → 0.667 (+0.027) |
| p4 | 0.688 → 0.706 (+0.018) | 0.644 → 0.644 (+0.001) | 0.597 → 0.615 (+0.018) | 0.781 → 0.691 (-0.090) | 0.729 → 0.666 (-0.063) |
| p5 | 0.646 → 0.826 (+0.180) | 0.688 → 0.796 (+0.108) | 0.606 → 0.714 (+0.107) | 0.780 → 0.735 (-0.045) | 0.577 → 0.703 (+0.126) |
| p6 | 0.690 → 0.735 (+0.045) | 0.635 → 0.662 (+0.027) | 0.682 → 0.637 (-0.044) | 0.730 → 0.523 (-0.207) | 0.685 → 0.667 (-0.018) |
| p7 | 0.814 → 0.841 (+0.027) | 0.682 → 0.790 (+0.108) | 0.633 → 0.768 (+0.135) | 0.772 → 0.818 (+0.045) | 0.725 → 0.788 (+0.063) |
| p8 | 0.637 → 0.547 (-0.090) | 0.677 → 0.632 (-0.045) | 0.635 → 0.590 (-0.045) | 0.726 → 0.483 (-0.243) | 0.588 → 0.588 (-0.000) |
| pA | 0.641 → 0.731 (+0.090) | 0.683 → 0.791 (+0.108) | 0.646 → 0.673 (+0.027) | 0.690 → 0.708 (+0.018) | 0.733 → 0.526 (-0.207) |
| pB | 0.731 → 0.866 (+0.135) | 0.639 → 0.819 (+0.180) | 0.684 → 0.792 (+0.108) | 0.777 → 0.669 (-0.108) | 0.729 → 0.819 (+0.090) |
| pC | 0.554 → 0.527 (-0.027) | 0.641 → 0.614 (-0.027) | 0.683 → 0.593 (-0.090) | 0.507 → 0.462 (-0.045) | 0.733 → 0.598 (-0.135) |
| pD | 0.640 → 0.730 (+0.090) | 0.583 → 0.709 (+0.126) | 0.640 → 0.685 (+0.045) | 0.733 → 0.733 (-0.000) | 0.636 → 0.636 (-0.000) |

## 1b. Per-dimension blended scores — gemini_run2

| pid | team (OA → Gem, Δ) | objective (OA → Gem, Δ) | strategy (OA → Gem, Δ) | advantages (OA → Gem, Δ) | feasibility (OA → Gem, Δ) |
|---|---|---|---|---|---|
| p1 | 0.644 → 0.707 (+0.063) | 0.685 → 0.820 (+0.135) | 0.643 → 0.643 (+0.000) | 0.735 → 0.492 (-0.243) | 0.689 → 0.689 (-0.000) |
| p2 | 0.728 → 0.728 (+0.000) | 0.631 → 0.604 (-0.027) | 0.686 → 0.668 (-0.018) | 0.674 → 0.512 (-0.162) | 0.727 → 0.637 (-0.090) |
| p3 | 0.824 → 0.851 (+0.027) | 0.666 → 0.684 (+0.018) | 0.691 → 0.646 (-0.045) | 0.735 → 0.870 (+0.135) | 0.640 → 0.685 (+0.045) |
| p4 | 0.688 → 0.733 (+0.045) | 0.644 → 0.644 (+0.001) | 0.597 → 0.642 (+0.045) | 0.781 → 0.691 (-0.090) | 0.729 → 0.684 (-0.045) |
| p5 | 0.646 → 0.826 (+0.180) | 0.688 → 0.688 (-0.000) | 0.606 → 0.696 (+0.089) | 0.780 → 0.735 (-0.045) | 0.577 → 0.640 (+0.063) |
| p6 | 0.690 → 0.735 (+0.045) | 0.635 → 0.635 (-0.000) | 0.682 → 0.611 (-0.071) | 0.730 → 0.523 (-0.207) | 0.685 → 0.640 (-0.045) |
| p7 | 0.814 → 0.859 (+0.045) | 0.682 → 0.817 (+0.135) | 0.633 → 0.786 (+0.153) | 0.772 → 0.845 (+0.072) | 0.725 → 0.788 (+0.063) |
| p8 | 0.637 → 0.664 (+0.027) | 0.677 → 0.632 (-0.045) | 0.635 → 0.608 (-0.027) | 0.726 → 0.483 (-0.243) | 0.588 → 0.606 (+0.018) |
| pA | 0.641 → 0.704 (+0.063) | 0.683 → 0.818 (+0.135) | 0.646 → 0.691 (+0.045) | 0.690 → 0.735 (+0.045) | 0.733 → 0.598 (-0.135) |
| pB | 0.731 → 0.848 (+0.117) | 0.639 → 0.792 (+0.153) | 0.684 → 0.792 (+0.108) | 0.777 → 0.705 (-0.072) | 0.729 → 0.819 (+0.090) |
| pC | 0.554 → 0.527 (-0.027) | 0.641 → 0.614 (-0.027) | 0.683 → 0.548 (-0.135) | 0.507 → 0.462 (-0.045) | 0.733 → 0.643 (-0.090) |
| pD | 0.640 → 0.730 (+0.090) | 0.583 → 0.673 (+0.090) | 0.640 → 0.703 (+0.063) | 0.733 → 0.733 (-0.000) | 0.636 → 0.636 (-0.000) |

---

## 2. Verdict agreement

| | matches OpenAI canonical |
|---|---|
| gemini_run1 | **10/12** |
| gemini_run2 | **11/12** |
| both Gemini runs agree with OpenAI | **10/12** |
| Gemini run1 vs run2 self-agreement | **11/12** |

Proposals that differ:

| run | pid | OpenAI | Gemini | OA overall | Gem overall | confidence |
|---|---|---|---|---|---|---|
| gemini_run1 | p5 | N | Y | 0.6580 | 0.7524 | 0.7791 |
| gemini_run1 | pB | N | Y | 0.7140 | 0.7918 | 0.7869 |
| gemini_run2 | pB | N | Y | 0.7140 | 0.7908 | 0.7869 |

All disagreements are one-directional: Gemini scores these proposals **above** the
0.725 GO threshold where OpenAI fell below it. No proposal flips the other way.

---

## 3. Do the paper's conclusions replicate?

Spearman ρ vs. 3-rater panel consensus, computed with the same
`evaluate_cohens_kappa` functions that produced the published OpenAI numbers
(the OpenAI column reproduces them exactly: ρ=0.1747, verdict accuracy 0.8333).

| target | OpenAI (canonical) | Gemini r1 | Gemini r2 | human LORO band | OA | G r1 | G r2 |
|---|---|---|---|---|---|---|---|
| overall_ranking | 0.1747 | 0.5604 | 0.6187 | [0.918, 0.952] | below | below | below |
| team | 0.3508 | 0.3947 | 0.4385 | [0.863, 0.927] | below | below | below |
| objective | n/a | n/a | n/a | [-0.091, -0.091] | undef. | undef. | undef. |
| strategy | 0.3862 | 0.1931 | 0.1931 | [0.817, 0.913] | below | below | below |
| advantages | 0.1472 | 0.7100 | 0.7590 | [0.798, 0.909] | below | below | below |
| feasibility | 0.0420 | -0.0359 | -0.1116 | [0.893, 0.940] | below | below | below |

| metric | OpenAI | Gemini r1 | Gemini r2 |
|---|---|---|---|
| verdict accuracy | **0.8333** | **0.6667** | **0.75** |
| pairwise rank concordance | 0.5769 | 0.7115 | 0.75 |

**Verdict on replication:**

1. **The human-ceiling finding replicates in full.** Under Gemini, AI-vs-panel ρ sits
   *below* the human LORO band on **every** scorable target, in **both** runs — same as
   OpenAI. `objective` is undefined for both providers (zero-variance consensus).
2. **The specific magnitude ρ=0.175 does NOT replicate.** Gemini reaches ρ=0.560/0.619
   on overall ranking — roughly 3× OpenAI's. The *direction* of the paper's claim holds;
   the *number* is provider-specific and should not be quoted as a property of
   "LLM screening" in general.
3. **Verdict accuracy does NOT replicate.** It drops from 0.833 to 0.667/0.750. Note the
   canonical summary already flags the 0.833 as coincidental (base-rate driven, 2-proposal
   GO set); this result supports that caution.
4. Gemini is better at *ranking* proposals but worse at *thresholding* them — it ranks
   closer to the panel while pushing more proposals over the 0.725 GO line.

---

## 4. Are cross-provider deltas larger than the noise floor?

Within-OpenAI 3-run floor (`stage6_repeatability_3run_analysis.json`): overall MAD **0.0056**, max range **0.0374**; per-dimension max range up to **0.108**.

| | mean \|Δ\| | median \|Δ\| | max \|Δ\| | within-OpenAI floor | within-Gemini 2-run | cells > floor |
|---|---|---|---|---|---|---|
| **overall** | 0.0496 | 0.0551 | 0.0944 | 0.0374 | 0.038 | 17/24 |
| team | 0.0644 | 0.045 | 0.1804 | 0.09 | 0.117 | 5/24 |
| objective | 0.069 | 0.0449 | 0.18 | 0.108 | 0.108 | 8/24 |
| strategy | 0.0618 | 0.0453 | 0.1528 | 0.045 | 0.045 | 13/24 |
| advantages | 0.1107 | 0.0896 | 0.2433 | 0.045 | 0.045 | 18/24 |
| feasibility | 0.0616 | 0.063 | 0.2071 | 0.045 | 0.072 | 13/24 |

**Distinguishable from noise: yes, for the overall score and for three of five dimensions.**

- Mean cross-provider \|Δ\| on overall is **0.0496** — **8.9×** the within-OpenAI MAD of 0.0056, and 17/24 proposal-runs exceed the full within-provider max range.
- Within-Gemini run-to-run spread (mean 0.0154) is far smaller than the cross-provider gap, so the provider effect is not run noise in disguise.
- Per dimension the median delta clears the within-OpenAI floor for **strategy, advantages,
  feasibility**, but **not** for **team** and **objective** — where these two providers differ
  by about as much as one provider differs from itself. Do not claim a provider effect on
  those two dimensions.
- `advantages` (innovation) shows the largest separation (mean \|Δ\| 0.111, max 0.243), driven
  by the AI-cap divergence in §5.

---

## 5. Rule compliance under Gemini

| check | Gemini | OpenAI baseline |
|---|---|---|
| score inside declared band | **119/120 = 0.9917** | 0.9705 |
| schema-valid category letters | **120/120 = 1.0000** | 0.9894 (8× invalid `S-D`) |

Per-dimension band compliance: team 1.0, objectives 1.0, strategy 1.0, innovation 0.9583, feasibility 1.0.

Band violations:

- `gemini_run1` **pB** / innovation: category I-C (band 0.4–0.6) but scored **0.68**

### The "AI-powered" innovation cap **binds under Gemini — it did not under OpenAI**

Rule: RULE 1 / I-C hard cap: AI-as-the-product -> I-C, score 0.42-0.56, NO EXCEPTIONS  
Test case: p1 (heart-failure clinical decision-support large model) — the prompt's own worked example.

| arm | category | score | rule applied |
|---|---|---|---|
| OpenAI (canonical) | I-B | 0.75 | **no** |
| gemini_run1 | I-C | 0.48 | **yes** |
| gemini_run2 | I-C | 0.48 | **yes** |

Across all Gemini calls, 11 cells were assigned I-C; 10/11 fall inside the header range 0.40–0.60 and 10/11 inside the stricter RULE 1 range 0.42–0.56 (OpenAI cap-range compliance was 0.4524).

**This matters for paper fix B4.** The AI-cap ablation was abandoned because the
cap-bound set was empty (0/12) under OpenAI, leaving the A-vs-B contrast with no
members. Under Gemini the cap *does* fire on p1 in both runs. The B4 finding is
therefore **provider-specific, not a property of the prompt**: the same rule text is
inert for one scorer and active for another. That is a stronger statement about
prompt-rule fragility than the original empty-set result, and it should be reported.

### Top-band firing

| dimension | Gemini A-band fires (of 24) | OpenAI A-band fires (of 151) |
|---|---|---|
| team | 8 | 4 |
| objectives | 9 | 0 |
| strategy | 4 | 0 |
| innovation | 4 | 0 |
| feasibility | 4 | 0 |

OpenAI fired a top band **only** on `team` (4/151) and never on the other four
dimensions. Gemini fires top bands across all five. Gemini uses more of the declared
scale, which is the most likely mechanism behind its higher rank correlation — and
means the paper's "scores cluster in a narrow band" observation is a property of
gpt-4o-mini, not of the rubric.

---

## Files

- `canonical/cross_provider_gemini.json` — per-proposal results, metrics, provenance
- `canonical/cross_provider_gemini_raw.json` — raw two-run Call B output
- `run_cross_provider_gemini.py` — runner (24 Call B invocations)
- `analyze_cross_provider_gemini.py` — analysis (0 LLM calls)

Not modified: existing `canonical/` files, `results/`, `resultsNew/`, `src/data/`.
