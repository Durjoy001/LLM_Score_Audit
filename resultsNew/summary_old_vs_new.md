# LLM Score Audit — OLD (single-rater) vs NEW (3-rater consensus), CANONICAL Jun-16 scores

_Both columns use the canonical Jun-16 AI score set (`canonical_ai_scores.json`), which reproduces the published paper numbers exactly (ρ=0.5894, verdict-acc=0.8333). The stale Jun-18 `src/data` cache is not used. pA is correctly scored as pA__openai (0.681)._

**Non-unanimous consensus verdicts:** ['p5', 'pD']

## 6.1 — Per-dimension AI-vs-human agreement

| Dimension | κ OLD | κ NEW | ρ OLD | ρ NEW |
|---|---|---|---|---|
| team | 0.3239 | 0.3846 | 0.3982 | 0.3508 |
| objective | -0.1379 | 0.0 | -0.2748 | n/a (constant) |
| strategy | -0.125 | 0.3333 | 0.3236 | 0.3862 |
| advantages | -0.08 | 0.2258 | 0.0612 | 0.1472 |
| feasibility | 0.25 | 0.05 | 0.1561 | 0.042 |

### Headline scalars

| Metric | OLD (paper) | NEW consensus |
|---|---|---|
| Overall Spearman ρ | 0.5894 | 0.1747 |
| Overall ICC(2,1) | 0.1188 | 0.1357 |
| Pairwise rank concordance | 0.7586 | 0.5769 |
| Verdict accuracy (OR-logic) | 0.8333 | 0.8333 |

_NEW overall ρ=0.175 uses integer-rounded consensus ranks (evaluate_cohens_kappa convention); the LORO table below uses un-rounded mean ranks (ρ=0.050). Both are weakly positive and far below the OLD single-rater 0.589 and the human LORO band. NEW verdict accuracy ties OLD at 0.833 only because the OR-logic AI GO-set {p3,p7} lands inside the consensus GO-set; pA and pD are the 2 misses._

## 6.3 — Calibration (canonical invest/qa echoes)

| Metric | OLD | NEW |
|---|---|---|
| INVEST_WEIGHT peak ρ | 0.6001 @ [1.0] | 0.0956 @ [0.95] |
| IW=0.90 ties peak? | False | False |
| max verdict-acc τ_r (smallest) | 0.6624 | 0.6881 |
| τ_c | 0.8112 | 0.8112 |

### INVEST_WEIGHT sweep (ρ at each step)

| IW | ρ OLD | ρ NEW |
|---|---|---|
| 0.00 | -0.3251 | -0.1027 |
| 0.05 | -0.3251 | -0.1027 |
| 0.10 | -0.3251 | -0.1487 |
| 0.15 | -0.1608 | -0.0354 |
| 0.20 | -0.1608 | -0.0354 |
| 0.25 | -0.1789 | -0.1028 |
| 0.30 | -0.0714 | -0.046 |
| 0.35 | 0.0322 | -0.0602 |
| 0.40 | 0.0965 | -0.1451 |
| 0.45 | 0.2129 | -0.0638 |
| 0.50 | 0.2286 | -0.0566 |
| 0.55 | 0.2643 | -0.0389 |
| 0.60 | 0.4037 | 0.0177 |
| 0.65 | 0.4287 | 0.0531 |
| 0.70 | 0.4894 | 0.0389 |
| 0.75 | 0.5215 | 0.0708 |
| 0.80 | 0.5797 | 0.0053 |
| 0.85 | 0.5958 | 0.0213 |
| 0.90 | 0.5894 | 0.0496 |
| 0.95 | 0.5716 | 0.0956 |
| 1.00 | 0.6001 | 0.0673 |

## Kickstarter 0%-recall transfer (canonical; cache NOT drifted)

- Max Kickstarter AI score = 0.64 (ref 0.64).
- New τ_r range vs 0.640, recall under all thresholds, finding holds: **True** (all zero recall: True).

## NEW — human-human reliability (unchanged; AI-independent)

| Target | ICC(2,1) | ICC(2,k) | mean pairwise wκ | Gwet AC1 |
|---|---|---|---|---|
| team | 0.875 | 0.9545 | 0.8671 | 0.8436 |
| objective | -0.0476 | -0.1579 | -0.0303 | 0.8759 |
| strategy | 0.7944 | 0.9206 | 0.7808 | 0.7778 |
| advantages | 0.7885 | 0.9179 | 0.7745 | 0.7838 |
| feasibility | 0.8553 | 0.9466 | 0.8421 | 0.8476 |
| overall_ranking | 0.8586 | 0.948 | — | — |

## NEW — AI-vs-panel vs human LORO band (canonical)

| Target | AI-vs-consensus ρ | human LORO band | position |
|---|---|---|---|
| team | 0.3508 | [0.8625, 0.927] | BELOW human band |
| objective | None | [-0.0909, -0.0909] | undefined |
| strategy | 0.3862 | [0.8173, 0.9129] | BELOW human band |
| advantages | 0.1472 | [0.7977, 0.9087] | BELOW human band |
| feasibility | 0.042 | [0.8926, 0.94] | BELOW human band |
| overall_ranking | 0.0496 | [0.9184, 0.9521] | BELOW human band |
