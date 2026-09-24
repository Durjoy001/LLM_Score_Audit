#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render canonical/ablation_b4.md from ablation_b4.json. Formatting only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "canonical" / "ablation_b4.json").read_text())
C1, C2, C3, C4 = (D["component_1_cap_ablation"], D["component_2_verdict_rules"],
                  D["component_3_blend_sweep"], D["component_4_score_vs_threshold_validity"])
PV = D["provenance"]
RULES = ["score_only", "confidence_only", "score_AND_confidence", "score_OR_confidence"]
L = []


def g(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


L += ["# B4 — Ablating the heuristic rules", "",
      f"**Date:** {PV['analysis_date_utc'][:10]} | **LLM calls:** {PV['cap_ablation_calls']} (cap ablation only; "
      "components 2–4 are pure re-derivation)  ",
      f"**Prompt variants:** {PV['prompt_variants']}  ",
      "**Ground truth:** Dataset1 = 3-rater panel consensus (n=12); Kickstarter = `funded_pct ≥ 100`, a real outcome (n=141)",
      "", "---", "",
      "## Component 1 — Remove the \"AI-powered\" innovation cap", "",
      "Three conditions, both providers, 2 runs per cell:", "",
      "| condition | prompt | meaning |", "|---|---|---|",
      "| **A** | sha `182a50db…` (9197 ch) | production prompt, cap present |",
      "| **B** | sha `27977547…` (8633 ch) | pre-built variant — removes RULE 1 + HARD CAP line only |",
      "| **C** | sha `926d894b…` (8485 ch) | **genuine ablation** — also removes the worked example restating the cap, the `(INCLUDES most AI proposals)` steer, and the dangling `DO NOT apply the I-C cap` clause |",
      "",
      "> **Why C exists.** B's own provenance records that cap language survives in it — most",
      "> critically the EXAMPLES bullet `→ AI heart failure decision support system (AI IS the",
      "> product) → I-C, score 0.42-0.56`, which restates the cap verbatim *and* describes p1's",
      "> exact archetype. B ablates the rule's statement while leaving an instruction that applies",
      "> the same rule to the test case. A null result under B would be uninterpretable.",
      "",
      "### Cap-bound cohort under condition A", "",
      "| provider | I-C any run | cap actually fires (I-C **and** score in 0.42–0.56) |", "|---|---|---|"]
for p in ("openai", "gemini"):
    co = C1["cohort"][p]
    L.append(f"| {p} | {', '.join(co['i_c_any_run']) or '—'} | **{', '.join(co['cap_fires_any_run']) or '—'}** "
             f"({len(co['cap_fires_any_run'])}/12) |")
L += ["", "### Effect of removing the cap (mean Δ vs condition A)", "",
      "| provider | group | n | Δ innovation (B−A) | Δ innovation (C−A) | Δ overall (C−A) |", "|---|---|---|---|---|---|"]
for p in ("openai", "gemini"):
    e = C1["effects"][p]
    for grp, lab in (("cohort", "cap-bound"), ("non_cohort", "not cap-bound")):
        i, o = e["innovation_delta"][grp], e["overall_delta"][grp]
        L.append(f"| {p} | {lab} | {i['B_minus_A'].get('n')} | {g(i['B_minus_A'].get('mean'))} | "
                 f"**{g(i['C_minus_A'].get('mean'))}** | {g(o['C_minus_A'].get('mean'))} |")
nf = C1["noise_floor"]
L += ["", f"Noise floor for reference: within-provider innovation max range **{nf['innovation_max_range']}**, "
      f"overall MAD **{nf['overall_mad']}**, overall max range **{nf['overall_max_range']}**.", "",
      "### Per-proposal innovation score, cap-bound proposals only", "",
      "| provider | pid | A | B | C | Δ C−A | category A → C |", "|---|---|---|---|---|---|---|"]
for p in ("openai", "gemini"):
    e = C1["effects"][p]
    for pid, v in e["per_proposal"].items():
        if not v["in_cap_cohort"]:
            continue
        L.append(f"| {p} | {pid} | {v['innovation_invest']['A']:.3f} | {v['innovation_invest']['B']:.3f} | "
                 f"{v['innovation_invest']['C']:.3f} | **{v['delta_innovation']['C_minus_A']:+.3f}** | "
                 f"{'/'.join(str(x) for x in v['categories']['A'])} → {'/'.join(str(x) for x in v['categories']['C'])} |")
L += ["", "**Findings.**", "",
      "1. **The cap binds for exactly one OpenAI proposal (pC) and five Gemini proposals.** The",
      "   B4 pre-flight that reported an empty cohort was measuring OpenAI's Jun-16 run, where",
      "   only pC reaches I-C. The rule is not inert — it is *rarely triggered* under gpt-4o-mini.",
      "2. **Removing the cap raises the capped score, and the effect clears the noise floor.**",
      "   OpenAI pC: 0.500 → 0.600 (+0.100 vs 0.045 floor), overall +0.0561 vs 0.0374 floor.",
      "3. **B and C diverge for Gemini, exactly as predicted.** On the cap-bound cohort B moves",
      "   innovation by +0.007 (nothing) while C moves it by +0.049. p1 is the clearest case:",
      "   0.480 → 0.475 under B, → 0.535 under C. **The pre-built variant would have produced a",
      "   false null.** For OpenAI B and C agree, because pC is not the archetype named in the",
      "   surviving example.",
      "4. **Caveat — the ablation is not perfectly surgical.** Gemini p3 (not cap-bound) swings",
      "   −0.255 on innovation under C, dropping I-A → I-C/I-B. Editing the INNOVATION section",
      "   perturbs anchoring for non-AI proposals too, so the causal claim should be limited to",
      "   the cap-bound cohort, where the direction is consistent.", "",
      "---", "", "## Component 2 — Verdict rule ablation", "",
      f"Thresholds held at production values: τ_r = {C2['thresholds']['tau_r']}, τ_c = {C2['thresholds']['tau_c']}.", "",
      "### Dataset1 (n=12, truth = 3-rater majority verdict, base rate 0.333)", "",
      "| arm | rule | accuracy | precision | recall | F1 | n predicted GO |", "|---|---|---|---|---|---|---|"]
for arm, m in C2["dataset1"].items():
    for r in RULES:
        x = m[r]
        L.append(f"| {arm} | `{r}` | {x['accuracy']:.3f} | {g(x['precision'],3)} | {g(x['recall'],3)} | "
                 f"{g(x['f1'],3)} | {x['n_predicted_go']} |")
k = C2["kickstarter"]["openai_canonical"]
L += ["", f"### Kickstarter (n={k['n']}, truth = funded ≥ 100%, base rate {k['base_rate_positive']})", "",
      f"Max AI overall score = **{k['max_ai_overall']}**, below τ_r = {C2['thresholds']['tau_r']}.", "",
      "| rule | accuracy | recall | n predicted GO | TP/FP/FN/TN |", "|---|---|---|---|---|"]
for r in RULES:
    x = k[r]
    L.append(f"| `{r}` | {x['accuracy']:.4f} | {g(x['recall'],3)} | {x['n_predicted_go']} | "
             f"{x['tp']}/{x['fp']}/{x['fn']}/{x['tn']} |")
L += ["", "**Findings.**", "",
      "1. On Dataset1 the production OR rule is the best of the four for OpenAI (0.833) but",
      "   **not** for Gemini, where score-only and confidence-only both beat it in run 1. The",
      "   OR rule's advantage is scorer-specific, not structural.",
      "2. The AND rule collapses to zero predicted GO for OpenAI — it is strictly worse than",
      "   either conjunct alone. The AND→OR change in commit `3e84844` was load-bearing.",
      "3. **On Kickstarter all four rules are numerically identical** — accuracy 0.8298, recall",
      "   0.000, zero predicted GO. Every rule is inert because no proposal clears either",
      "   threshold. Four structurally different decision rules are indistinguishable from",
      "   thresholded accuracy. This is the setup for component 4.", "",
      "---", "", "## Component 3 — Vary the 0.90/0.10 blend", "",
      "### Dataset1", "",
      "| arm | ρ @ IW=0.90 | peak ρ | peak IW | production ties peak | distinct IW across LOO folds | production picked |",
      "|---|---|---|---|---|---|---|"]
for arm, m in C3["dataset1"].items():
    L.append(f"| {arm} | {g(m['rho_at_production_iw_0.90'])} | {g(m['peak_rho'])} | {m['peak_iw']} | "
             f"{m['production_ties_peak']} | {m['loo_iw_distinct']} | {m['loo_production_selected_frac']*100:.0f}% |")
ks = C3["kickstarter"]
L += ["", f"### Kickstarter (n={ks['n']}, {ks['n_excluded']} excluded; weights: {ks['dimension_weights']})", "",
      f"ρ @ IW=0.90 = **{ks['rho_at_production_iw_0.90']}**, AUC @ IW=0.90 = **{ks['auc_at_production_iw_0.90']}**; "
      f"peak ρ = {ks['peak_rho']} @ IW={ks['peak_iw']}.", "",
      "Nested 5-fold CV (inner loop selects IW on train, outer evaluates on held-out fold):", "",
      "| fold | inner-selected IW | train ρ | test ρ @ selected | test ρ @ 0.90 | test AUC @ selected | test AUC @ 0.90 |",
      "|---|---|---|---|---|---|---|"]
for f in ks["nested_cv_5fold"]:
    L.append(f"| {f['fold']} | {f['inner_selected_iw']} | {f['inner_train_rho']:.4f} | "
             f"{g(f['outer_test_rho_at_selected'])} | {g(f['outer_test_rho_at_production_0.90'])} | "
             f"{f['outer_test_auc_at_selected']:.4f} | {f['outer_test_auc_at_production_0.90']:.4f} |")
L += ["", f"Mean held-out ρ at inner-selected IW = **{ks['nested_mean_test_rho_at_selected']}**; "
      f"at production IW=0.90 = **{ks['nested_mean_test_rho_at_production']}**.", "",
      "**Findings.**", "",
      "1. **IW=0.90 is not an optimum and is not stable.** On Dataset1 it never ties the peak for",
      "   OpenAI, and LOO resampling selects five different IW values across twelve folds —",
      "   production is chosen in 0% of them. At n=12 the blend weight is unidentifiable.",
      "2. **On Kickstarter the objective is flat.** ρ moves from 0.348 to 0.374 across IW ∈",
      "   [0.6, 1.0] and AUC from 0.809 to 0.820. Any weight in that range is statistically",
      "   indistinguishable.",
      "3. **Nested CV shows tuning buys nothing.** The inner loop selects IW=1.00 in all five",
      "   folds, yet held-out ρ at the selected weight (0.3522) is *marginally worse* than at the",
      "   untuned production weight (0.3533). Selecting the blend on training data does not",
      "   generalise — it is fitting a flat surface.",
      "4. **The QA component carries almost no signal.** At IW=0.00 (pure QA) ρ = −0.038 and",
      "   AUC = 0.521 — chance. All discriminative content is in the investment score. The",
      "   0.90/0.10 blend is therefore a presentational choice, not a tuned parameter.", "",
      "---", "", "## Component 4 — Score validity vs threshold validity", "",
      "### Dataset1: the two orderings invert", "",
      "| arm | score validity ρ | score validity AUC | thresholded accuracy | recall |", "|---|---|---|---|---|"]
for a, m in C4["dataset1"].items():
    L.append(f"| {a} | {g(m['score_validity_rho'])} | {g(m['score_validity_auc'])} | "
             f"{m['threshold_validity_accuracy']:.4f} | {m['threshold_validity_recall']:.4f} |")
L += ["", f"- Ranking by **score validity**: {' > '.join(C4['ranking_by_score_validity'])}",
      f"- Ranking by **thresholded accuracy**: {' > '.join(C4['ranking_by_thresholded_accuracy'])}",
      f"- Orderings agree: **{C4['orderings_agree']}** — they are exactly reversed.", ""]
k4 = C4["kickstarter"]
L += [f"### Kickstarter (n={k4['n']}, real outcome): the sharpest case", "",
      "| quantity | value |", "|---|---|",
      f"| score validity — ρ vs `funded_pct` | **{k4['score_validity_rho_vs_funded_pct']}** |",
      f"| score validity — AUC (funded vs not) | **{k4['score_validity_auc']}** |",
      f"| max AI overall score | {k4['max_ai_overall']} (τ_r = {k4['tau_r']}) |",
      f"| thresholded accuracy, **all four rules** | {k4['threshold_validity']['score_only']['accuracy']:.4f} |",
      f"| recall, all four rules | 0.000 |",
      f"| negative base rate | {1 - k4['base_rate_positive']:.4f} |", "",
      "**The finding.** The AI score ranks Kickstarter outcomes with **AUC 0.764** — genuinely",
      "informative. Thresholded at the production τ_r it predicts **zero** GO, yielding accuracy",
      "0.8298 that is *exactly* the negative base rate and recall 0.000. A reader looking only at",
      "83% accuracy would conclude the system works; a reader looking at recall would conclude it",
      "is useless; both would miss that the underlying score has real discriminative power the",
      "threshold discards.", "",
      "On Dataset1 the same distinction appears as an inversion: OpenAI has the **worst** score",
      "validity (ρ=0.175, AUC=0.625) but the **best** thresholded accuracy (0.833), while Gemini",
      "run 2 has the best score validity (ρ=0.619, AUC=0.844) and lower accuracy (0.750).",
      "",
      "**Thresholded accuracy is not a proxy for score validity — on these data it is",
      "anti-correlated with it.** Any evaluation that reports only verdict accuracy is blind to",
      "whether the scorer ranks proposals well, and that is precisely the distinction the paper",
      "argues for.", "",
      "---", "", "## Consolidated verdict on the heuristics", "",
      "| heuristic | status after B4 |", "|---|---|",
      "| \"AI-powered\" innovation cap | **Fires rarely (1/12 OpenAI, 5/12 Gemini) and its effect is scorer-dependent.** Real but small; demote to a documented heuristic. |",
      "| OR-logic verdict rule | **Best of four for OpenAI only.** Not structurally justified; scorer-specific. |",
      "| τ_r = 0.725 / τ_c = 0.811 | **Inert on Kickstarter** — no proposal reaches either. Not transferable across datasets. |",
      "| INVEST_WEIGHT = 0.90 | **Not an optimum, not stable, and tuning it does not generalise.** Any IW ∈ [0.6, 1.0] is equivalent. Demote to a presentational default. |",
      "", "All four should be reported as **current heuristic settings**, not tuned parameters.",
      "The evidence for demotion is now quantitative rather than an admission of time pressure.", "",
      "---", "", "## Files", "",
      "- `canonical/ablation_b4.json` — all four components, full detail",
      "- `canonical/b4_prompt_variants.json` — A/B/C prompt texts, shas, edit log, residual-marker check",
      "- `canonical/b4_cap_ablation_raw.json` — 96 raw Call B invocations",
      "- `canonical/b4_kickstarter_invest_qa.json` — recovered invest/qa echoes + validation",
      "- Scripts: `build_b4_prompt_c.py`, `run_b4_cap_ablation.py`, `parse_kickstarter_invest_qa.py`, `analyze_b4.py`", "",
      "Not modified: existing `canonical/` files, `results/`, `resultsNew/`, `src/data/`.", ""]

out = ROOT / "canonical" / "ablation_b4.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"[OK] wrote {out}")
