#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render canonical/cross_provider_gemini.md (paper tables) from the analysis JSON.

Pure formatting. No LLM calls, no recomputation.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "canonical" / "cross_provider_gemini.json").read_text())
PP, REP, NC, RC, VA = (D["per_proposal"], D["replication"], D["noise_context"],
                       D["rule_compliance"], D["verdict_agreement"])
PV = D["provenance"]
PIDS = list(PP)
DIMS = ["team", "objective", "strategy", "advantages", "feasibility"]
RUNS = ["gemini_run1", "gemini_run2"]
L = []


def fmt(v, n=4):
    return "n/a" if v is None else f"{v:.{n}f}"


L += [
    "# Cross-provider replication check (paper fix B3) — OpenAI vs Gemini, Dataset1",
    "",
    f"**Model:** `{PV['model']}` ({PV['model_tier']})  ",
    f"**Temperature:** {PV['temperature']} | **Thinking:** {PV['thinking_level']} | "
    f"**Seed:** requested {PV['seed_requested']}, reached provider: **{PV['seed_reached_provider']}**  ",
    f"**Date:** {PV['date_local']} | **Runs:** {PV['runs']} × 12 proposals = "
    f"{PV['runs'] * 12} Call B invocations  ",
    f"**Evidence:** {PV['evidence_source']}  ",
    f"**Prompt SHA256:** `{PV['system_prompt_sha256'][:32]}…` (unmodified production prompt)",
    "",
    "> **Seed caveat.** Gemini exposes no seed parameter. The OpenAI arm is nominally",
    "> seeded (best-effort); the Gemini arm is **unseeded**. Run-to-run spread within",
    "> Gemini is therefore an upper bound on what a seeded Gemini would show.",
    "",
    "---",
    "",
    "## 1. Per-proposal overall score",
    "",
    "| pid | OpenAI | Gemini r1 | Δ r1 | Gemini r2 | Δ r2 | OA verdict | G r1 | G r2 |",
    "|---|---|---|---|---|---|---|---|---|",
]
for p in PIDS:
    r = PP[p]
    L.append(f"| {p} | {r['openai']['overall']:.4f} | {r['gemini_run1']['overall']:.4f} | "
             f"{r['gemini_run1']['delta_overall']:+.4f} | {r['gemini_run2']['overall']:.4f} | "
             f"{r['gemini_run2']['delta_overall']:+.4f} | {r['openai']['verdict']} | "
             f"{r['gemini_run1']['verdict']} | {r['gemini_run2']['verdict']} |")
mo = statistics.mean(PP[p]["openai"]["overall"] for p in PIDS)
L += ["", f"Mean overall: OpenAI **{mo:.4f}**, " + ", ".join(
    f"Gemini {r[-1]} **{statistics.mean(PP[p][r]['overall'] for p in PIDS):.4f}**" for r in RUNS) + ".", ""]

for run in RUNS:
    L += [f"## 1b. Per-dimension blended scores — {run}", "",
          "| pid | " + " | ".join(f"{d} (OA → Gem, Δ)" for d in DIMS) + " |",
          "|---" * (len(DIMS) + 1) + "|"]
    for p in PIDS:
        cells = []
        for d in DIMS:
            oa = PP[p]["openai"]["dimensions"][d]
            g = PP[p][run]["dimensions"][d]
            dl = PP[p][run]["delta_dimensions"][d]
            cells.append(f"{oa:.3f} → {g:.3f} ({dl:+.3f})")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L.append("")

L += ["---", "", "## 2. Verdict agreement", "",
      "| | matches OpenAI canonical |", "|---|---|"]
for r in RUNS:
    L.append(f"| {r} | **{VA[r]['n_matching']}/12** |")
L += [f"| both Gemini runs agree with OpenAI | **{VA['both_runs_agree_with_openai']}/12** |",
      f"| Gemini run1 vs run2 self-agreement | **{VA['gemini_run1_vs_run2_agreement']}/12** |", ""]
L += ["Proposals that differ:", "",
      "| run | pid | OpenAI | Gemini | OA overall | Gem overall | confidence |", "|---|---|---|---|---|---|---|"]
for r in RUNS:
    for x in VA[r]["differing"]:
        L.append(f"| {r} | {x['pid']} | {x['openai']} | {x['gemini']} | {x['openai_overall']:.4f} | "
                 f"{x['gemini_overall']:.4f} | {x['confidence']:.4f} |")
L += ["", "All disagreements are one-directional: Gemini scores these proposals **above** the",
      "0.725 GO threshold where OpenAI fell below it. No proposal flips the other way.", ""]

L += ["---", "", "## 3. Do the paper's conclusions replicate?", "",
      "Spearman ρ vs. 3-rater panel consensus, computed with the same",
      "`evaluate_cohens_kappa` functions that produced the published OpenAI numbers",
      "(the OpenAI column reproduces them exactly: ρ=0.1747, verdict accuracy 0.8333).", "",
      "| target | OpenAI (canonical) | Gemini r1 | Gemini r2 | human LORO band | OA | G r1 | G r2 |",
      "|---|---|---|---|---|---|---|---|"]
short = {"BELOW human band": "below", "ABOVE human band": "above",
         "INSIDE human band": "inside", "undefined": "undef."}
for t in ["overall_ranking"] + DIMS:
    o = REP["openai_canonical"]["metrics"][t]["spearman_r"]
    g = [REP["gemini"][r]["metrics"][t]["spearman_r"] for r in RUNS]
    lo = REP["loro"]["openai_canonical"]["targets"][t]
    b = lo["human_band"]
    bs = "n/a" if b[0] is None else f"[{b[0]:.3f}, {b[1]:.3f}]"
    pos = [short[lo["ai_position"]]] + [short[REP["loro"]["gemini"][r]["targets"][t]["ai_position"]] for r in RUNS]
    L.append(f"| {t} | {fmt(o)} | {fmt(g[0])} | {fmt(g[1])} | {bs} | {pos[0]} | {pos[1]} | {pos[2]} |")
L += ["", "| metric | OpenAI | Gemini r1 | Gemini r2 |", "|---|---|---|---|",
      f"| verdict accuracy | **{REP['openai_canonical']['verdict_accuracy']}** | "
      f"**{REP['gemini']['gemini_run1']['verdict_accuracy']}** | "
      f"**{REP['gemini']['gemini_run2']['verdict_accuracy']}** |",
      f"| pairwise rank concordance | {REP['openai_canonical']['rank_order']['pairwise_rank_concordance']} | "
      f"{REP['gemini']['gemini_run1']['rank_order']['pairwise_rank_concordance']} | "
      f"{REP['gemini']['gemini_run2']['rank_order']['pairwise_rank_concordance']} |", "",
      "**Verdict on replication:**", "",
      "1. **The human-ceiling finding replicates in full.** Under Gemini, AI-vs-panel ρ sits",
      "   *below* the human LORO band on **every** scorable target, in **both** runs — same as",
      "   OpenAI. `objective` is undefined for both providers (zero-variance consensus).",
      "2. **The specific magnitude ρ=0.175 does NOT replicate.** Gemini reaches ρ=0.560/0.619",
      "   on overall ranking — roughly 3× OpenAI's. The *direction* of the paper's claim holds;",
      "   the *number* is provider-specific and should not be quoted as a property of",
      "   \"LLM screening\" in general.",
      "3. **Verdict accuracy does NOT replicate.** It drops from 0.833 to 0.667/0.750. Note the",
      "   canonical summary already flags the 0.833 as coincidental (base-rate driven, 2-proposal",
      "   GO set); this result supports that caution.",
      "4. Gemini is better at *ranking* proposals but worse at *thresholding* them — it ranks",
      "   closer to the panel while pushing more proposals over the 0.725 GO line.", ""]

L += ["---", "", "## 4. Are cross-provider deltas larger than the noise floor?", "",
      f"Within-OpenAI 3-run floor (`stage6_repeatability_3run_analysis.json`): overall "
      f"MAD **{NC['within_openai_3run_floor']['overall_mad']}**, max range "
      f"**{NC['within_openai_3run_floor']['overall_max_range']}**; per-dimension max range up to "
      f"**{max(NC['within_openai_3run_floor']['dimension_max_range'].values())}**.", "",
      "| | mean \\|Δ\\| | median \\|Δ\\| | max \\|Δ\\| | within-OpenAI floor | within-Gemini 2-run | cells > floor |",
      "|---|---|---|---|---|---|---|"]
o = NC["overall"]
L.append(f"| **overall** | {o['cross_provider_mean_abs_delta']} | {o['cross_provider_median_abs_delta']} | "
         f"{o['cross_provider_max_abs_delta']} | {NC['within_openai_3run_floor']['overall_max_range']} | "
         f"{o['within_gemini_2run_max_abs_diff']} | {o['n_exceeding_openai_max_range']}/{o['n_cells']} |")
for d in DIMS:
    m = NC["per_dimension"][d]
    L.append(f"| {d} | {m['cross_provider_mean_abs_delta']} | {m['cross_provider_median_abs_delta']} | "
             f"{m['cross_provider_max_abs_delta']} | {m['within_openai_3run_max_range']} | "
             f"{m['within_gemini_2run_max_range']} | {m['n_cells_exceeding_openai_floor']}/{m['n_cells']} |")
ratio = o["cross_provider_mean_abs_delta"] / NC["within_openai_3run_floor"]["overall_mad"]
L += ["",
      f"**Distinguishable from noise: yes, for the overall score and for three of five dimensions.**", "",
      f"- Mean cross-provider \\|Δ\\| on overall is **{o['cross_provider_mean_abs_delta']}** — "
      f"**{ratio:.1f}×** the within-OpenAI MAD of {NC['within_openai_3run_floor']['overall_mad']}, and "
      f"{o['n_exceeding_openai_max_range']}/{o['n_cells']} proposal-runs exceed the full within-provider max range.",
      f"- Within-Gemini run-to-run spread (mean {o['within_gemini_2run_mean_abs_diff']}) is far smaller than "
      "the cross-provider gap, so the provider effect is not run noise in disguise.",
      "- Per dimension the median delta clears the within-OpenAI floor for **strategy, advantages,",
      "  feasibility**, but **not** for **team** and **objective** — where these two providers differ",
      "  by about as much as one provider differs from itself. Do not claim a provider effect on",
      "  those two dimensions.",
      "- `advantages` (innovation) shows the largest separation (mean \\|Δ\\| 0.111, max 0.243), driven",
      "  by the AI-cap divergence in §5.", ""]

L += ["---", "", "## 5. Rule compliance under Gemini", "",
      "| check | Gemini | OpenAI baseline |", "|---|---|---|"]
b = RC["check_score_within_declared_band"]
s = RC["check_schema_validity"]
L += [f"| score inside declared band | **{b['compliant']}/{b['checked']} = {b['rate']}** | {b['openai_baseline_rate']} |",
      f"| schema-valid category letters | **{s['total_cells'] - s['invalid_cells']}/{s['total_cells']} = "
      f"{1 - s['rate']:.4f}** | 0.9894 (8× invalid `S-D`) |", "",
      "Per-dimension band compliance: " + ", ".join(
          f"{k} {v['rate']}" for k, v in b["per_dimension"].items()) + ".", ""]
if b["violations"]:
    L += ["Band violations:", ""]
    for v in b["violations"]:
        L.append(f"- `{v['run']}` **{v['pid']}** / {v['dimension']}: category {v['category']} "
                 f"(band {v['band'][0]}–{v['band'][1]}) but scored **{v['score']}**")
    L.append("")
cap = RC["check_ai_cap"]
L += ["### The \"AI-powered\" innovation cap **binds under Gemini — it did not under OpenAI**", "",
      f"Rule: {cap['rule']}  ",
      f"Test case: {cap['precondition_proposal']} — the prompt's own worked example.", "",
      "| arm | category | score | rule applied |", "|---|---|---|---|",
      f"| OpenAI (canonical) | {cap['openai_baseline']['actually_assigned']['category']} | "
      f"{cap['openai_baseline']['actually_assigned']['score']} | **no** |"]
for r, v in cap["gemini"].items():
    L.append(f"| {r} | {v['category']} | {v['score']} | **yes** |")
ic = cap["all_I_C_cells"]
L += ["", f"Across all Gemini calls, {ic['n']} cells were assigned I-C; "
      f"{ic['inside_header_range_0.40_0.60']}/{ic['n']} fall inside the header range 0.40–0.60 and "
      f"{ic['inside_cap_range_0.42_0.56']}/{ic['n']} inside the stricter RULE 1 range 0.42–0.56 "
      f"(OpenAI cap-range compliance was 0.4524).", "",
      "**This matters for paper fix B4.** The AI-cap ablation was abandoned because the",
      "cap-bound set was empty (0/12) under OpenAI, leaving the A-vs-B contrast with no",
      "members. Under Gemini the cap *does* fire on p1 in both runs. The B4 finding is",
      "therefore **provider-specific, not a property of the prompt**: the same rule text is",
      "inert for one scorer and active for another. That is a stronger statement about",
      "prompt-rule fragility than the original empty-set result, and it should be reported.", ""]
bf = RC["check_band_firing"]
L += ["### Top-band firing", "",
      "| dimension | Gemini A-band fires (of 24) | OpenAI A-band fires (of 151) |", "|---|---|---|"]
for k, v in bf["per_dimension"].items():
    L.append(f"| {k} | {v['A_fired']} | {bf['openai_baseline'][k]['A_fired']} |")
L += ["", "OpenAI fired a top band **only** on `team` (4/151) and never on the other four",
      "dimensions. Gemini fires top bands across all five. Gemini uses more of the declared",
      "scale, which is the most likely mechanism behind its higher rank correlation — and",
      "means the paper's \"scores cluster in a narrow band\" observation is a property of",
      "gpt-4o-mini, not of the rubric.", ""]

L += ["---", "", "## Files", "",
      "- `canonical/cross_provider_gemini.json` — per-proposal results, metrics, provenance",
      "- `canonical/cross_provider_gemini_raw.json` — raw two-run Call B output",
      "- `run_cross_provider_gemini.py` — runner (24 Call B invocations)",
      "- `analyze_cross_provider_gemini.py` — analysis (0 LLM calls)", "",
      "Not modified: existing `canonical/` files, `results/`, `resultsNew/`, `src/data/`.", ""]

out = ROOT / "canonical" / "cross_provider_gemini.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"[OK] wrote {out}")
