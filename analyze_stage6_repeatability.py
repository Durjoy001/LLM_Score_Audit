#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-run repeatability analysis for Stage 6 Call B (deterministic investment scoring).

Combines the three independent samples of the SAME deterministic call
(temperature=0, seed=42, identical dimensions_v2.json evidence):

  run1  canonical/canonical_ai_scores_dataset1.json      (Jun-16, paper baseline)
  run2  src/data/expert_reports/<pid>/ai_expert_opinion.json (Jun-18 re-run)
  run3  canonical/stage6_repeatability_run3.json         (produced by
        run_stage6_callb_run3.py)

Method notes
------------
* Every run's overall_score and verdict are RE-DERIVED here from that run's raw
  Call B invest scores using one identical formula and one identical verdict
  rule, so all run-to-run deltas are attributable to Call B and not to the
  AND->OR verdict-rule change that landed in git between runs 1 and 2.
      blended_dim = 0.90 * invest_dim + 0.10 * qa_dim   (qa frozen, Stage 5)
      overall     = dimension-weighted mean of blended_dim
      verdict     = GO iff overall >= 0.725 OR confidence >= 0.811
                    NO-GO iff overall < 0.40 or blended innovation < 0.30
                              or blended feasibility < 0.30
                    else HOLD
  The re-derivation is asserted against the published canonical run-1 numbers
  (tolerance 0.0006, the documented cross-source tolerance) before anything else
  is computed; the script aborts if it does not reproduce them.
* 95% CIs use `_bootstrap_ci` imported unchanged from
  src/tools/evaluate_cohens_kappa.py -- the same 1000-resample percentile
  bootstrap with numpy Generator seeded 42 used for every other CI in this
  project. With n=3 samples a bootstrap CI is extremely coarse (it can only ever
  span the observed values); it is reported because the project uses this method
  throughout, but std and the mean-absolute-deviation statistic are the more
  honest summaries at this n.
* std is the sample standard deviation (ddof=1).

Outputs (new files only; no existing artefact is modified):
  canonical/stage6_repeatability_3run_analysis.json
  canonical/stage6_repeatability_3run_analysis.md
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src" / "tools"))

from evaluate_cohens_kappa import _bootstrap_ci  # noqa: E402  (1000 resamples, rng seed 42)

DIMS = ["team", "objectives", "strategy", "innovation", "feasibility"]
# canonical file uses the human-rubric dimension names for the blended scores
CANON_DIM = {"team": "team", "objectives": "objective", "strategy": "strategy",
             "innovation": "advantages", "feasibility": "feasibility"}
PIDS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]

INVEST_WEIGHT = 0.90
SCORE_THRESHOLD = 0.725
CONF_THRESHOLD = 0.811
TOL = 0.0006

CANON_PATH = BASE_DIR / "canonical" / "canonical_ai_scores_dataset1.json"
RUN2_DIR = BASE_DIR / "src" / "data" / "expert_reports"
RUN3_PATH = BASE_DIR / "canonical" / "stage6_repeatability_run3.json"
OUT_JSON = BASE_DIR / "canonical" / "stage6_repeatability_3run_analysis.json"
OUT_MD = BASE_DIR / "canonical" / "stage6_repeatability_3run_analysis.md"


def derive(invest: Dict[str, float], qa: Dict[str, float],
           weights: Dict[str, float], confidence: float) -> Dict[str, Any]:
    blended = {d: round(INVEST_WEIGHT * invest[d] + (1 - INVEST_WEIGHT) * qa[d], 4)
               for d in DIMS}
    num = sum(blended[d] * float(weights.get(d, 1.0)) for d in DIMS)
    den = sum(float(weights.get(d, 1.0)) for d in DIMS)
    overall = round(num / den, 4)
    if overall >= SCORE_THRESHOLD or confidence >= CONF_THRESHOLD:
        verdict = "GO"
    elif overall < 0.40 or blended["innovation"] < 0.30 or blended["feasibility"] < 0.30:
        verdict = "NO-GO"
    else:
        verdict = "HOLD"
    return {"invest": dict(invest), "blended": blended,
            "overall": overall, "confidence": confidence, "verdict": verdict}


def load_runs() -> Dict[str, Dict[str, Any]]:
    canon = json.loads(CANON_PATH.read_text(encoding="utf-8"))["canonical"]

    if not RUN3_PATH.exists():
        raise SystemExit(
            f"[FAIL] {RUN3_PATH} not found.\n"
            "       Run 3 has not been executed yet. Set OPENAI_API_KEY in .env and run:\n"
            "           python run_stage6_callb_run3.py")
    run3_raw = json.loads(RUN3_PATH.read_text(encoding="utf-8"))["run3"]

    out: Dict[str, Dict[str, Any]] = {}
    for pid in PIDS:
        c = canon[pid]
        qa, w, conf = c["qa_echo"], c["dim_weights"], float(c["confidence"])

        r1 = derive(c["invest_echo"], qa, w, conf)
        # Guard: the re-derivation must reproduce the published canonical numbers.
        assert abs(r1["overall"] - c["overall_ranking"]) <= TOL, (
            f"{pid}: re-derived run1 overall {r1['overall']} != canonical {c['overall_ranking']}")
        for d in DIMS:
            assert abs(r1["blended"][d] - c[CANON_DIM[d]]) <= TOL, (
                f"{pid}/{d}: re-derived run1 blended != canonical")
        assert ("Y" if r1["verdict"] == "GO" else "N") == c["verdict"], (
            f"{pid}: re-derived run1 verdict {r1['verdict']} != canonical {c['verdict']}")

        r2_file = RUN2_DIR / f"{pid}__openai" / "ai_expert_opinion.json"
        r2_json = json.loads(r2_file.read_text(encoding="utf-8"))
        r2_invest = {d: float(r2_json["dimensions"][d]["invest_score_echo"]) for d in DIMS}
        r2 = derive(r2_invest, qa, w, conf)

        if pid not in run3_raw:
            raise SystemExit(f"[FAIL] run3 file is missing proposal {pid}")
        r3_invest = {d: float(run3_raw[pid]["invest_scores_raw"][d]) for d in DIMS}
        r3 = derive(r3_invest, qa, w, conf)

        out[pid] = {"file_name": c.get("file_name", ""), "confidence": conf,
                    "run1": r1, "run2": r2, "run3": r3}
    return out


def stats3(vals: List[float]) -> Dict[str, Any]:
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals)  # sample std, ddof=1
    lo, hi = _bootstrap_ci(vals, statistics.fmean)
    return {"mean": round(mean, 4), "std": round(std, 4), "ci95": [lo, hi],
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "range": round(max(vals) - min(vals), 4)}


def main() -> int:
    runs = load_runs()
    print("[OK] run-1 re-derivation reproduces all 12 canonical overall scores, "
          "dimension scores and verdicts.\n")

    per_proposal: Dict[str, Any] = {}
    abs_devs_all: List[float] = []
    flips: List[Dict[str, Any]] = []
    same_verdict_count = 0

    for pid in PIDS:
        r = runs[pid]
        overalls = [r[f"run{i}"]["overall"] for i in (1, 2, 3)]
        verdicts = [r[f"run{i}"]["verdict"] for i in (1, 2, 3)]

        o = stats3(overalls)
        devs = [abs(v - o["mean"]) for v in overalls]
        abs_devs_all.extend(devs)

        dim_stats = {}
        for d in DIMS:
            blended_vals = [r[f"run{i}"]["blended"][d] for i in (1, 2, 3)]
            invest_vals = [r[f"run{i}"]["invest"][d] for i in (1, 2, 3)]
            dim_stats[d] = {
                "runs_blended": blended_vals,
                "blended": stats3(blended_vals),
                "runs_invest_raw": invest_vals,
                "invest_raw": stats3(invest_vals),
            }

        same = len(set(verdicts)) == 1
        same_verdict_count += int(same)
        if not same:
            flips.append({"pid": pid, "verdicts": verdicts,
                          "distinct": sorted(set(verdicts)),
                          "transition": " -> ".join(verdicts),
                          "overall_scores": overalls,
                          "confidence": r["confidence"]})

        per_proposal[pid] = {
            "file_name": r["file_name"],
            "confidence": r["confidence"],
            "overall_runs": overalls,
            "overall": o,
            "mean_abs_dev_from_3run_mean": round(statistics.fmean(devs), 4),
            "verdicts": {"run1": verdicts[0], "run2": verdicts[1], "run3": verdicts[2]},
            "same_verdict": same,
            "dimensions": dim_stats,
        }

    n = len(PIDS)
    mad_overall = round(statistics.fmean(abs_devs_all), 4)

    # dimension-level aggregates
    dim_agg = {}
    for d in DIMS:
        stds = [per_proposal[p]["dimensions"][d]["blended"]["std"] for p in PIDS]
        ranges = [per_proposal[p]["dimensions"][d]["blended"]["range"] for p in PIDS]
        n_var = sum(1 for p in PIDS
                    if len(set(per_proposal[p]["dimensions"][d]["runs_invest_raw"])) > 1)
        dim_agg[d] = {
            "mean_std_across_proposals": round(statistics.fmean(stds), 4),
            "max_range_across_proposals": round(max(ranges), 4),
            "n_proposals_with_any_variation": n_var,
            "pct_proposals_with_any_variation": round(100.0 * n_var / n, 1),
        }

    n_identical_all_dims = sum(
        1 for p in PIDS
        if all(len(set(per_proposal[p]["dimensions"][d]["runs_invest_raw"])) == 1 for d in DIMS))

    summary = {
        "n_proposals": n,
        "n_runs": 3,
        "same_verdict_count": same_verdict_count,
        "same_verdict_rate": round(same_verdict_count / n, 4),
        "flip_count": n - same_verdict_count,
        "flip_rate": round((n - same_verdict_count) / n, 4),
        "mean_absolute_deviation_from_3run_mean_overall": mad_overall,
        "max_abs_deviation_overall": round(max(abs_devs_all), 4),
        "mean_std_overall_across_proposals": round(
            statistics.fmean([per_proposal[p]["overall"]["std"] for p in PIDS]), 4),
        "max_overall_range_across_proposals": round(
            max(per_proposal[p]["overall"]["range"] for p in PIDS), 4),
        "n_proposals_identical_on_all_5_dimensions_across_3_runs": n_identical_all_dims,
        "n_proposals_with_any_dimension_variation": n - n_identical_all_dims,
        "flips": flips,
        "dimension_aggregates": dim_agg,
    }

    # ---------------- consolidated table ----------------
    hdr = (f"| {'proposal_id':<11} | {'run1':>6} | {'run2':>6} | {'run3':>6} | {'mean':>6} | "
           f"{'std':>6} | {'95% CI':>17} | {'verdict_run1':<12} | {'verdict_run2':<12} | "
           f"{'verdict_run3':<12} | same_verdict |")
    sep = ("|" + "-" * 13 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|"
           + "-" * 8 + "|" + "-" * 19 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 14
           + "|" + "-" * 14 + "|")
    lines = [hdr, sep]
    for pid in PIDS:
        pp = per_proposal[pid]
        o = pp["overall"]
        lo, hi = o["ci95"]
        ci = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "n/a"
        v = pp["verdicts"]
        lines.append(
            f"| {pid:<11} | {pp['overall_runs'][0]:>6.4f} | {pp['overall_runs'][1]:>6.4f} | "
            f"{pp['overall_runs'][2]:>6.4f} | {o['mean']:>6.4f} | {o['std']:>6.4f} | "
            f"{ci:>17} | {v['run1']:<12} | {v['run2']:<12} | {v['run3']:<12} | "
            f"{'Y' if pp['same_verdict'] else 'N':^12} |")
    table = "\n".join(lines)

    print(table)
    print()
    print(f"Same-verdict across all 3 runs : {same_verdict_count}/{n} "
          f"({100*summary['same_verdict_rate']:.1f}%)")
    print(f"Verdict flip rate              : {summary['flip_count']}/{n} "
          f"({100*summary['flip_rate']:.1f}%)")
    print(f"MAD from 3-run mean (overall)  : {mad_overall:.4f}")
    print(f"Mean per-proposal std (overall): {summary['mean_std_overall_across_proposals']:.4f}")
    print(f"Proposals identical on all 5 dims across 3 runs: "
          f"{n_identical_all_dims}/{n}")
    if flips:
        print("\nVerdict flips:")
        for f in flips:
            print(f"  {f['pid']}: {f['transition']}  "
                  f"(overall {', '.join(f'{s:.4f}' for s in f['overall_scores'])}; "
                  f"conf {f['confidence']:.4f})")

    # ---------------- write outputs ----------------
    run3_prov = json.loads(RUN3_PATH.read_text(encoding="utf-8"))["provenance"]
    OUT_JSON.write_text(json.dumps({
        "analysis": "stage6_call_b_3run_repeatability",
        "method": {
            "invest_weight": INVEST_WEIGHT,
            "verdict_rule": (f"GO iff overall >= {SCORE_THRESHOLD} OR confidence >= "
                             f"{CONF_THRESHOLD}; NO-GO iff overall < 0.40 or blended "
                             f"innovation < 0.30 or blended feasibility < 0.30; else HOLD"),
            "verdict_rule_applied_uniformly_to_all_runs": True,
            "std": "sample standard deviation (ddof=1), n=3",
            "ci": ("percentile bootstrap, 1000 resamples, numpy default_rng(42); "
                   "_bootstrap_ci imported unchanged from "
                   "src/tools/evaluate_cohens_kappa.py"),
            "ci_caveat": ("with n=3 the bootstrap CI can only span the observed values "
                          "and is reported for method consistency, not as a precise "
                          "interval estimate"),
            "frozen_across_runs": ["qa_echo (Stage 5)", "dim_weights", "confidence (Stage 5)",
                                   "dimensions_v2.json evidence input"],
        },
        "sources": {
            "run1": "canonical/canonical_ai_scores_dataset1.json (Jun-16 baseline)",
            "run2": "src/data/expert_reports/<pid>__openai/ai_expert_opinion.json (Jun-18)",
            "run3": "canonical/stage6_repeatability_run3.json",
            "run3_provenance": run3_prov,
        },
        "summary": summary,
        "per_proposal": per_proposal,
        "consolidated_table": table,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Stage 6 Call B — 3-run repeatability (Dataset1, n=12)",
        "",
        f"- Call: deterministic investment-grade scoring, `temperature=0`, `seed=42`, "
        f"identical `dimensions_v2.json` evidence for all three runs.",
        f"- run1 = Jun-16 canonical baseline · run2 = Jun-18 re-run · "
        f"run3 = {run3_prov.get('date_local', '')} "
        f"(model `{run3_prov.get('model', '')}`).",
        "- Overall score and verdict re-derived for all three runs with one identical "
        "formula and verdict rule, so deltas isolate Call B.",
        "",
        "## Consolidated table",
        "",
        table,
        "",
        "## Summary statistics",
        "",
        f"- **Same-verdict rate (identical GO/HOLD/NO-GO across all 3 runs):** "
        f"{same_verdict_count}/{n} = {100*summary['same_verdict_rate']:.1f}%",
        f"- **Verdict flip rate (≥1 differing verdict):** {summary['flip_count']}/{n} = "
        f"{100*summary['flip_rate']:.1f}%",
        f"- **Mean absolute deviation from 3-run mean overall_score:** {mad_overall:.4f}",
        f"- Max absolute deviation: {summary['max_abs_deviation_overall']:.4f}",
        f"- Mean per-proposal std (overall): "
        f"{summary['mean_std_overall_across_proposals']:.4f}",
        f"- Proposals with identical scores on all 5 dimensions across all 3 runs: "
        f"{n_identical_all_dims}/{n}",
        "",
        "## Verdict flips",
        "",
    ]
    if flips:
        md.append("| proposal_id | run1 | run2 | run3 | overall scores | confidence |")
        md.append("|---|---|---|---|---|---|")
        for f in flips:
            md.append(f"| {f['pid']} | {f['verdicts'][0]} | {f['verdicts'][1]} | "
                      f"{f['verdicts'][2]} | "
                      f"{', '.join(f'{s:.4f}' for s in f['overall_scores'])} | "
                      f"{f['confidence']:.4f} |")
    else:
        md.append("_No proposal changed verdict across the three runs._")
    md += ["", "## Per-dimension variation (blended dimension scores)", "",
           "| dimension | mean std | max range | proposals with any variation |",
           "|---|---:|---:|---:|"]
    for d in DIMS:
        a = dim_agg[d]
        md.append(f"| {d} | {a['mean_std_across_proposals']:.4f} | "
                  f"{a['max_range_across_proposals']:.4f} | "
                  f"{a['n_proposals_with_any_variation']}/{n} "
                  f"({a['pct_proposals_with_any_variation']:.1f}%) |")
    md += ["", "## Per-proposal dimension detail", ""]
    for pid in PIDS:
        md.append(f"### {pid}")
        md.append("")
        md.append("| dimension | run1 | run2 | run3 | mean | std | 95% CI |")
        md.append("|---|---:|---:|---:|---:|---:|---|")
        for d in DIMS:
            ds = per_proposal[pid]["dimensions"][d]
            b, v = ds["blended"], ds["runs_blended"]
            lo, hi = b["ci95"]
            ci = f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "n/a"
            md.append(f"| {d} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | "
                      f"{b['mean']:.4f} | {b['std']:.4f} | {ci} |")
        md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"\n[OK] wrote {OUT_JSON}")
    print(f"[OK] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
