#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-provider replication analysis (paper fix B3) — OpenAI vs Gemini, Dataset1.

Pure re-derivation from cached artifacts. NO LLM calls.

Inputs
------
  canonical/canonical_ai_scores_dataset1.json   OpenAI arm (frozen Jun-16)
  canonical/cross_provider_gemini_raw.json      Gemini arm (2 runs, this study)
  canonical/stage6_repeatability_3run_analysis.json   within-OpenAI noise floor
  canonical/rule_compliance_audit.json          declared band ranges + OpenAI baseline
  resultsNew/Dataset1/consensus_build.json      3-rater panel consensus
  resultsNew/Dataset1/raters_long.csv           per-rater scores (for LORO)

Metric functions are imported from the production evaluation module so the
Gemini numbers are computed by exactly the code that produced the published
OpenAI numbers (rho=0.1747, verdict accuracy 0.8333).

Output
------
  canonical/cross_provider_gemini.json          per-proposal results + provenance
  canonical/cross_provider_gemini.md            summary tables for the paper

Nothing under results/ or resultsNew/ is written, and no existing canonical
file is modified.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import spearmanr

from src.tools.evaluate_cohens_kappa import _column_metrics, _rank_order_metrics

ROOT = Path(__file__).resolve().parent
CANON = json.loads((ROOT / "canonical" / "canonical_ai_scores_dataset1.json").read_text())["canonical"]
GEM = json.loads((ROOT / "canonical" / "cross_provider_gemini_raw.json").read_text())
REPEAT = json.loads((ROOT / "canonical" / "stage6_repeatability_3run_analysis.json").read_text())
RULES = json.loads((ROOT / "canonical" / "rule_compliance_audit.json").read_text())
CONS = json.loads((ROOT / "resultsNew" / "Dataset1" / "consensus_build.json").read_text())["consensus"]
RATERS_CSV = ROOT / "resultsNew" / "Dataset1" / "raters_long.csv"

PIDS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]
# human-rubric dimension name -> Stage 6 internal dimension name
H2L = {"team": "team", "objective": "objectives", "strategy": "strategy",
       "advantages": "innovation", "feasibility": "feasibility"}
DIMS = list(H2L)
LOG_DIMS = list(H2L.values())
RANK = "overall_ranking"
RUNS = ["gemini_run1", "gemini_run2"]

BANDS = RULES["provenance"]["band_ranges_parsed_from_prompt"]
DIM_PREFIX = {"team": "T", "objectives": "O", "strategy": "S",
              "innovation": "I", "feasibility": "F"}


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def hnorm(v, is_rank=False):
    """Human rubric -> 0-1. Integer-rounded ranks per evaluate_cohens_kappa convention."""
    return round((round(v) - 1) / 19, 4) if is_rank else round((v - 1) / 4, 4)


def gem_dim(rec, human_dim):
    return rec["dimension_scores_blended"][H2L[human_dim]]


# --------------------------------------------------------------- 1. per-proposal
def per_proposal():
    out = {}
    for pid in PIDS:
        o = CANON[pid]
        rec = {"pid": pid, "file_name": o["file_name"],
               "openai": {"overall": o[RANK], "verdict": o["verdict"],
                          "dimensions": {d: o[d] for d in DIMS}}}
        for r in RUNS:
            g = GEM[r][pid]
            v = "Y" if g["verdict"] == "GO" else "N"
            rec[r] = {
                "overall": g["overall_score"],
                "verdict_raw": g["verdict"],
                "verdict": v,
                "dimensions": {d: gem_dim(g, d) for d in DIMS},
                "categories": g["categories"],
                "delta_overall": round(g["overall_score"] - o[RANK], 4),
                "delta_dimensions": {d: round(gem_dim(g, d) - o[d], 4) for d in DIMS},
                "verdict_matches_openai": v == o["verdict"],
            }
        rec["gemini_run1_vs_run2"] = {
            "overall_abs_diff": round(abs(GEM["gemini_run1"][pid]["overall_score"]
                                          - GEM["gemini_run2"][pid]["overall_score"]), 4),
            "dimension_abs_diff": {
                d: round(abs(gem_dim(GEM["gemini_run1"][pid], d)
                             - gem_dim(GEM["gemini_run2"][pid], d)), 4) for d in DIMS},
        }
        out[pid] = rec
    return out


# --------------------------------------------------------------- 3. replication
def consensus_eval(score_of, verdict_of, label):
    """Reproduces rebuild_canonical_resultsNew.consensus_eval for an arbitrary arm."""
    pairs = {c: [] for c in DIMS + [RANK]}
    rows, vmatch = [], 0
    for pid in PIDS:
        c = CONS[pid]
        row = {"file_name": CANON[pid]["file_name"], "pid": pid}
        for d in DIMS:
            hv, av = hnorm(c[d]), score_of(pid, d)
            row[f"human_{d}"], row[f"ai_{d}"] = hv, av
            pairs[d].append((hv, av))
        hv, av = hnorm(c["overall_20_mean"], is_rank=True), score_of(pid, RANK)
        row[f"human_{RANK}"], row[f"ai_{RANK}"] = hv, av
        pairs[RANK].append((hv, av))
        rows.append(row)
        if c["verdict_majority"] == verdict_of(pid):
            vmatch += 1
    metrics = {}
    for d in DIMS:
        m = _column_metrics(pairs[d], is_ranking=False)
        metrics[d] = {k: _clean(m.get(k)) for k in
                      ("weighted_kappa", "spearman_r", "spearman_p", "mean_diff", "mean_human", "mean_ai")}
    mr = _column_metrics(pairs[RANK], is_ranking=True)
    metrics[RANK] = {k: _clean(mr.get(k)) for k in
                     ("icc", "spearman_r", "spearman_p", "weighted_kappa", "mean_diff")}
    ro = _rank_order_metrics(rows, RANK)
    return {"arm": label, "verdict_accuracy": round(vmatch / 12, 4), "metrics": metrics,
            "rank_order": {"pairwise_rank_concordance": ro["pairwise_rank_concordance"],
                           "pairwise_comparable": ro["pairwise_comparable"]}}


def load_raters():
    raters = {t: {1: {}, 2: {}, 3: {}} for t in DIMS + [RANK]}
    id2pid = {i: p for i, p in enumerate(PIDS, 1)}
    with RATERS_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raters[r["dimension"]][int(r["rater_id"])][id2pid[int(r["proposal_id"])]] = float(r["score"])
    return raters


def loro(score_of, label):
    """Reproduces rebuild_canonical_resultsNew.loro for an arbitrary arm."""
    raters = load_raters()

    def sp(xs, ys):
        if len(xs) < 3:
            return None
        rho, _ = spearmanr(xs, ys)
        return None if rho != rho else round(float(rho), 4)

    out = {"arm": label, "targets": {}}
    for t in DIMS + [RANK]:
        agg = statistics.mean if t == RANK else statistics.median
        cons_all = [agg([raters[t][r][p] for r in (1, 2, 3)]) for p in PIDS]
        ai_rho = sp([score_of(p, t) for p in PIDS], cons_all)
        lv = {}
        for r in (1, 2, 3):
            others = [x for x in (1, 2, 3) if x != r]
            lv[f"rater{r}_vs_others"] = sp([raters[t][r][p] for p in PIDS],
                                           [agg([raters[t][o][p] for o in others]) for p in PIDS])
        band = [v for v in lv.values() if v is not None]
        lo, hi = (min(band), max(band)) if band else (None, None)
        pos = "undefined" if (ai_rho is None or not band) else (
            "BELOW human band" if ai_rho < lo else
            "ABOVE human band" if ai_rho > hi else "INSIDE human band")
        out["targets"][t] = {"ai_vs_consensus_rho": ai_rho, "human_band": [lo, hi],
                             "loro": lv, "ai_position": pos}
    return out


# --------------------------------------------------------------- 4. noise floor
def noise_context(pp):
    """Cross-provider deltas vs the measured within-provider repeatability floor."""
    oa_overall_mad = REPEAT["summary"]["mean_absolute_deviation_from_3run_mean_overall"]
    oa_overall_max_range = REPEAT["summary"]["max_overall_range_across_proposals"]
    oa_dim_max_range = {d: REPEAT["summary"]["dimension_aggregates"][d]["max_range_across_proposals"]
                        for d in LOG_DIMS}

    cross_overall = [abs(pp[p][r]["delta_overall"]) for p in PIDS for r in RUNS]
    within_gem_overall = [pp[p]["gemini_run1_vs_run2"]["overall_abs_diff"] for p in PIDS]

    per_dim = {}
    for d in DIMS:
        cross = [abs(pp[p][r]["delta_dimensions"][d]) for p in PIDS for r in RUNS]
        within = [pp[p]["gemini_run1_vs_run2"]["dimension_abs_diff"][d] for p in PIDS]
        floor = oa_dim_max_range[H2L[d]]
        per_dim[d] = {
            "cross_provider_mean_abs_delta": round(statistics.mean(cross), 4),
            "cross_provider_median_abs_delta": round(statistics.median(cross), 4),
            "cross_provider_max_abs_delta": round(max(cross), 4),
            "within_openai_3run_max_range": floor,
            "within_gemini_2run_max_range": round(max(within), 4),
            "n_cells_exceeding_openai_floor": sum(1 for c in cross if c > floor),
            "n_cells": len(cross),
            "median_delta_exceeds_floor": statistics.median(cross) > floor,
        }
    return {
        "within_openai_3run_floor": {
            "overall_mad": oa_overall_mad,
            "overall_max_abs_deviation": REPEAT["summary"]["max_abs_deviation_overall"],
            "overall_max_range": oa_overall_max_range,
            "dimension_max_range": oa_dim_max_range,
        },
        "overall": {
            "cross_provider_mean_abs_delta": round(statistics.mean(cross_overall), 4),
            "cross_provider_median_abs_delta": round(statistics.median(cross_overall), 4),
            "cross_provider_max_abs_delta": round(max(cross_overall), 4),
            "within_gemini_2run_mean_abs_diff": round(statistics.mean(within_gem_overall), 4),
            "within_gemini_2run_max_abs_diff": round(max(within_gem_overall), 4),
            "n_exceeding_openai_max_range": sum(1 for c in cross_overall if c > oa_overall_max_range),
            "n_cells": len(cross_overall),
        },
        "per_dimension": per_dim,
    }


# --------------------------------------------------------------- 5. rule compliance
def rule_compliance():
    checked = compliant = 0
    invalid, violations = [], []
    per_dim = {d: {"n": 0, "ok": 0} for d in LOG_DIMS}
    per_band = {}
    for r in RUNS:
        for pid in PIDS:
            g = GEM[r][pid]
            cats, scores = g["categories"], g["invest_scores_raw"]
            for ld in LOG_DIMS:
                cat = (cats.get(ld) or "").strip()
                sc = scores.get(ld)
                if sc is None:
                    continue
                if cat not in BANDS:
                    invalid.append({"run": r, "pid": pid, "dimension": ld, "returned": cat,
                                    "allowed": sorted(k for k in BANDS if k.startswith(DIM_PREFIX[ld] + "-"))})
                    continue
                lo, hi = BANDS[cat]
                checked += 1
                per_dim[ld]["n"] += 1
                b = per_band.setdefault(cat, {"n": 0, "ok": 0, "range": [lo, hi]})
                b["n"] += 1
                if lo - 1e-9 <= sc <= hi + 1e-9:
                    compliant += 1
                    per_dim[ld]["ok"] += 1
                    b["ok"] += 1
                else:
                    violations.append({"run": r, "pid": pid, "dimension": ld,
                                       "category": cat, "score": sc, "band": [lo, hi]})
    for d in per_dim:
        per_dim[d]["rate"] = round(per_dim[d]["ok"] / per_dim[d]["n"], 4) if per_dim[d]["n"] else None
    for b in per_band.values():
        b["rate"] = round(b["ok"] / b["n"], 4)

    # AI-powered innovation cap (RULE 1 / I-C hard cap). p1 is the prompt's own
    # worked example: an AI clinical decision-support system where AI IS the product.
    cap = {"rule": "RULE 1 / I-C hard cap: AI-as-the-product -> I-C, score 0.42-0.56, NO EXCEPTIONS",
           "precondition_proposal": "p1 (heart-failure clinical decision-support large model)",
           "openai_baseline": RULES["check5_ai_cap_rule1"],
           "gemini": {}}
    for r in RUNS:
        g = GEM[r]["p1"]
        cat = g["categories"].get("innovation")
        sc = g["invest_scores_raw"]["innovation"]
        cap["gemini"][r] = {
            "category": cat, "score": sc,
            "rule_applied": cat == "I-C" and 0.42 - 1e-9 <= sc <= 0.56 + 1e-9,
            "rationale": (g["rationale"] or {}).get("innovation", ""),
        }
    cap["binds_under_gemini"] = all(v["rule_applied"] for v in cap["gemini"].values())

    ic_cells = [(r, pid, GEM[r][pid]["invest_scores_raw"]["innovation"])
                for r in RUNS for pid in PIDS
                if (GEM[r][pid]["categories"].get("innovation") or "").strip() == "I-C"]
    cap["all_I_C_cells"] = {
        "n": len(ic_cells),
        "inside_header_range_0.40_0.60": sum(1 for _, _, s in ic_cells if 0.40 <= s <= 0.60),
        "inside_cap_range_0.42_0.56": sum(1 for _, _, s in ic_cells if 0.42 <= s <= 0.56),
        "note": ("the prompt states two different numeric ranges for I-C (header 0.40-0.60 "
                 "vs RULE 1 / hard cap 0.42-0.56); OpenAI cap-range compliance was 0.4524"),
    }

    return {
        "check_schema_validity": {"invalid_cells": len(invalid), "total_cells": checked + len(invalid),
                                  "rate": round(len(invalid) / (checked + len(invalid)), 4) if (checked + len(invalid)) else 0,
                                  "violations": invalid},
        "check_score_within_declared_band": {
            "checked": checked, "compliant": compliant,
            "rate": round(compliant / checked, 4) if checked else None,
            "openai_baseline_rate": RULES["check2_score_within_declared_band"]["rate"],
            "per_dimension": per_dim, "per_band": per_band, "violations": violations},
        "check_ai_cap": cap,
        "check_band_firing": {
            "note": "top ('A') band firing rate across all Gemini calls",
            "per_dimension": {
                ld: {"A_fired": sum(1 for r in RUNS for pid in PIDS
                                    if (GEM[r][pid]["categories"].get(ld) or "").strip()
                                    == f"{DIM_PREFIX[ld]}-A"),
                     "of": len(RUNS) * len(PIDS)} for ld in LOG_DIMS},
            "openai_baseline": RULES["check6_band_firing"]["per_dimension"]},
    }


def main():
    pp = per_proposal()

    openai_eval = consensus_eval(
        lambda p, t: CANON[p][t], lambda p: CANON[p]["verdict"], "openai_canonical")
    openai_loro = loro(lambda p, t: CANON[p][t], "openai_canonical")

    gem_eval, gem_loro = {}, {}
    for r in RUNS:
        gem_eval[r] = consensus_eval(
            lambda p, t, r=r: (GEM[r][p]["overall_score"] if t == RANK else gem_dim(GEM[r][p], t)),
            lambda p, r=r: ("Y" if GEM[r][p]["verdict"] == "GO" else "N"), r)
        gem_loro[r] = loro(
            lambda p, t, r=r: (GEM[r][p]["overall_score"] if t == RANK else gem_dim(GEM[r][p], t)), r)

    vagree = {r: {"n_matching": sum(1 for p in PIDS if pp[p][r]["verdict_matches_openai"]),
                  "of": 12,
                  "differing": [{"pid": p, "openai": CANON[p]["verdict"],
                                 "gemini": pp[p][r]["verdict"],
                                 "openai_overall": CANON[p][RANK],
                                 "gemini_overall": pp[p][r]["overall"],
                                 "confidence": GEM[r][p]["confidence"]}
                                for p in PIDS if not pp[p][r]["verdict_matches_openai"]]}
              for r in RUNS}
    vagree["both_runs_agree_with_openai"] = sum(
        1 for p in PIDS if all(pp[p][r]["verdict_matches_openai"] for r in RUNS))
    vagree["gemini_run1_vs_run2_agreement"] = sum(
        1 for p in PIDS if pp[p]["gemini_run1"]["verdict"] == pp[p]["gemini_run2"]["verdict"])

    payload = {
        "provenance": {
            **GEM["provenance"],
            "analysis_date_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_llm_calls": 0,
            "metric_code": ("src.tools.evaluate_cohens_kappa._column_metrics / "
                            "_rank_order_metrics — the same functions that produced the "
                            "published OpenAI numbers"),
            "ground_truth": ("3-rater panel consensus (median dims, mean overall/20, "
                             "majority verdict) from resultsNew/Dataset1/consensus_build.json"),
            "openai_comparison_arm": "canonical/canonical_ai_scores_dataset1.json (frozen Jun-16)",
            "verdict_mapping": "GO -> Y; HOLD/NO-GO -> N (canonical convention)",
        },
        "per_proposal": pp,
        "verdict_agreement": vagree,
        "replication": {
            "openai_canonical": openai_eval,
            "gemini": gem_eval,
            "loro": {"openai_canonical": openai_loro, "gemini": gem_loro},
        },
        "noise_context": noise_context(pp),
        "rule_compliance": rule_compliance(),
    }

    out = ROOT / "canonical" / "cross_provider_gemini.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")
    return payload


if __name__ == "__main__":
    main()
