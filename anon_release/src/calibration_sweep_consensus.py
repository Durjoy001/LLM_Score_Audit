#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4 — Re-run the calibration_sweep.py logic against the 3-rater CONSENSUS
ground truth instead of the original single-rater scores.

The existing calibration_sweep.py hardcodes HUMAN_RAW (the single rater). To
avoid changing that script's logic, this wrapper imports it unchanged and only
swaps HUMAN_RAW for the consensus values built by build_consensus_ground_truth.py.
Every statistic (Spearman sweep over INVEST_WEIGHT, 2-D tau_r x tau_c OR-logic
verdict grid) is computed by the original functions.

Zero LLM calls — reads cached invest_score_echo / qa_score_echo only.

Output: resultsNew/Dataset1/calibration_sweep_consensus.json
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import calibration_sweep as cs

BASE = Path(__file__).resolve().parent
RES = BASE / "resultsNew" / "Dataset1"
CONSENSUS_JSON = RES / "consensus_build.json"

# Maximum AI overall score observed anywhere in the 141-proposal Kickstarter set.
KICKSTARTER_MAX_AI_SCORE = 0.640
PROD_TAU_R, PROD_TAU_C = 0.725, 0.811


def build_consensus_human_raw():
    data = json.loads(CONSENSUS_JSON.read_text(encoding="utf-8"))["consensus"]
    human_raw = {}
    for pid, c in data.items():
        human_raw[pid] = dict(
            team=c["team"], objective=c["objective"], strategy=c["strategy"],
            advantages=c["advantages"], feasibility=c["feasibility"],
            rank=c["overall_20_mean"], verdict=c["verdict_majority"],
        )
    return human_raw


def main():
    # ---- swap in consensus ground truth (logic untouched) ----
    cs.HUMAN_RAW = build_consensus_human_raw()

    ai_raw = cs.load_ai_raw()
    results = cs.sweep(ai_raw)

    # Spearman depends only on INVEST_WEIGHT -> collapse to one row per weight.
    per_weight = {}
    for r in results:
        iw = r["invest_weight"]
        per_weight.setdefault(iw, r)  # spearman identical across vt for a given iw
    weight_table = [
        {"invest_weight": iw,
         "spearman": per_weight[iw]["spearman"],
         "concordance": per_weight[iw]["concordance"],
         "ai_range": per_weight[iw]["ai_range"]}
        for iw in sorted(per_weight)
    ]
    peak_sp = max(w["spearman"] for w in weight_table)
    peak_weights = [w["invest_weight"] for w in weight_table if abs(w["spearman"] - peak_sp) < 1e-9]
    prod_row = next((w for w in weight_table if abs(w["invest_weight"] - 0.90) < 1e-9), None)
    prod_ties_peak = prod_row is not None and abs(prod_row["spearman"] - peak_sp) < 1e-9

    best_iw = peak_weights[0]
    with redirect_stdout(io.StringIO()):
        best_vt = cs.sweep_verdict_thresholds(ai_raw, best_iw)
    tau_r = best_vt["t_overall"]
    tau_c = best_vt["t_conf"]

    # Global fine 2-D threshold search across ALL invest weights: the verdict
    # accuracy maximizer, and the smallest tau_r among the maximizers (the most
    # favorable case for the score ever reaching the Kickstarter range).
    vacc_by_iw = []
    for iw in [round(w * 0.05, 2) for w in range(21)]:
        with redirect_stdout(io.StringIO()):
            r = cs.sweep_verdict_thresholds(ai_raw, iw)
        vacc_by_iw.append({"invest_weight": iw, "tau_r": r["t_overall"],
                           "tau_c": r["t_conf"], "verdict_acc": round(r["verdict_acc"], 4)})
    max_vacc = max(v["verdict_acc"] for v in vacc_by_iw)
    maximizers = [v for v in vacc_by_iw if abs(v["verdict_acc"] - max_vacc) < 1e-9]
    min_tau_r_maximizer = min(maximizers, key=lambda v: v["tau_r"])
    all_maximizers_above_ceiling = all(v["tau_r"] > KICKSTARTER_MAX_AI_SCORE for v in maximizers)

    # ---- the critical 0.640 comparison ----
    if tau_r > KICKSTARTER_MAX_AI_SCORE:
        transfer_status = (
            f"tau_r={tau_r} > {KICKSTARTER_MAX_AI_SCORE} (max Kickstarter AI score): "
            "threshold still sits ABOVE the entire Kickstarter score range -> the "
            "'threshold transfers with 0% recall' finding STILL HOLDS."
        )
        finding_holds = True
    else:
        transfer_status = (
            f"tau_r={tau_r} <= {KICKSTARTER_MAX_AI_SCORE} (max Kickstarter AI score): "
            "threshold now falls WITHIN the Kickstarter score range -> at least one "
            "Kickstarter proposal would be classified GO. The '0% recall' finding "
            "NEEDS REWRITING."
        )
        finding_holds = False

    out = {
        "ground_truth": "3-rater consensus (median dims, mean overall/20, majority verdict)",
        "invest_weight_sweep": weight_table,
        "invest_weight_peak_spearman": peak_sp,
        "invest_weight_peak_weights": peak_weights,
        "production_invest_weight": 0.90,
        "production_iw_ties_peak": prod_ties_peak,
        "best_threshold": {
            "invest_weight": best_iw,
            "tau_r": tau_r,
            "tau_c": tau_c,
            "verdict_acc": best_vt["verdict_acc"],
            "logic": "GO if overall >= tau_r OR conf >= tau_c",
            "note": "thresholds at the Spearman-peak invest_weight",
        },
        "verdict_accuracy_maximizer": {
            "max_verdict_acc": max_vacc,
            "n_maximizing_invest_weights": len(maximizers),
            "smallest_tau_r_among_maximizers": min_tau_r_maximizer,
            "all_maximizers_tau_r_above_kickstarter_ceiling": all_maximizers_above_ceiling,
            "per_invest_weight": vacc_by_iw,
        },
        "kickstarter_max_ai_score": KICKSTARTER_MAX_AI_SCORE,
        "production_threshold": {"tau_r": PROD_TAU_R, "tau_c": PROD_TAU_C},
        "zero_recall_finding_holds": finding_holds,
        "transfer_status": transfer_status,
        "full_sweep": results,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "calibration_sweep_consensus.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("INVEST_WEIGHT sweep (Spearman vs consensus overall ranking):")
    print(f"{'IW':>5s} {'Spearman':>9s} {'Conc':>7s} {'AIrng':>7s}")
    for w in weight_table:
        star = "  <-- peak" if abs(w["spearman"] - peak_sp) < 1e-9 else ""
        print(f"{w['invest_weight']:5.2f} {w['spearman']:+9.4f} {w['concordance']:7.4f} {w['ai_range']:7.4f}{star}")
    print(f"\nPeak Spearman = {peak_sp:+.4f} at IW = {peak_weights}")
    print(f"Production IW=0.90 ties the peak? {prod_ties_peak}  "
          f"(IW=0.90 spearman = {prod_row['spearman'] if prod_row else 'n/a'})")
    print(f"\nBest verdict threshold @ Spearman-peak IW={best_iw}: tau_r={tau_r}  tau_c={tau_c}  "
          f"verdict_acc={best_vt['verdict_acc']:.0%}")
    print(f"Global max verdict_acc={max_vacc:.0%}; smallest tau_r among maximizers = "
          f"{min_tau_r_maximizer['tau_r']} (IW={min_tau_r_maximizer['invest_weight']}). "
          f"All maximizer tau_r above {KICKSTARTER_MAX_AI_SCORE}? {all_maximizers_above_ceiling}")
    print(f"\n*** {transfer_status} ***")


if __name__ == "__main__":
    main()
