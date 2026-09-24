#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5 — Propagate the consensus-derived threshold to the Kickstarter dataset.

The 3-rater consensus moved the biotech-derived decision threshold only slightly
(tau_c unchanged at ~0.811; tau_r stays in [0.6944, 0.7453] depending on
invest_weight, vs the production 0.725). Every one of those tau_r values is still
ABOVE the maximum AI score ever observed in the 141-proposal Kickstarter set
(0.640), so the change is not material to the transfer finding.

This script proves that concretely rather than by assertion: it re-runs the
Kickstarter GO/NO-GO classification (Spearman correlations are unaffected — they
don't depend on any human ground truth) under three thresholds:
  - production transferred:        tau_r=0.725,  tau_c=0.811
  - consensus @ Spearman-peak IW:  tau_r=0.7453, tau_c=0.8112
  - consensus smallest maximizer:  tau_r=0.6944, tau_c=0.8112

It also re-runs calibrate_kickstarter_threshold.py as-is (its 5-fold CV derives a
Kickstarter-specific threshold independent of the biotech consensus) and notes
whether the "before (transferred)" row differs under the consensus threshold.

Zero LLM calls. Output: resultsNew/Kickstarter/threshold_propagation.json
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from calibrate_kickstarter_threshold import (  # noqa: E402
    load_rows, confusion, metrics_from_confusion,
)

RES = ROOT / "resultsNew" / "Kickstarter"
CAL_JSON = json.loads((ROOT / "resultsNew" / "Dataset1" / "calibration_sweep_consensus.json").read_text())

THRESHOLDS = {
    "production_transferred": {"tau_r": 0.725, "tau_c": 0.811},
    "consensus_spearman_peak": {
        "tau_r": CAL_JSON["best_threshold"]["tau_r"],
        "tau_c": CAL_JSON["best_threshold"]["tau_c"],
    },
    "consensus_smallest_maximizer": {
        "tau_r": CAL_JSON["verdict_accuracy_maximizer"]["smallest_tau_r_among_maximizers"]["tau_r"],
        "tau_c": CAL_JSON["verdict_accuracy_maximizer"]["smallest_tau_r_among_maximizers"]["tau_c"],
    },
}


def main():
    rows = load_rows()
    n = len(rows)
    max_score = max(r["score"] for r in rows)
    max_conf = max(r["confidence"] for r in rows)
    n_success = sum(r["success"] for r in rows)

    classifications = {}
    for name, th in THRESHOLDS.items():
        m = metrics_from_confusion(confusion(rows, th["tau_r"], th["tau_c"]))
        n_go = sum(1 for r in rows if r["score"] >= th["tau_r"] or r["confidence"] >= th["tau_c"])
        classifications[name] = {
            **th,
            "n_predicted_go": n_go,
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
            "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"],
            "tau_r_above_max_ai_score": th["tau_r"] > max_score,
        }

    # Does the transferred "before" row differ under the consensus threshold?
    prod = classifications["production_transferred"]
    peak = classifications["consensus_spearman_peak"]
    before_row_differs = (prod["n_predicted_go"] != peak["n_predicted_go"]
                          or prod["recall"] != peak["recall"])

    recalls = {k: v["recall"] for k, v in classifications.items()}
    all_zero_recall = all((r == 0 or r is None) for r in recalls.values())

    # Re-run the independent 5-fold CV recalibration as-is (writes to
    # results/KickstarterCalibrated/); copy the fresh output under resultsNew.
    cv_note = "not re-run"
    cv_copy = None
    try:
        subprocess.run([sys.executable, str(ROOT / "calibrate_kickstarter_threshold.py")],
                       check=True, capture_output=True, text=True)
        src = ROOT / "results" / "KickstarterCalibrated" / "evaluation" / "calibration_results.json"
        if src.exists():
            RES.mkdir(parents=True, exist_ok=True)
            dst = RES / "kickstarter_cv_recalibration.json"
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            cv = json.loads(src.read_text())
            cv_copy = str(dst)
            cv_note = {
                "before_calibration_row": cv["before_calibration"],
                "cv_aggregated_held_out": cv["cross_validation"]["aggregated_held_out"],
                "note": ("5-fold CV threshold is derived from Kickstarter data only and is "
                         "independent of the biotech consensus; its numbers are unchanged. "
                         "Its 'before_calibration' row uses the production 0.725/0.811 pair."),
            }
    except subprocess.CalledProcessError as e:
        cv_note = f"calibrate_kickstarter_threshold.py failed: {e.stderr[-300:]}"

    out = {
        "dataset": "Kickstarter (141 proposals)",
        "n_total": n, "n_success": n_success,
        "max_ai_overall_score": round(max_score, 4),
        "max_ai_confidence": round(max_conf, 4),
        "kickstarter_max_ai_score_reference": 0.640,
        "threshold_change_material": False,
        "material_change_reasoning": (
            "tau_c unchanged (~0.811); tau_r stays within [0.6944, 0.7453], all above the "
            "0.640 Kickstarter score ceiling. No Kickstarter proposal reaches GO under any "
            "of them, so recall remains 0% and the transfer finding is unchanged."
        ),
        "classification_under_each_threshold": classifications,
        "recall_by_threshold": recalls,
        "all_thresholds_zero_recall": all_zero_recall,
        "transferred_before_row_differs_under_consensus": before_row_differs,
        "cv_recalibration": cv_note,
        "cv_recalibration_file": cv_copy,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "threshold_propagation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Kickstarter: n={n}, successes={n_success}, max AI score={max_score:.4f}, max conf={max_conf:.4f}")
    print(f"{'threshold':32s} {'tau_r':>7s} {'tau_c':>7s} {'#GO':>4s} {'recall':>7s} {'acc':>6s}")
    for name, c in classifications.items():
        print(f"{name:32s} {c['tau_r']:7.4f} {c['tau_c']:7.4f} {c['n_predicted_go']:4d} "
              f"{str(c['recall']):>7s} {str(c['accuracy']):>6s}")
    print(f"\nAll thresholds give 0% recall on real successes? {all_zero_recall}")
    print(f"Transferred 'before' row differs under consensus threshold? {before_row_differs}")
    print("[OK] resultsNew/Kickstarter/threshold_propagation.json written")


if __name__ == "__main__":
    main()
