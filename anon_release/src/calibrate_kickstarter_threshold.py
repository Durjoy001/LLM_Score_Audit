#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate a Kickstarter-specific GO decision threshold using stratified 5-fold
cross-validation, and compare it against the production threshold (score >= 0.725
OR confidence >= 0.811, from `verdict_rule()` in ai_expert_opinion.py) transferred
unmodified from the biotech/semiconductor dataset it was tuned on.

Why cross-validation instead of a single train/test split: only 24 of 141 Kickstarter
proposals actually succeeded (funded >= 100% of goal). A single 70/30 split risks a
test fold with too few positives to estimate precision/recall reliably. Stratified
5-fold CV uses every proposal exactly once as held-out test data, while the decision
threshold for that fold is always fit on a separate training fold -- so the reported
"after calibration" numbers are genuinely out-of-sample, not overfit to the same 141
points being evaluated.

Threshold shape mirrors the production verdict rule exactly: predict GO if
  score >= tau_r  OR  confidence >= tau_c
tau_r and tau_c are jointly grid-searched (candidate cutpoints = midpoints between
consecutive sorted values actually observed in the training fold) to maximize F1 on
the training fold only. F1 (not accuracy) is the objective because the class is
imbalanced (83% failure rate) -- "always predict failure" already scores 83% accuracy
without predicting a single success, so accuracy alone would just re-derive the
existing broken threshold.

Usage:
    python3 calibrate_kickstarter_threshold.py
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "tools"))

from kickstarter_dataset import load_funded_pct  # noqa: E402
from src.backend.utils.score_extraction import extract_report_scores  # noqa: E402

REPORTS_DIR = ROOT / "src" / "data" / "reports"
OUT_DIR = ROOT / "results" / "KickstarterCalibrated" / "evaluation"
OUT_PATH = OUT_DIR / "calibration_results.json"

PROD_TAU_R = 0.725
PROD_TAU_C = 0.811

SEED = 42
K_FOLDS = 5


def load_rows() -> List[Dict[str, Any]]:
    funded = load_funded_pct()
    rows: List[Dict[str, Any]] = []
    for pid, pct in funded.items():
        p = REPORTS_DIR / f"{pid}__openai_final_report.md"
        if not p.exists():
            p = REPORTS_DIR / f"{pid}_final_report.md"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        s = extract_report_scores(text)
        city = "Toronto" if pid.startswith("kt") else "Montreal"
        rows.append({
            "pid": pid, "city": city, "funded_pct": pct,
            "success": 1 if pct >= 100 else 0,
            "score": s["overall_ranking"], "confidence": s["confidence"],
        })
    return rows


def confusion(rows: List[Dict[str, Any]], tau_r: float, tau_c: float) -> Dict[str, int]:
    tp = fp = fn = tn = 0
    for r in rows:
        pred = 1 if (r["score"] >= tau_r or r["confidence"] >= tau_c) else 0
        actual = r["success"]
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_confusion(c: Dict[str, int]) -> Dict[str, Any]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    total = tp + fp + fn + tn
    accuracy = round((tp + tn) / total, 4) if total else None
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    elif precision is not None and recall is not None:
        f1 = 0.0
    else:
        f1 = None
    return {**c, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def f1_only(rows: List[Dict[str, Any]], tau_r: float, tau_c: float) -> float:
    m = metrics_from_confusion(confusion(rows, tau_r, tau_c))
    return m["f1"] if m["f1"] is not None else 0.0


def candidate_thresholds(rows: List[Dict[str, Any]], key: str) -> List[float]:
    """Midpoints between consecutive sorted unique values -- tests every decision
    boundary actually present in this fold's data, no fixed-grid blind spots."""
    vals = sorted({r[key] for r in rows})
    if len(vals) < 2:
        return vals if vals else [0.5]
    mids = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
    return [vals[0] - 0.01] + mids + [vals[-1] + 0.01]


def grid_search_best(train_rows: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Return (tau_r, tau_c, train_f1) maximizing F1 via OR-logic over train_rows only."""
    r_cands = candidate_thresholds(train_rows, "score")
    c_cands = candidate_thresholds(train_rows, "confidence")
    best_r, best_c, best_f1 = PROD_TAU_R, PROD_TAU_C, -1.0
    for tr in r_cands:
        for tc in c_cands:
            f1 = f1_only(train_rows, tr, tc)
            if f1 > best_f1:
                best_r, best_c, best_f1 = tr, tc, f1
    return best_r, best_c, best_f1


def stratified_kfold_indices(rows: List[Dict[str, Any]], k: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    pos = [i for i, r in enumerate(rows) if r["success"] == 1]
    neg = [i for i, r in enumerate(rows) if r["success"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds: List[List[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(pos):
        folds[i % k].append(idx)
    for i, idx in enumerate(neg):
        folds[i % k].append(idx)
    return folds


def roc_auc(rows: List[Dict[str, Any]]) -> Any:
    """Threshold-free discrimination: P(score_success > score_failure), via the
    Mann-Whitney U relation. Answers whether the score has real signal at all,
    independent of where any particular decision threshold is drawn."""
    pos = [r["score"] for r in rows if r["success"] == 1]
    neg = [r["score"] for r in rows if r["success"] == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return round(wins / (len(pos) * len(neg)), 4)


def average_precision(rows: List[Dict[str, Any]]) -> Any:
    ranked = sorted(rows, key=lambda r: -r["score"])
    n_pos = sum(r["success"] for r in rows)
    if n_pos == 0:
        return None
    tp = 0
    ap = 0.0
    for i, r in enumerate(ranked, start=1):
        if r["success"] == 1:
            tp += 1
            ap += tp / i
    return round(ap / n_pos, 4)


def main():
    rows = load_rows()
    n = len(rows)
    n_success = sum(r["success"] for r in rows)

    auc = roc_auc(rows)
    ap = average_precision(rows)

    before = metrics_from_confusion(confusion(rows, PROD_TAU_R, PROD_TAU_C))
    before["tau_r"] = PROD_TAU_R
    before["tau_c"] = PROD_TAU_C
    for r in rows:
        r["before_predicted_go"] = 1 if (r["score"] >= PROD_TAU_R or r["confidence"] >= PROD_TAU_C) else 0

    is_tr, is_tc, _is_train_f1 = grid_search_best(rows)
    in_sample = metrics_from_confusion(confusion(rows, is_tr, is_tc))
    in_sample["tau_r"] = round(is_tr, 4)
    in_sample["tau_c"] = round(is_tc, 4)
    for r in rows:
        r["in_sample_predicted_go"] = 1 if (r["score"] >= is_tr or r["confidence"] >= is_tc) else 0

    folds = stratified_kfold_indices(rows, K_FOLDS, SEED)
    per_fold = []
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for fi in range(K_FOLDS):
        test_idx = set(folds[fi])
        train_rows = [r for i, r in enumerate(rows) if i not in test_idx]
        test_rows = [r for i, r in enumerate(rows) if i in test_idx]
        tr, tc, train_f1 = grid_search_best(train_rows)
        c = confusion(test_rows, tr, tc)
        for key in agg:
            agg[key] += c[key]
        fold_metrics = metrics_from_confusion(c)
        per_fold.append({
            "fold": fi + 1, "n_test": len(test_rows),
            "n_test_success": sum(r["success"] for r in test_rows),
            "tau_r": round(tr, 4), "tau_c": round(tc, 4),
            "train_f1": round(train_f1, 4),
            **fold_metrics,
        })
        for i in sorted(test_idx):
            rows[i]["cv_fold"] = fi + 1
            rows[i]["cv_predicted_go"] = 1 if (rows[i]["score"] >= tr or rows[i]["confidence"] >= tc) else 0

    cv_aggregated = metrics_from_confusion(agg)

    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method_note": (
            "Threshold shape: predict GO if score >= tau_r OR confidence >= tau_c "
            "(same OR-logic shape as production verdict_rule()). 'before_calibration' "
            "uses the production thresholds transferred unmodified from the biotech "
            "dataset. 'in_sample_calibration' fits tau_r/tau_c by grid search on all "
            "141 proposals and evaluates on the same 141 -- reported only as an "
            "overfitting-risk reference, not a valid generalization estimate. "
            "'cross_validation.aggregated_held_out' is the honest out-of-sample result: "
            "each proposal is scored only by a threshold fit on the other 4/5 folds."
        ),
        "n_total": n,
        "n_success": n_success,
        "n_fail": n - n_success,
        "success_rate": round(n_success / n, 4),
        "discrimination": {"roc_auc": auc, "average_precision": ap},
        "before_calibration": before,
        "in_sample_calibration": in_sample,
        "cross_validation": {
            "k_folds": K_FOLDS,
            "seed": SEED,
            "per_fold": per_fold,
            "aggregated_held_out": cv_aggregated,
        },
        "per_proposal": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] wrote {OUT_PATH}")
    print(f"n={n} success={n_success} ({n_success/n:.1%})")
    print(f"ROC-AUC={auc}  Average Precision={ap}")
    print(f"BEFORE     tau_r={PROD_TAU_R} tau_c={PROD_TAU_C}  -> {before}")
    print(f"IN-SAMPLE  tau_r={in_sample['tau_r']} tau_c={in_sample['tau_c']}  -> {in_sample}")
    print(f"CV HELD-OUT (aggregated across {K_FOLDS} folds) -> {cv_aggregated}")
    for f in per_fold:
        print(f"  fold {f['fold']}: n_test={f['n_test']} tau_r={f['tau_r']} tau_c={f['tau_c']} "
              f"train_f1={f['train_f1']} test_f1={f['f1']} precision={f['precision']} recall={f['recall']}")


if __name__ == "__main__":
    main()
