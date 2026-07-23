#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test whether a better composite score improves discrimination between funded and
unfunded Kickstarter campaigns, compared to the original production overall_score.

All three scores are evaluated on equal footing: a single score threshold (no OR-logic
confidence gate -- that's dropped here to isolate score quality alone), calibrated via
the same honest stratified 5-fold cross-validation used in
calibrate_kickstarter_threshold.py (same fold assignment, same seed=42, so results are
directly comparable across scripts).

Method A: overall_score      -- production weighted average across all 5 dimensions
                                 (team/objective/strategy/advantages/feasibility),
                                 unchanged from earlier reports. Baseline.
Method B: simple_composite    -- fixed, unweighted average of feasibility+strategy+
                                 advantages only (the 3 dimensions shown in the earlier
                                 correlation study to significantly predict real funding
                                 outcomes; team/objective dropped). No fitting involved,
                                 so zero leakage risk.
Method C: logistic_regression -- L2-regularized logistic regression on all 5 dimension
                                 scores, fit *within* each training fold only (features
                                 standardized on train-fold stats), producing pooled
                                 out-of-fold predicted probabilities used for both
                                 ROC-AUC and threshold calibration -- so score
                                 construction and threshold selection are both honestly
                                 out-of-sample, not just the threshold.

Usage:
    python3 improve_kickstarter_score.py
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "tools"))

from kickstarter_dataset import load_funded_pct  # noqa: E402
from src.backend.utils.score_extraction import extract_report_scores  # noqa: E402

REPORTS_DIR = ROOT / "src" / "data" / "reports"
OUT_DIR = ROOT / "results" / "KickstarterCalibrated" / "evaluation"
OUT_PATH = OUT_DIR / "score_improvement_results.json"

SEED = 42
K_FOLDS = 5
DIMS = ["team", "objective", "strategy", "advantages", "feasibility"]
COMPOSITE_DIMS = ["feasibility", "strategy", "advantages"]
L2_C = 1.0  # regularization strength (sklearn-style "C": larger = weaker penalty)


# ── Data loading ──────────────────────────────────────────────────────────────

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
            "overall_score": s["overall_ranking"],
            "confidence": s["confidence"],
            **{d: s[d] for d in DIMS},
        })
    return rows


def simple_composite(row: Dict[str, Any]) -> float:
    return sum(row[d] for d in COMPOSITE_DIMS) / len(COMPOSITE_DIMS)


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


# ── Metrics ──────────────────────────────────────────────────────────────────

def confusion_single(rows: List[Dict[str, Any]], scores: List[float], tau: float) -> Dict[str, int]:
    tp = fp = fn = tn = 0
    for r, s in zip(rows, scores):
        pred = 1 if s >= tau else 0
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


def f1_for_threshold(rows, scores, tau) -> float:
    m = metrics_from_confusion(confusion_single(rows, scores, tau))
    return m["f1"] if m["f1"] is not None else 0.0


def best_threshold(rows, scores) -> Tuple[float, float]:
    vals = sorted(set(scores))
    if len(vals) < 2:
        cands = vals if vals else [0.5]
    else:
        mids = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        cands = [vals[0] - 0.01] + mids + [vals[-1] + 0.01]
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        f1 = f1_for_threshold(rows, scores, t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def roc_auc(rows, scores) -> Any:
    pos = [s for r, s in zip(rows, scores) if r["success"] == 1]
    neg = [s for r, s in zip(rows, scores) if r["success"] == 0]
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


def average_precision(rows, scores) -> Any:
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    n_pos = sum(r["success"] for r in rows)
    if n_pos == 0:
        return None
    tp = 0
    ap = 0.0
    for rank, i in enumerate(order, start=1):
        if rows[i]["success"] == 1:
            tp += 1
            ap += tp / rank
    return round(ap / n_pos, 4)


# ── L2-regularized logistic regression (via scipy, no sklearn dependency) ─────

def fit_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    n, d = X.shape

    def nll(w):
        z = X @ w[1:] + w[0]
        loss = np.mean(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - z * y)
        reg = (1.0 / (2 * C)) * np.sum(w[1:] ** 2)
        return loss + reg

    w0 = np.zeros(d + 1)
    res = minimize(nll, w0, method="L-BFGS-B")
    return res.x  # [bias, coef_1..coef_d]


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    z = X @ w[1:] + w[0]
    return 1.0 / (1.0 + np.exp(-z))


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


# ── CV evaluation for a fixed-formula score (Methods A, B) ────────────────────

def evaluate_fixed_score(rows, scores, folds) -> Dict[str, Any]:
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_fold = []
    for fi in range(K_FOLDS):
        test_idx = set(folds[fi])
        train_rows = [r for i, r in enumerate(rows) if i not in test_idx]
        train_scores = [s for i, s in enumerate(scores) if i not in test_idx]
        test_rows = [r for i, r in enumerate(rows) if i in test_idx]
        test_scores = [s for i, s in enumerate(scores) if i in test_idx]
        tau, train_f1 = best_threshold(train_rows, train_scores)
        c = confusion_single(test_rows, test_scores, tau)
        for k in agg:
            agg[k] += c[k]
        m = metrics_from_confusion(c)
        per_fold.append({"fold": fi + 1, "tau": round(tau, 4), "train_f1": round(train_f1, 4), **m})
    return {
        "roc_auc": roc_auc(rows, scores),
        "average_precision": average_precision(rows, scores),
        "cv_aggregated_held_out": metrics_from_confusion(agg),
        "per_fold": per_fold,
    }


# ── CV evaluation for the learned logistic-regression score (Method C) ────────

def evaluate_logreg(rows, folds) -> Dict[str, Any]:
    X_all = np.array([[r[d] for d in DIMS] for r in rows])
    y_all = np.array([r["success"] for r in rows], dtype=float)

    oof_proba = np.zeros(len(rows))
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_fold = []

    for fi in range(K_FOLDS):
        test_idx_set = set(folds[fi])
        train_idx = [i for i in range(len(rows)) if i not in test_idx_set]
        test_idx = [i for i in range(len(rows)) if i in test_idx_set]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test = X_all[test_idx]

        mu, sd = standardize_fit(X_train)
        X_train_std = (X_train - mu) / sd
        X_test_std = (X_test - mu) / sd

        w = fit_logreg(X_train_std, y_train, C=L2_C)

        train_proba = predict_proba(X_train_std, w)
        test_proba = predict_proba(X_test_std, w)
        for i, p in zip(test_idx, test_proba):
            oof_proba[i] = p

        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        tau, train_f1 = best_threshold(train_rows, list(train_proba))
        c = confusion_single(test_rows, list(test_proba), tau)
        for k in agg:
            agg[k] += c[k]
        m = metrics_from_confusion(c)
        per_fold.append({
            "fold": fi + 1, "tau": round(float(tau), 4), "train_f1": round(train_f1, 4),
            "coef": {d: round(float(w[i + 1]), 4) for i, d in enumerate(DIMS)},
            "intercept": round(float(w[0]), 4),
            **m,
        })

    # Fit on the full dataset purely for interpretability (NOT used in any evaluation above).
    mu_full, sd_full = standardize_fit(X_all)
    X_full_std = (X_all - mu_full) / sd_full
    w_full = fit_logreg(X_full_std, y_all, C=L2_C)
    full_fit_coef = {d: round(float(w_full[i + 1]), 4) for i, d in enumerate(DIMS)}

    return {
        "roc_auc": roc_auc(rows, list(oof_proba)),
        "average_precision": average_precision(rows, list(oof_proba)),
        "cv_aggregated_held_out": metrics_from_confusion(agg),
        "per_fold": per_fold,
        "full_dataset_fit_coefficients_for_interpretation_only": {
            "intercept": round(float(w_full[0]), 4),
            "coef": full_fit_coef,
        },
        "oof_proba": [round(float(p), 4) for p in oof_proba],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = load_rows()
    n = len(rows)
    n_success = sum(r["success"] for r in rows)
    folds = stratified_kfold_indices(rows, K_FOLDS, SEED)

    overall_scores = [r["overall_score"] for r in rows]
    composite_scores = [simple_composite(r) for r in rows]

    result_a = evaluate_fixed_score(rows, overall_scores, folds)
    result_b = evaluate_fixed_score(rows, composite_scores, folds)
    result_c = evaluate_logreg(rows, folds)

    per_proposal = [
        {
            "pid": r["pid"], "city": r["city"], "funded_pct": r["funded_pct"], "success": r["success"],
            "overall_score": r["overall_score"], "simple_composite": round(composite_scores[i], 4),
            "logreg_oof_proba": result_c["oof_proba"][i],
        }
        for i, r in enumerate(rows)
    ]

    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_total": n,
        "n_success": n_success,
        "success_rate": round(n_success / n, 4),
        "k_folds": K_FOLDS,
        "seed": SEED,
        "l2_C": L2_C,
        "method_note": (
            "All three methods use a SINGLE score threshold (no OR-logic confidence "
            "gate) to isolate score quality alone, calibrated via the same stratified "
            "5-fold CV (same folds/seed as calibrate_kickstarter_threshold.py). Method A "
            "and B thresholds are fit-free (fixed formulas); Method C's logistic "
            "regression weights AND threshold are both fit on training folds only, "
            "producing pooled out-of-fold probabilities for ROC-AUC/AP."
        ),
        "method_a_overall_score": result_a,
        "method_b_simple_composite": result_b,
        "method_c_logistic_regression": result_c,
        "per_proposal": per_proposal,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] wrote {OUT_PATH}")
    print(f"n={n} success={n_success} ({n_success/n:.1%})\n")

    for label, res in (
        ("A: overall_score (production)", result_a),
        ("B: simple composite (feas+strat+adv)/3", result_b),
        ("C: logistic regression (5 dims, CV-fit)", result_c),
    ):
        cv = res["cv_aggregated_held_out"]
        print(f"--- {label} ---")
        print(f"  ROC-AUC={res['roc_auc']}  AP={res['average_precision']}")
        print(f"  CV held-out: tp={cv['tp']} fp={cv['fp']} fn={cv['fn']} tn={cv['tn']}  "
              f"acc={cv['accuracy']} precision={cv['precision']} recall={cv['recall']} f1={cv['f1']}")
        print()

    print("Full-dataset logistic regression coefficients (interpretation only):")
    print(json.dumps(result_c["full_dataset_fit_coefficients_for_interpretation_only"], indent=2))


if __name__ == "__main__":
    main()
