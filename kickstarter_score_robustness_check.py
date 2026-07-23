#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two rigor checks on the Method B (simple composite) result from
improve_kickstarter_score.py:

1. Bootstrap confidence intervals on the ROC-AUC differences between methods
   (B vs A, B vs C, C vs A), to check whether the observed improvements are
   distinguishable from sampling noise given only 24 positive examples.

2. Per-fold dimension-selection stability: re-derive, using ONLY each training
   fold's data, which 3 dimensions would be selected by the same "top-3 by
   Spearman correlation with funded_pct" rule that (on the full dataset)
   produced Method B's {feasibility, strategy, advantages}. Checks whether
   that selection is stable across folds, or an artifact of running the
   selection on the full 141-proposal dataset. Also evaluates a fully-nested
   version of Method B, where each fold's composite is built only from that
   fold's own top-3 selection and its own threshold, via the same honest CV
   procedure used throughout this project.

Usage:
    python3 kickstarter_score_robustness_check.py
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "tools"))

from kickstarter_dataset import load_funded_pct  # noqa: E402
from src.backend.utils.score_extraction import extract_report_scores  # noqa: E402

REPORTS_DIR = ROOT / "src" / "data" / "reports"
OUT_DIR = ROOT / "results" / "KickstarterCalibrated" / "evaluation"
OUT_PATH = OUT_DIR / "score_robustness_check.json"
IMPROVE_PATH = OUT_DIR / "score_improvement_results.json"

SEED = 42
K_FOLDS = 5
DIMS = ["team", "objective", "strategy", "advantages", "feasibility"]
N_BOOT = 10000
BOOT_SEED = 123


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
            **{d: s[d] for d in DIMS},
        })
    return rows


def stratified_kfold_indices(rows, k, seed):
    rng = random.Random(seed)
    pos = [i for i, r in enumerate(rows) if r["success"] == 1]
    neg = [i for i, r in enumerate(rows) if r["success"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(pos):
        folds[i % k].append(idx)
    for i, idx in enumerate(neg):
        folds[i % k].append(idx)
    return folds


def simple_composite(row, dims):
    return sum(row[d] for d in dims) / len(dims)


def metrics(tp, fp, fn, tn):
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
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ── Fast vectorized AUC (rank-based Mann-Whitney, ties = average rank) ────────

def fast_auc(y_true: np.ndarray, y_score: np.ndarray):
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_auc_diff(y_true, score_1, score_2, n_boot, seed, label_1, label_2):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        a1 = fast_auc(yt, score_1[idx])
        a2 = fast_auc(yt, score_2[idx])
        if a1 is None or a2 is None:
            continue
        diffs.append(a2 - a1)
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "comparison": f"{label_2} minus {label_1}",
        "n_boot_valid": int(len(diffs)),
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_95_lower": round(float(ci_lo), 4),
        "ci_95_upper": round(float(ci_hi), 4),
        "prob_method_2_better": round(float((diffs > 0).mean()), 4),
        "significant_at_0.05": bool(ci_lo > 0 or ci_hi < 0),
    }


def best_tau(y_true_local: np.ndarray, scores: np.ndarray):
    vals = sorted(set(scores.tolist()))
    if len(vals) < 2:
        return 0.5, -1.0
    mids = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
    cands = [vals[0] - 0.01] + mids + [vals[-1] + 0.01]
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true_local == 1)).sum())
        fp = int(((pred == 1) & (y_true_local == 0)).sum())
        fn = int(((pred == 0) & (y_true_local == 1)).sum())
        tn = int(((pred == 0) & (y_true_local == 0)).sum())
        f1 = metrics(tp, fp, fn, tn)["f1"] or 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def main():
    rows = load_rows()
    n = len(rows)
    y_true = np.array([r["success"] for r in rows])
    funded_pct = np.array([r["funded_pct"] for r in rows])

    overall_score = np.array([r["overall_score"] for r in rows])
    composite_b = np.array([simple_composite(r, ["feasibility", "strategy", "advantages"]) for r in rows])

    improve_data = json.loads(IMPROVE_PATH.read_text(encoding="utf-8"))
    pid_to_oof = {p["pid"]: p["logreg_oof_proba"] for p in improve_data["per_proposal"]}
    logreg_c = np.array([pid_to_oof[r["pid"]] for r in rows])

    # sanity check against previously reported (non-bootstrapped) AUCs
    print("Sanity check (should match improve_kickstarter_score.py output):")
    print(f"  A (overall_score) AUC = {fast_auc(y_true, overall_score):.4f}  (expected 0.7644)")
    print(f"  B (simple composite) AUC = {fast_auc(y_true, composite_b):.4f}  (expected 0.8048)")
    print(f"  C (logreg oof) AUC = {fast_auc(y_true, logreg_c):.4f}  (expected 0.7682)\n")

    # 1. Bootstrap CIs
    print(f"Running {N_BOOT} bootstrap iterations per comparison...")
    boot_results = {
        "B_vs_A": bootstrap_auc_diff(y_true, overall_score, composite_b, N_BOOT, BOOT_SEED, "A", "B"),
        "B_vs_C": bootstrap_auc_diff(y_true, logreg_c, composite_b, N_BOOT, BOOT_SEED + 1, "C", "B"),
        "C_vs_A": bootstrap_auc_diff(y_true, overall_score, logreg_c, N_BOOT, BOOT_SEED + 2, "A", "C"),
    }
    for k, v in boot_results.items():
        print(f"  {k}: mean_diff={v['mean_diff']:+.4f}  95% CI=[{v['ci_95_lower']:+.4f}, {v['ci_95_upper']:+.4f}]  "
              f"P(better)={v['prob_method_2_better']:.3f}  significant={v['significant_at_0.05']}")

    # 2. Per-fold dimension-selection stability
    folds = stratified_kfold_indices(rows, K_FOLDS, SEED)
    fixed_selection = {"feasibility", "strategy", "advantages"}
    per_fold_selection = []
    nested_scores = np.zeros(n)

    for fi in range(K_FOLDS):
        test_idx = set(folds[fi])
        train_idx = [i for i in range(n) if i not in test_idx]
        train_funded = funded_pct[train_idx]

        rhos = {}
        for d in DIMS:
            dim_vals = np.array([rows[i][d] for i in train_idx])
            rho, pval = spearmanr(dim_vals, train_funded)
            rhos[d] = {"rho": round(float(rho), 4), "p_value": round(float(pval), 4)}

        ranked = sorted(rhos.items(), key=lambda kv: -kv[1]["rho"])
        top3 = {d for d, _ in ranked[:3]}
        matches_fixed = top3 == fixed_selection

        per_fold_selection.append({
            "fold": fi + 1,
            "rhos_on_train": rhos,
            "top3_selected": sorted(top3),
            "matches_fixed_selection": matches_fixed,
        })

        for i in test_idx:
            nested_scores[i] = simple_composite(rows[i], list(top3))

    n_folds_matching = sum(1 for f in per_fold_selection if f["matches_fixed_selection"])

    # 3. Fully-nested Method B: dimension selection AND threshold both fit per-fold
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for fi in range(K_FOLDS):
        test_idx = list(folds[fi])
        train_idx = [i for i in range(n) if i not in set(test_idx)]
        top3 = per_fold_selection[fi]["top3_selected"]
        train_scores_local = np.array([simple_composite(rows[i], top3) for i in train_idx])
        tau, train_f1 = best_tau(y_true[train_idx], train_scores_local)
        test_scores_local = nested_scores[test_idx]
        pred = (test_scores_local >= tau).astype(int)
        yt = y_true[test_idx]
        agg["tp"] += int(((pred == 1) & (yt == 1)).sum())
        agg["fp"] += int(((pred == 1) & (yt == 0)).sum())
        agg["fn"] += int(((pred == 0) & (yt == 1)).sum())
        agg["tn"] += int(((pred == 0) & (yt == 0)).sum())

    nested_cv_metrics = metrics(**agg)
    nested_auc = fast_auc(y_true, nested_scores)

    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bootstrap": {
            "n_boot": N_BOOT,
            "method_note": (
                "Paired bootstrap over the 141 proposals (resample rows, recompute AUC "
                "for both methods on the same resample, repeat). Each method's "
                "per-proposal score is treated as fixed: either a closed-form composite "
                "formula (A, B), or Method C's pooled out-of-fold CV predictions. "
                "95% CI excluding 0 = statistically significant difference at alpha=0.05."
            ),
            **boot_results,
        },
        "dimension_stability": {
            "fixed_selection_used_in_report": sorted(fixed_selection),
            "n_folds_matching_fixed_selection": n_folds_matching,
            "k_folds": K_FOLDS,
            "per_fold": per_fold_selection,
        },
        "nested_method_b": {
            "description": (
                "Method B re-evaluated with dimension selection AND threshold both fit "
                "within each training fold only (fully nested, most rigorous version -- "
                "no part of this number was computed using the full 141-proposal dataset)."
            ),
            "roc_auc": round(float(nested_auc), 4) if nested_auc is not None else None,
            "cv_aggregated_held_out": nested_cv_metrics,
            "per_proposal_scores": [
                {"pid": rows[i]["pid"], "nested_score": round(float(nested_scores[i]), 4)}
                for i in range(n)
            ],
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_PATH}")

    print("\nDimension stability across folds:")
    for f in per_fold_selection:
        print(f"  fold {f['fold']}: top3={f['top3_selected']}  matches_fixed={f['matches_fixed_selection']}")
    print(f"  {n_folds_matching}/{K_FOLDS} folds matched the fixed selection {sorted(fixed_selection)}")

    print(f"\nNested Method B (fully out-of-sample): ROC-AUC={result['nested_method_b']['roc_auc']}")
    print(f"  CV aggregated: {nested_cv_metrics}")


if __name__ == "__main__":
    main()
