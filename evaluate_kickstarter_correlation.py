#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate AI proposal scores against real Kickstarter funding outcomes.

Ground truth here is `funded (%)` from Kickstarter/dataset_info.xlsx, not a
human-expert rubric, so this is a separate script from evaluate_cohens_kappa.py
(which expects the 5-dimension + verdict human Excel shape used by the
original VC/biotech dataset).

Metrics, per city (Toronto, Montreal) and combined:
  - Overall Spearman: AI overall_ranking (0-1) vs funded (%), with p-value.
    Rank-based rather than Pearson because funded (%) is heavily right-skewed
    (a handful of proposals exceed 1000%).
  - Per-dimension Spearman: each of team/objective/strategy/advantages/
    feasibility vs funded (%), individually.
  - Binary classification: success = funded% >= 100. This is not an arbitrary
    cutoff -- Kickstarter is all-or-nothing funding, so anything under 100%
    means the campaign collected zero dollars, not partial success. Compared
    against the AI's GO/HOLD verdict for accuracy/precision/recall/F1.

Output: results/Kickstarter/evaluation/evaluation_summary.json
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src" / "tools"))

from kickstarter_dataset import load_funded_pct  # noqa: E402
from src.backend.utils.score_extraction import extract_report_scores  # noqa: E402

REPORTS_DIR = BASE_DIR / "src" / "data" / "reports"
OUT_DIR = BASE_DIR / "results" / "Kickstarter" / "evaluation"

DIMENSIONS = ["team", "objective", "strategy", "advantages", "feasibility"]


def load_ai_scores() -> Dict[str, Dict[str, Any]]:
    """Read each Kickstarter proposal's final report (if it exists yet) and extract AI scores."""
    funded = load_funded_pct()
    scores: Dict[str, Dict[str, Any]] = {}
    for pid in funded:
        report_path = REPORTS_DIR / f"{pid}__openai_final_report.md"
        if not report_path.exists():
            report_path = REPORTS_DIR / f"{pid}_final_report.md"
        if not report_path.exists():
            continue
        text = report_path.read_text(encoding="utf-8")
        scores[pid] = extract_report_scores(text)
    return scores


def _spearman(pairs: List[Tuple[float, float]]) -> Dict[str, Any]:
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": None, "p_value": None}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    rho, pval = spearmanr(a, b)
    if rho != rho:  # NaN check (e.g. one side is constant)
        return {"n": len(pairs), "rho": None, "p_value": None}
    return {"n": len(pairs), "rho": round(float(rho), 4), "p_value": round(float(pval), 4)}


def _classification(pairs: List[Tuple[str, float]]) -> Dict[str, Any]:
    """pairs: (ai_verdict 'Y'/'N', funded_pct)."""
    tp = fp = fn = tn = 0
    for verdict, funded_pct in pairs:
        predicted_go = verdict == "Y"
        actually_funded = funded_pct >= 100.0
        if predicted_go and actually_funded:
            tp += 1
        elif predicted_go and not actually_funded:
            fp += 1
        elif not predicted_go and actually_funded:
            fn += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    accuracy = round((tp + tn) / total, 4) if total else None
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) else None
    f1: Optional[float]
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    elif precision is not None and recall is not None:
        f1 = 0.0
    else:
        f1 = None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
    }


def holm_correction(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction for a family of p-values, matching the
    correction method the reference VC-dataset audit uses across its six Spearman tests."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    holm_p = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, adj)
        holm_p[idx] = running_max
    return holm_p


def compute_group_metrics(
    pids: List[str],
    ai_scores: Dict[str, Dict[str, Any]],
    funded: Dict[str, float],
) -> Dict[str, Any]:
    overall_pairs: List[Tuple[float, float]] = []
    dim_pairs: Dict[str, List[Tuple[float, float]]] = {d: [] for d in DIMENSIONS}
    verdict_pairs: List[Tuple[str, float]] = []

    for pid in pids:
        s = ai_scores.get(pid)
        if s is None:
            continue
        pct = funded[pid]
        overall_pairs.append((s["overall_ranking"], pct))
        for d in DIMENSIONS:
            dim_pairs[d].append((s[d], pct))
        verdict_pairs.append((s["verdict"], pct))

    labels = ["overall_ranking"] + DIMENSIONS
    spearman_results = [_spearman(overall_pairs)] + [_spearman(dim_pairs[d]) for d in DIMENSIONS]
    raw_pvals = [r["p_value"] if r["p_value"] is not None else 1.0 for r in spearman_results]
    holm_pvals = holm_correction(raw_pvals)
    for result, hp in zip(spearman_results, holm_pvals):
        result["holm_p_value"] = round(hp, 5) if result["p_value"] is not None else None

    return {
        "n_proposals": len(pids),
        "n_scored": len(overall_pairs),
        "overall_spearman": spearman_results[0],
        "dimension_spearman": dict(zip(DIMENSIONS, spearman_results[1:])),
        "classification": _classification(verdict_pairs),
    }


def main() -> None:
    funded = load_funded_pct()
    ai_scores = load_ai_scores()

    all_pids = sorted(funded.keys())
    toronto_pids = [p for p in all_pids if p.startswith("kt")]
    montreal_pids = [p for p in all_pids if p.startswith("km")]
    scored_pids = [p for p in all_pids if p in ai_scores]

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ground_truth": "Kickstarter dataset_info.xlsx funded (%)",
        "success_threshold": "funded % >= 100 (Kickstarter is all-or-nothing funding)",
        "n_total_dataset": len(all_pids),
        "n_evaluated": len(scored_pids),
        "toronto": compute_group_metrics([p for p in toronto_pids if p in scored_pids], ai_scores, funded),
        "montreal": compute_group_metrics([p for p in montreal_pids if p in scored_pids], ai_scores, funded),
        "combined": compute_group_metrics(scored_pids, ai_scores, funded),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "evaluation_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] evaluation summary written: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
