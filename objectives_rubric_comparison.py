#!/usr/bin/env python3
"""
Objectives rubric comparison: Human vs AI scores.

Uses the existing human objective scores from HUMAN_RAW (calibration_sweep.py)
and compares them against AI sub-rubric scores from generate_objectives_rubric_scores.py.

Usage:
    python objectives_rubric_comparison.py
    python objectives_rubric_comparison.py --scores_dir src/data/objectives_rubric_scores
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCORES_DIR = PROJECT_ROOT / "src" / "data" / "objectives_rubric_scores"

# Human objective scores from ExpertsEvaluations.xlsx (via calibration_sweep.py HUMAN_RAW)
# Key = short pid (p1, p2 ...), value = human objective score (1-5)
# Note: human Excel column is "objective"; AI dimension name is "objectives"
HUMAN_OBJECTIVE = {
    "p1": 4, "p2": 5, "p3": 5, "p4": 5,
    "p5": 4, "p6": 4, "p7": 5, "p8": 5,
    "pA": 3, "pB": 4, "pC": 3, "pD": 5,
}

PID_SUFFIX = "__openai"

RUBRICS = [
    ("objectives_R1_problem_clarity",     "R1  Problem Clarity & Specificity ", "document-verifiable"),
    ("objectives_R2_market_size",         "R2  Market Size Quantification    ", "document-verifiable"),
    ("objectives_R3_buyer_pathway",       "R3  Buyer / Payer Pathway         ", "document-verifiable"),
    ("objectives_R4_unmet_need_evidence", "R4  Unmet Need Evidence           ", "document-verifiable"),
    ("objectives_R5_why_now",             "R5  Why-Now Timing Rationale      ", "judgment-dependent"),
    ("objectives_R6_competitive_context", "R6  Competitive Context           ", "judgment-dependent"),
    ("objectives_R7_problem_solution_fit","R7  Problem-Solution Fit          ", "judgment-dependent"),
    ("objectives_R8_addressability",      "R8  Market Addressability         ", "judgment-dependent"),
]


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None

    def rank(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_vals[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denom_x = sum((v - mean_rx) ** 2 for v in rx) ** 0.5
    denom_y = sum((v - mean_ry) ** 2 for v in ry) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None
    return round(num / (denom_x * denom_y), 3)


def load_ai_scores(scores_dir: Path) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for short_pid in HUMAN_OBJECTIVE:
        full_pid = short_pid + PID_SUFFIX
        path = scores_dir / full_pid / "objectives_rubric_scores.json"
        if not path.exists():
            print(f"[WARN] No AI scores for {short_pid}: {path}", flush=True)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rubrics_section = data.get("objectives_rubrics", {})
            scores: Dict[str, float] = {}
            for key, _, _ in RUBRICS:
                entry = rubrics_section.get(key)
                val = entry.get("score") if isinstance(entry, dict) else entry
                if val is not None:
                    scores[key] = float(val)
            if scores:
                result[short_pid] = scores
        except Exception as exc:
            print(f"[WARN] Failed to load AI scores for {short_pid}: {exc}", flush=True)
    return result


def compute_comparison(
    ai: Dict[str, Dict[str, float]],
) -> Tuple[List[Tuple], List[str]]:
    shared = [p for p in HUMAN_OBJECTIVE if p in ai]
    rows = []
    for key, label, rubric_type in RUBRICS:
        pairs = [
            (HUMAN_OBJECTIVE[p], ai[p][key])
            for p in shared if key in ai[p]
        ]
        if not pairs:
            rows.append((label, rubric_type, None, None, None, None, 0))
            continue
        h_vals = [h for h, _ in pairs]
        a_vals = [a for _, a in pairs]
        h_mean = round(sum(h_vals) / len(h_vals), 2)
        a_mean = round(sum(a_vals) / len(a_vals), 2)
        diff   = round(a_mean - h_mean, 2)
        sp     = _spearman(h_vals, a_vals)
        rows.append((label, rubric_type, h_mean, a_mean, diff, sp, len(pairs)))
    return rows, shared


def print_table(rows: List[Tuple], shared: List[str]) -> None:
    SEP = "-" * 92
    print()
    print(f"  Objectives Rubric Comparison: Human vs AI  (n={len(shared)} proposals)")
    print(f"  Human baseline: overall objective score (1-5) from ExpertsEvaluations.xlsx")
    print(SEP)
    print(f"  {'Rubric':<35} {'Type':<22} {'Human avg':>9} {'AI avg':>7} {'Diff':>6} {'Spearman':>9}")
    print(SEP)

    for label, rubric_type, h_mean, a_mean, diff, sp, n in rows:
        h_str  = f"{h_mean:.2f}" if h_mean is not None else "    n/a"
        a_str  = f"{a_mean:.2f}" if a_mean is not None else "  n/a"
        d_str  = (f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}") if diff is not None else "  n/a"
        sp_str = f"{sp:.3f}" if sp is not None else "  n/a"
        flag   = "  ← GAP" if (sp is not None and sp < 0.45) else ""
        print(f"  {label:<35} {rubric_type:<22} {h_str:>9} {a_str:>7} {d_str:>6} {sp_str:>9}{flag}")

    print(SEP)
    doc_sps  = [r[5] for r in rows if r[1] == "document-verifiable" and r[5] is not None]
    judg_sps = [r[5] for r in rows if r[1] == "judgment-dependent"  and r[5] is not None]
    if doc_sps:
        print(f"  Avg Spearman  document-verifiable  (R1-R4): {round(sum(doc_sps)/len(doc_sps), 3):>6}")
    if judg_sps:
        print(f"  Avg Spearman  judgment-dependent   (R5-R8): {round(sum(judg_sps)/len(judg_sps), 3):>6}")

    print()
    print("  Interpretation:")
    print("  • R1-R4 Spearman ≥ 0.60  →  AI reads documents as well as humans")
    print("  • R5-R8 Spearman ≤ 0.40  →  AI cannot replicate human judgment on tacit signals")
    print("  • Negative Diff           →  AI systematically under-scores vs human early-stage standards")
    print()
    print(f"  Proposals scored: {sorted(shared)}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare human vs AI objectives rubric scores.")
    ap.add_argument("--scores_dir", type=str, default=str(SCORES_DIR),
                    help="Directory containing <pid>/objectives_rubric_scores.json files")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    ai = load_ai_scores(scores_dir)

    if not ai:
        sys.exit(
            "[ERROR] No AI objectives rubric scores found.\n"
            "Run first:  python run_objectives_rubric_batch.py"
        )

    rows, shared = compute_comparison(ai)
    print_table(rows, shared)


if __name__ == "__main__":
    main()
