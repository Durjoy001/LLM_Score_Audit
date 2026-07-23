#!/usr/bin/env python3
"""
Advantages rubric comparison: Human vs AI scores.

Uses the existing human advantages scores from HUMAN_RAW (calibration_sweep.py)
and compares them against AI sub-rubric scores from generate_advantages_rubric_scores.py.

Note: Human experts call this dimension "advantages"; the AI pipeline calls it "innovation".
This script uses "advantages" throughout to match human-facing terminology.

Usage:
    python advantages_rubric_comparison.py
    python advantages_rubric_comparison.py --scores_dir src/data/advantages_rubric_scores
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCORES_DIR = PROJECT_ROOT / "src" / "data" / "advantages_rubric_scores"

# Human advantages scores from ExpertsEvaluations.xlsx (via calibration_sweep.py HUMAN_RAW)
# Key = short pid (p1, p2 ...), value = human advantages score (1-5)
# Note: human Excel column is "advantages"; AI dimension name is "innovation"
HUMAN_ADVANTAGES = {
    "p1": 3, "p2": 3, "p3": 5, "p4": 3,
    "p5": 4, "p6": 3, "p7": 5, "p8": 4,
    "pA": 3, "pB": 3, "pC": 4, "pD": 4,
}

PID_SUFFIX = "__openai"

RUBRICS = [
    ("advantages_R1_mechanism_novelty",    "R1  Mechanism Novelty             ", "document-verifiable"),
    ("advantages_R2_ip_status",            "R2  IP Status                     ", "document-verifiable"),
    ("advantages_R3_performance_proof",    "R3  Performance Proof             ", "document-verifiable"),
    ("advantages_R4_competitor_benchmark", "R4  Competitor Benchmarking       ", "document-verifiable"),
    ("advantages_R5_defensibility",        "R5  Defensibility of Position     ", "judgment-dependent"),
    ("advantages_R6_platform_potential",   "R6  Platform Potential            ", "judgment-dependent"),
    ("advantages_R7_validation_signals",   "R7  External Validation Signals   ", "judgment-dependent"),
    ("advantages_R8_adoption_readiness",   "R8  Market Adoption Readiness     ", "judgment-dependent"),
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
    for short_pid in HUMAN_ADVANTAGES:
        full_pid = short_pid + PID_SUFFIX
        path = scores_dir / full_pid / "advantages_rubric_scores.json"
        if not path.exists():
            print(f"[WARN] No AI scores for {short_pid}: {path}", flush=True)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rubrics_section = data.get("advantages_rubrics", {})
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
    shared = [p for p in HUMAN_ADVANTAGES if p in ai]
    rows = []
    for key, label, rubric_type in RUBRICS:
        pairs = [
            (HUMAN_ADVANTAGES[p], ai[p][key])
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
    print(f"  Advantages Rubric Comparison: Human vs AI  (n={len(shared)} proposals)")
    print(f"  Human baseline: overall advantages score (1-5) from ExpertsEvaluations.xlsx")
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
    print("  • Positive Diff on R5/R6  →  AI over-rates claims it cannot independently verify")
    print()
    print(f"  Proposals scored: {sorted(shared)}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare human vs AI advantages rubric scores.")
    ap.add_argument("--scores_dir", type=str, default=str(SCORES_DIR),
                    help="Directory containing <pid>/advantages_rubric_scores.json files")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    ai = load_ai_scores(scores_dir)

    if not ai:
        sys.exit(
            "[ERROR] No AI advantages rubric scores found.\n"
            "Run first:  python run_advantages_rubric_batch.py"
        )

    rows, shared = compute_comparison(ai)
    print_table(rows, shared)


if __name__ == "__main__":
    main()
