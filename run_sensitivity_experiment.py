#!/usr/bin/env python3
"""
Content Sensitivity Experiment — All Proposals.

For each proposal:
  PLUS  variant: inject explicit revenue model + regulatory pathway text
  MINUS variant: strip named partner mentions and phased-funding/timing rationale

Scores all variants via the strategy rubric tool, stores a summary JSON,
and prints a comparison table.

Usage:
    python3 run_sensitivity_experiment.py
    python3 run_sensitivity_experiment.py --pids p1 p8 pA   # subset
    python3 run_sensitivity_experiment.py --skip_score       # use cached scores, just print
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "src" / "data" / "reports"
SCORES_DIR = ROOT / "src" / "data" / "strategy_rubric_scores"
RESULTS_DIR = ROOT / "src" / "data" / "sensitivity_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_PIDS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]
PID_SUFFIX = "__openai"

TARGET_RUBRICS = [
    "strategy_R3_revenue",
    "strategy_R4_regulatory",
    "strategy_R6_partners",
    "strategy_R8_timing",
]
LABELS = {
    "strategy_R3_revenue":    "R3 Revenue Model Clarity",
    "strategy_R4_regulatory": "R4 Regulatory Path Awareness",
    "strategy_R6_partners":   "R6 Partner Engagement",
    "strategy_R8_timing":     "R8 Market Timing",
}

# ── Fixed injection text (same for all proposals) ─────────────────────────────
PLUS_INJECTION = """
**Revenue Model:** The company will commercialize via a B2B SaaS licensing model at $240,000 per hospital system per year (enterprise tier), $48,000 per year (mid-market tier). Unit economics: estimated cost to serve = $18,000/year per customer; gross margin = 62.5%. Payment structure: annual upfront license with quarterly usage-based overage at $0.40 per processed patient record. Year-1 revenue target: $2.4M from 10 enterprise contracts already in late-stage negotiation.

**Regulatory Strategy:** The product is classified as a Software as a Medical Device (SaMD) under FDA 21 CFR Part 820. The team filed a Pre-Submission (Q-Sub) with FDA in March 2026 (reference Q260312-01) and received written feedback confirming the 510(k) pathway via predicate device K213456. De Novo classification has been ruled out. CE mark submission under EU MDR Article 51 is scheduled for Q3 2026, with a notified body (TÜV SÜD) already engaged under a formal contract.
"""

# ── Patterns to strip for MINUS variant ───────────────────────────────────────
PARTNER_NAMES = ["Merck", "Fujifilm", "Pfizer", "Roche", "AstraZeneca", "J&J", "Novartis",
                 "GSK", "Sanofi", "Bayer", "Abbvie", "Amgen", "Gilead"]
TIMING_PHRASES = ["phased funding strategy", "Phased funding strategy",
                  "timing rationale", "market window", "first-mover"]


def _inject_plus(text: str) -> str:
    """Append PLUS_INJECTION after the strategy section Recommendations block."""
    # Try to find end of strategy section recommendations
    marker = "##### Technology and product innovation"
    if marker in text:
        return text.replace(marker, PLUS_INJECTION.rstrip() + "\n\n" + marker, 1)
    # Fallback: append before the next ##### section after "strategy"
    strategy_idx = text.find("##### Implementation path and strategy")
    if strategy_idx == -1:
        # Append before the Q&A section or at end
        qa_marker = "## 2. Dimension Q&A"
        if qa_marker in text:
            return text.replace(qa_marker, PLUS_INJECTION.rstrip() + "\n\n" + qa_marker, 1)
        return text + PLUS_INJECTION
    return text


def _strip_minus(text: str) -> str:
    """Remove partner mentions and phased-funding/timing rationale lines."""
    lines = text.split("\n")
    filtered = []
    for line in lines:
        # Drop bullet/numbered list lines containing partner names
        stripped = line.strip()
        has_partner = any(name in line for name in PARTNER_NAMES)
        is_list_item = bool(re.match(r"^-\s|^\d+\.\s", stripped))

        if has_partner and is_list_item:
            continue  # drop whole bullet

        if has_partner:
            # Inline mention — neutralise with generic term
            for name in PARTNER_NAMES:
                line = line.replace(name, "strategic partner")
            # Clean up doubled "strategic partner and strategic partner"
            line = re.sub(r"strategic partner(?:\s+and\s+strategic partner)+", "strategic partners", line)

        # Drop phased-funding and timing-rationale bullets
        has_timing = any(ph in line for ph in TIMING_PHRASES)
        if has_timing and is_list_item:
            continue

        if has_timing:
            # Neutralise inline
            line = re.sub(r"[Pp]hased funding strategy[^.]*\.", "", line)

        filtered.append(line)
    return "\n".join(filtered)


def create_variant_reports(short_pid: str) -> tuple[Path, Path]:
    full_pid = short_pid + PID_SUFFIX
    src = REPORT_DIR / f"{full_pid}_final_report.md"
    plus_path  = REPORT_DIR / f"{short_pid}_plus{PID_SUFFIX}_final_report.md"
    minus_path = REPORT_DIR / f"{short_pid}_minus{PID_SUFFIX}_final_report.md"

    text = src.read_text(encoding="utf-8")
    plus_path.write_text(_inject_plus(text), encoding="utf-8")
    minus_path.write_text(_strip_minus(text), encoding="utf-8")
    return plus_path, minus_path


def score_variant(pid: str) -> bool:
    """Run generate_strategy_rubric_scores.py for a given full PID. Returns True on success."""
    cmd = [
        sys.executable,
        str(ROOT / "src" / "tools" / "generate_strategy_rubric_scores.py"),
        "--pid", pid,
        "--use_llm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] Scoring failed for {pid}: {result.stderr.strip()}")
        return False
    return True


def load_scores(full_pid: str) -> dict | None:
    path = SCORES_DIR / full_pid / "strategy_rubric_scores.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["score"] for k, v in data.get("strategy_rubrics", {}).items()}


def arrow(new: int, old: int) -> str:
    if new > old:
        return f"+{new-old}↑"
    if new < old:
        return f"{new-old}↓"
    return "  →"


def main():
    ap = argparse.ArgumentParser(description="Content sensitivity experiment for all proposals.")
    ap.add_argument("--pids", nargs="+", default=ALL_PIDS, help="Short PIDs to process.")
    ap.add_argument("--skip_score", action="store_true", help="Skip LLM scoring; use cached scores.")
    args = ap.parse_args()

    pids = args.pids
    results = {}

    # ── Step 1: Create variants and score ────────────────────────────────────
    for short_pid in pids:
        full_pid   = short_pid + PID_SUFFIX
        plus_pid   = f"{short_pid}_plus{PID_SUFFIX}"
        minus_pid  = f"{short_pid}_minus{PID_SUFFIX}"

        print(f"\n[{short_pid}] Creating PLUS / MINUS variant reports …")
        create_variant_reports(short_pid)

        if not args.skip_score:
            print(f"[{short_pid}] Scoring PLUS …")
            score_variant(plus_pid)
            print(f"[{short_pid}] Scoring MINUS …")
            score_variant(minus_pid)

        baseline = load_scores(full_pid)
        plus_s   = load_scores(plus_pid)
        minus_s  = load_scores(minus_pid)

        if not baseline:
            print(f"  [WARN] No baseline scores for {short_pid}, skipping.")
            continue

        results[short_pid] = {
            "baseline": baseline,
            "plus":     plus_s,
            "minus":    minus_s,
        }

    # ── Step 2: Store summary ────────────────────────────────────────────────
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pids": pids,
        "target_rubrics": TARGET_RUBRICS,
        "results": results,
    }
    out_path = RESULTS_DIR / "sensitivity_experiment.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Results stored → {out_path}")

    # ── Step 3: Print table ───────────────────────────────────────────────────
    SEP  = "─" * 102
    SEP2 = "═" * 102

    print()
    print(SEP2)
    print("  CONTENT SENSITIVITY EXPERIMENT — All Proposals  |  Strategy Dimension")
    print("  PLUS  = inject explicit revenue model + regulatory filing")
    print("  MINUS = strip named partner mentions + phased-funding language")
    print(SEP2)

    for rubric in TARGET_RUBRICS:
        label = LABELS[rubric]
        print()
        print(f"  ▌ {label}")
        print(f"  {'PID':<6} {'Baseline':>9} {'PLUS':>10} {'MINUS':>10}   Sensitivity")
        print(f"  {SEP[:98]}")

        improvements = 0
        declines     = 0
        neutral      = 0

        for short_pid in pids:
            if short_pid not in results:
                continue
            b = results[short_pid]["baseline"]
            p = results[short_pid]["plus"]
            m = results[short_pid]["minus"]

            b_score = b.get(rubric, "n/a") if b else "n/a"
            p_score = p.get(rubric, "n/a") if p else "n/a"
            m_score = m.get(rubric, "n/a") if m else "n/a"

            if isinstance(b_score, int) and isinstance(p_score, int) and isinstance(m_score, int):
                p_arrow = arrow(p_score, b_score)
                m_arrow = arrow(m_score, b_score)
                sensitivity = []
                if p_score > b_score:
                    sensitivity.append("PLUS ↑")
                    improvements += 1
                if m_score < b_score:
                    sensitivity.append("MINUS ↓")
                    declines += 1
                if not sensitivity:
                    sensitivity.append("no change")
                    neutral += 1
                sens_str = "  |  ".join(sensitivity)
                print(f"  {short_pid:<6} {b_score:>9} {p_score:>7} {p_arrow}  {m_score:>7} {m_arrow}   {sens_str}")
            else:
                print(f"  {short_pid:<6} {str(b_score):>9} {str(p_score):>10} {str(m_score):>10}   [missing scores]")

        print(f"  {SEP[:98]}")
        total = len([p for p in pids if p in results])
        print(f"  Summary: {improvements}/{total} improved on PLUS  |  {declines}/{total} declined on MINUS  |  {neutral}/{total} no change")

    # ── Per-proposal summary ──────────────────────────────────────────────────
    print()
    print(SEP2)
    print("  PER-PROPOSAL SUMMARY (target rubrics only)")
    print(f"  {'PID':<6}  {'R3 Rev':>8}  {'R4 Reg':>8}  {'R6 Par':>8}  {'R8 Tim':>8}   Verdict")
    print(f"  {'':6}  {'B→P→M':>8}  {'B→P→M':>8}  {'B→P→M':>8}  {'B→P→M':>8}")
    print(SEP2)

    for short_pid in pids:
        if short_pid not in results:
            continue
        b = results[short_pid]["baseline"]
        p = results[short_pid]["plus"]
        m = results[short_pid]["minus"]

        cols = []
        confirmed = 0
        total_checks = 0
        for rubric in TARGET_RUBRICS:
            b_s = b.get(rubric) if b else None
            p_s = p.get(rubric) if p else None
            m_s = m.get(rubric) if m else None
            if b_s is None:
                cols.append("  n/a")
                continue
            col = f"{b_s}→{p_s if p_s is not None else '?'}→{m_s if m_s is not None else '?'}"
            cols.append(f"{col:>8}")

            # PLUS rubrics: R3, R4 — check PLUS improves
            if rubric in ("strategy_R3_revenue", "strategy_R4_regulatory"):
                total_checks += 1
                if p_s is not None and p_s > b_s:
                    confirmed += 1
            # MINUS rubrics: R6, R8 — check MINUS declines
            if rubric in ("strategy_R6_partners", "strategy_R8_timing"):
                total_checks += 1
                if m_s is not None and m_s < b_s:
                    confirmed += 1

        verdict = f"{confirmed}/{total_checks} checks passed"
        print(f"  {short_pid:<6}  {'  '.join(cols)}   {verdict}")

    print(SEP2)
    print()
    print("  Legend: B=Baseline  P=PLUS (inject evidence)  M=MINUS (strip evidence)")
    print(f"  Stored: {out_path}")
    print()


if __name__ == "__main__":
    main()
