#!/usr/bin/env python3
"""
Batch runner: generate team rubric scores for all 12 proposals.

Usage:
    python run_team_rubric_batch.py
    python run_team_rubric_batch.py --provider openai
    python run_team_rubric_batch.py --dry_run
"""

import argparse
import subprocess
import sys
from pathlib import Path

PIDS = [
    "p1__openai", "p2__openai", "p3__openai", "p4__openai",
    "p5__openai", "p6__openai", "p7__openai", "p8__openai",
    "pA__openai", "pB__openai", "pC__openai", "pD__openai",
]

SCRIPT = Path(__file__).parent / "src" / "tools" / "generate_team_rubric_scores.py"


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-score team rubrics for all proposals.")
    ap.add_argument("--provider",    type=str, default="openai", help="LLM provider.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--dry_run",     action="store_true", help="Write prompts only; skip LLM calls.")
    args = ap.parse_args()

    passed, failed = [], []

    for pid in PIDS:
        print(f"\n{'='*55}")
        print(f"  Scoring: {pid}")
        print(f"{'='*55}")

        cmd = [
            sys.executable, str(SCRIPT),
            "--pid",      pid,
            "--provider", args.provider,
            "--temperature", str(args.temperature),
            "--use_llm",
        ]
        if args.dry_run:
            cmd.append("--dry_run")

        result = subprocess.run(cmd)
        if result.returncode == 0:
            passed.append(pid)
        else:
            failed.append(pid)

    print(f"\n{'='*55}")
    print(f"  Done. Passed: {len(passed)}  Failed: {len(failed)}")
    if failed:
        print(f"  Failed PIDs: {failed}")
    print(f"{'='*55}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
