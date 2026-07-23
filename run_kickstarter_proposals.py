#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch pipeline runner for the Kickstarter dataset: Stages 0-8 for every
proposal in Kickstarter/Toronto, Canada/ and Kickstarter/Montreal, Canada/.

Unlike run_all_proposals.py (which also runs Cohen's kappa evaluation against
human expert scores), this script only runs the pipeline and collects final
reports. Ground truth here is `funded (%)`, not a human rubric, so evaluation
is a separate step: evaluate_kickstarter_correlation.py.

Outputs go to results/Kickstarter/reports/.
"""
import argparse
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PY = BASE_DIR / ".venv" / "bin" / "python"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable

sys.path.insert(0, str(BASE_DIR / "src" / "tools"))
from kickstarter_dataset import list_proposal_files, pilot_pids  # noqa: E402

REPORTS_SRC = BASE_DIR / "src" / "data" / "reports"
RESULTS = BASE_DIR / "results" / "Kickstarter"

_LOG_LOCK = threading.Lock()


def log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with _LOG_LOCK:
        print(f"[{ts}] [{level}] {msg}", flush=True)


def run_lifecycle(pid: str, file_path: Path) -> bool:
    log("PIPELINE", f"Starting pid={pid} file={file_path.name}")
    cmd = [
        PYTHON,
        "src/tools/run_multi_provider_lifecycle.py",
        "--file", str(file_path),
        "--pid", pid,
        "--providers", "openai",
        "--workers", "1",
    ]
    result = subprocess.run(cmd, cwd=str(BASE_DIR), text=True)
    ok = result.returncode == 0
    log("PIPELINE", f"{'OK' if ok else 'FAILED'} pid={pid}")
    return ok


def collect_report(pid: str) -> Path | None:
    candidate = REPORTS_SRC / f"{pid}__openai_final_report.md"
    if candidate.exists():
        return candidate
    candidate2 = REPORTS_SRC / f"{pid}_final_report.md"
    if candidate2.exists():
        return candidate2
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the pipeline for the Kickstarter dataset.")
    ap.add_argument("--pilot", action="store_true",
                     help="Run only the pilot subset (10 Toronto + 4 Montreal, spanning the funded%% range)")
    ap.add_argument("--workers", type=int, default=4,
                     help="Number of proposals to process concurrently (default: 4)")
    ap.add_argument("--force", action="store_true",
                     help="Re-run proposals even if a final report already exists (default: skip them)")
    args = ap.parse_args()

    all_proposals = dict(list_proposal_files())
    if args.pilot:
        pids = pilot_pids()
        target_set = [(pid, all_proposals[pid]) for pid in pids]
    else:
        target_set = list(all_proposals.items())

    # copy_candidates always covers the full target set (pilot or full), regardless of
    # --force, so an interrupted run's already-generated-but-not-yet-copied reports get
    # picked up on resume instead of being silently skipped.
    copy_candidates = target_set

    if args.force:
        proposals = target_set
    else:
        before = len(target_set)
        proposals = [(pid, path) for pid, path in target_set if collect_report(pid) is None]
        skipped_existing = before - len(proposals)
        if skipped_existing:
            log("SKIP", f"{skipped_existing} proposal(s) already have a report; skipping (use --force to redo)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    reports_out = RESULTS / "reports"
    reports_out.mkdir(parents=True, exist_ok=True)

    log("START", f"Processing {len(proposals)} proposals with workers={args.workers}")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_lifecycle, pid, path): pid for pid, path in proposals}
        for fut in as_completed(futures):
            pid = futures[fut]
            ok = fut.result()
            if not ok:
                failures.append(pid)

    log("COPY", "Collecting final reports -> results/Kickstarter/reports/")
    copied = []
    for pid, _ in copy_candidates:
        report = collect_report(pid)
        if report:
            dest = reports_out / f"{pid}_final_report.md"
            shutil.copy2(report, dest)
            copied.append(pid)
        else:
            log("WARN", f"  No report found for pid={pid}")
    log("COPY", f"Copied {len(copied)}/{len(copy_candidates)} reports")

    print("\n" + "=" * 60)
    log("DONE", f"Reports copied: {len(copied)}/{len(copy_candidates)}")
    log("DONE", f"Pipeline failures: {failures if failures else 'none'}")
    log("DONE", f"Results stored in: {RESULTS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
