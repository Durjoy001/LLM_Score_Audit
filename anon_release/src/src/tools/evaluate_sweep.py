# -*- coding: utf-8 -*-
"""
Evaluation: sweep consistency for LLM scoring.

Runs the full pipeline (Stages 1-7) multiple times for the same proposal and
measures how stable the dimension scores are across runs. Scores are read
directly from each run's final_report.md as 0-1 floats — no integer conversion
is applied, so run-to-run variance is fully visible.

By default the full pipeline is re-run each iteration so that stochastic LLM
variance in the upstream stages propagates to the final scores. Pass
--scores_only to skip the pipeline and only re-extract from the existing
report (fast, but produces zero variance since extraction is deterministic).

Outputs:
  - src/data/evaluations/runs/<run_id>/sweep_summary.json
  - src/data/evaluations/runs/<run_id>/sweep_report.xlsx
  - src/data/evaluations/runs/<run_id>/sweep_bundle.zip
"""

import os
import sys
import json
import uuid
import argparse
import subprocess
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from openpyxl import Workbook

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.score_extraction import extract_report_scores

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
EVAL_DIR = DATA_DIR / "evaluations"
RUNS_DIR = EVAL_DIR / "runs"
REPORT_DIR = DATA_DIR / "reports"
PREPARED_DIR = DATA_DIR / "prepared"

# All numeric score dimensions (0-1 floats)
SCORE_DIMS = ["team", "objective", "strategy", "advantages", "feasibility", "overall_ranking"]
FIELDS_ALL = SCORE_DIMS + ["verdict"]

PIPELINE_SCRIPTS_FROM_PREPARED = [
    ("extract_facts_by_chunk", ["--proposal_id"]),
    ("build_dimensions_from_facts", ["--proposal_id"]),
    ("generate_questions", ["--proposal_id"]),
    ("llm_answering", ["--proposal_id", "--qs_file"]),
    ("post_processing", ["--pid", "--qs_file"]),
    ("ai_expert_opinion", ["--pid"]),
    ("generate_final_report", ["--pid"]),
]


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / (n - 1)) ** 0.5


def _mode(values: List[Any]):
    counts: Dict[Any, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None, 0
    mode_val = max(counts.items(), key=lambda x: (x[1], str(x[0])))[0]
    return mode_val, counts[mode_val]


def _cv(values: List[float]) -> Optional[float]:
    """Coefficient of variation (stddev / mean). None if mean is near zero."""
    if not values:
        return None
    mu = _mean(values)
    if abs(mu) < 1e-9:
        return None
    return round(_stddev(values) / abs(mu), 4)


def _run_script(script_name: str, extra_args: list) -> None:
    cmd = [sys.executable, f"src/tools/{script_name}.py"] + extra_args
    _log("RUN", f"Pipeline step: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline step failed: {script_name}")


def _run_pipeline_from_prepared(pid: str) -> None:
    qs_file = str(DATA_DIR / "questions" / pid / "generated_questions.json")
    for script_name, arg_keys in PIPELINE_SCRIPTS_FROM_PREPARED:
        extra: list = []
        for key in arg_keys:
            if key in ("--proposal_id", "--pid"):
                extra += [key, pid]
            elif key == "--qs_file":
                extra += [key, qs_file]
        _run_script(script_name, extra)


def detect_latest_pid() -> str:
    if not REPORT_DIR.exists():
        return ""
    cands = [
        (p.name.replace("_final_report.md", ""), p.stat().st_mtime)
        for p in REPORT_DIR.glob("*_final_report.md")
    ]
    if not cands:
        return ""
    return max(cands, key=lambda x: x[1])[0]


def summarize(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "runs": len(scores),
        "fields": {},
        "overall": {},
    }

    # Check if all runs produced identical scores
    all_identical = True
    if scores:
        first = scores[0]
        for s in scores[1:]:
            if any(s.get(k) != first.get(k) for k in FIELDS_ALL):
                all_identical = False
                break
    summary["overall"]["all_exact_match"] = all_identical

    for field in FIELDS_ALL:
        values = [s.get(field) for s in scores if s.get(field) is not None]
        mode_val, mode_count = _mode(values)
        n = len(values)

        field_info: Dict[str, Any] = {
            "mode": mode_val,
            "mode_count": mode_count,
        }

        if field in SCORE_DIMS:
            num_vals = [float(v) for v in values]
            stddev = round(_stddev(num_vals), 6)
            field_info.update({
                "mean": round(_mean(num_vals), 4),
                "stddev": stddev,
                "cv": _cv(num_vals),
                "min": round(min(num_vals), 4) if num_vals else None,
                "max": round(max(num_vals), 4) if num_vals else None,
                "range": round(max(num_vals) - min(num_vals), 4) if num_vals else None,
            })
        else:
            # verdict: categorical — exact match rate is meaningful here
            exact_rate = round(mode_count / n, 4) if n else 0.0
            field_info["exact_rate"] = exact_rate

        summary["fields"][field] = field_info

    return summary


def write_excel(
    out_path: Path,
    scores: List[Dict[str, Any]],
    summary: Dict[str, Any],
    run_files: List[Dict[str, str]],
) -> None:
    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = "summary"
    ws_sum.append(["metric", "value"])
    ws_sum.append(["runs", summary.get("runs")])
    ws_sum.append(["all_exact_match", summary.get("overall", {}).get("all_exact_match")])
    ws_sum.append(["", ""])

    ws_sum.append([
        "field", "mode", "mode_count",
        "mean", "stddev (sample)", "cv", "min", "max", "range", "exact_rate",
    ])
    for field, info in summary.get("fields", {}).items():
        ws_sum.append([
            field,
            info.get("mode"),
            info.get("mode_count"),
            info.get("mean"),
            info.get("stddev"),
            info.get("cv"),
            info.get("min"),
            info.get("max"),
            info.get("range"),
            info.get("exact_rate"),  # only populated for verdict
        ])

    ws_runs = wb.create_sheet("runs")
    ws_runs.append(["run", "report_file", *FIELDS_ALL])
    for idx, s in enumerate(scores, start=1):
        files = run_files[idx - 1] if idx - 1 < len(run_files) else {}
        ws_runs.append([
            idx,
            files.get("report", ""),
            *[s.get(f) for f in FIELDS_ALL],
        ])

    wb.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep score consistency for a proposal.")
    ap.add_argument("--pid", type=str, default="", help="Proposal ID; latest report if omitted.")
    ap.add_argument("--runs", type=int, default=5, help="Number of sweep runs (min 2).")
    ap.add_argument("--out_dir", type=str, default="", help="Optional output directory.")
    ap.add_argument("--scores_only", action="store_true",
                    help="Skip pipeline re-run; only re-extract from the existing report. "
                         "Fast, but produces zero variance since extraction is deterministic.")
    # Backward-compatibility aliases — hidden from --help
    ap.add_argument("--skip_pipeline", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--full_pipeline", action="store_true", help=argparse.SUPPRESS)
    # Kept for backward compat with callers but ignored (scoring is now deterministic)
    ap.add_argument("--temperature", type=float, default=0.2, help=argparse.SUPPRESS)
    ap.add_argument("--threshold", type=float, default=0.8, help=argparse.SUPPRESS)
    ap.add_argument("--provider", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--model", type=str, default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.runs < 2:
        raise RuntimeError("Sweep runs must be >= 2.")

    skip_pipeline = args.scores_only or args.skip_pipeline

    pid = args.pid.strip() or detect_latest_pid()
    if not pid:
        raise RuntimeError("No report found to sweep.")

    report_path = REPORT_DIR / f"{pid}_final_report.md"
    if not report_path.exists():
        raise FileNotFoundError(f"Final report not found: {report_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("INFO", (
        f"Sweep: pid={pid}, runs={args.runs}, "
        f"mode={'scores_only' if skip_pipeline else 'full_pipeline'}"
    ))

    scores: List[Dict[str, Any]] = []
    run_files: List[Dict[str, str]] = []

    for i in range(1, args.runs + 1):
        run_dir = out_dir / f"run_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if not skip_pipeline:
            _log("INFO", f"Run {i}/{args.runs}: running full pipeline...")
            _run_pipeline_from_prepared(pid)
        else:
            _log("INFO", f"Run {i}/{args.runs}: extracting scores from report...")

        report_text = report_path.read_text(encoding="utf-8")
        run_scores = extract_report_scores(report_text)
        scores.append(run_scores)

        copied_report = run_dir / "final_report.md"
        if report_path.exists():
            shutil.copy2(report_path, copied_report)

        run_files.append({
            "report": copied_report.name if copied_report.exists() else "",
        })

    summary = summarize(scores)
    summary_payload = {
        "meta": {
            "run_id": run_id,
            "generated_at": now_str(),
            "pid": pid,
            "runs": args.runs,
            "mode": "scores_only" if skip_pipeline else "full_pipeline",
        },
        "summary": summary,
    }

    summary_path = out_dir / "sweep_summary.json"
    xlsx_path = out_dir / "sweep_report.xlsx"
    write_json(summary_path, summary_payload)
    write_excel(xlsx_path, scores, summary, run_files)

    zip_path = out_dir / "sweep_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_path, arcname=summary_path.name)
        zf.write(xlsx_path, arcname=xlsx_path.name)
        for i in range(1, args.runs + 1):
            run_dir = out_dir / f"run_{i}"
            if not run_dir.exists():
                continue
            for item in run_dir.iterdir():
                if item.is_file():
                    zf.write(item, arcname=f"run_{i}/{item.name}")

    _log("OK", f"summary written: {summary_path}")
    _log("OK", f"report written: {xlsx_path}")
    _log("OK", f"bundle written: {zip_path}")


if __name__ == "__main__":
    main()
