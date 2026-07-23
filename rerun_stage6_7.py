#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-run Stage 6 (ai_expert_opinion) + Stage 7 (generate_final_report) for all 12 proposals,
then copy reports to Results/ and re-run evaluation.
"""
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PY  = BASE_DIR / ".venv" / "bin" / "python"
PYTHON   = str(VENV_PY) if VENV_PY.exists() else sys.executable

DATASET1  = BASE_DIR / "DeltaProjectDatasets" / "Dataset1"
DATASET2  = BASE_DIR / "DeltaProjectDatasets" / "Dataset2"
RESULTS   = BASE_DIR / "DeltaProjectDatasets" / "Results"
HUMAN_XLS = BASE_DIR / "DeltaProjectDatasets" / "ExpertsEvaluations.xlsx"

REPORTS_DIR = BASE_DIR / "src" / "data" / "reports"
EVAL_HUMAN  = BASE_DIR / "src" / "data" / "evaluations" / "human"

PROPOSALS = [
    ("p1",  DATASET1 / "1-附件1：心衰专病大模型BP.pdf"),
    ("p2",  DATASET1 / "2-附件1：2025_0410新型高端酶制剂的智能化创制与应用.pdf"),
    ("p3",  DATASET1 / "3-附件1：基于非天然氨基酸调控的高效抗体结合蛋白.pdf"),
    ("p4",  DATASET1 / "4-附件1：（简介版）中国首创的第二代肿瘤治疗电场（TTF）解决方案【水印】(1).pdf"),
    ("p5",  DATASET1 / "5-bp1.pdf"),
    ("p6",  DATASET1 / "6-bp2.pdf"),
    ("p7",  DATASET1 / "7-04-1.（项目经理汇报）LPNP-mRNA免疫调节因子项目.pptx"),
    ("p8",  DATASET1 / "8-2025_10 Ebovir_LNP 拨投结合项目商业计划书.pptx"),
    ("pA",  DATASET2 / "A_安海半导体产业化方案v1.4-YF(1)(1).pdf"),
    ("pB",  DATASET2 / "B_BMT技术（基石资本)v01.pptx"),
    ("pC",  DATASET2 / "C_1.ZX学院商业计划书20210328.pptx"),
    ("pD",  DATASET2 / "D_210614大瞬科技投决报告.pptx"),
]


def log(level: str, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def run_stages(pid: str, file_path: Path) -> bool:
    log("STAGE6+7", f"Starting pid={pid}")
    cmd = [
        PYTHON,
        "src/tools/run_multi_provider_lifecycle.py",
        "--file", str(file_path),
        "--pid",  pid,
        "--providers", "openai",
        "--from_stage", "ai_expert_opinion",
        "--to_stage",   "generate_final_report",
        "--workers", "1",
    ]
    result = subprocess.run(cmd, cwd=str(BASE_DIR), text=True)
    ok = result.returncode == 0
    log("STAGE6+7", f"{'OK' if ok else 'FAILED'} pid={pid}")
    return ok


def collect_report(pid: str) -> Path | None:
    for name in [f"{pid}__openai_final_report.md", f"{pid}_final_report.md"]:
        p = REPORTS_DIR / name
        if p.exists():
            return p
    return None


def run_evaluation(eval_out_dir: Path) -> bool:
    log("EVAL", f"Running evaluation → {eval_out_dir}")
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        "src/tools/evaluate_cohens_kappa.py",
        "--human_xlsx", str(EVAL_HUMAN / "human_scores.xlsx"),
        "--out_dir",    str(eval_out_dir),
    ]
    result = subprocess.run(cmd, cwd=str(BASE_DIR), text=True)
    ok = result.returncode == 0
    log("EVAL", f"{'OK' if ok else 'FAILED'}")
    return ok


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    reports_out = RESULTS / "reports"
    reports_out.mkdir(parents=True, exist_ok=True)
    eval_out = RESULTS / "evaluation"
    eval_out.mkdir(parents=True, exist_ok=True)

    log("START", f"Re-running Stage 6+7 for {len(PROPOSALS)} proposals")

    failures = []
    for pid, file_path in PROPOSALS:
        if not file_path.exists():
            log("SKIP", f"File not found: {file_path.name}")
            failures.append(pid)
            continue
        ok = run_stages(pid, file_path)
        if not ok:
            failures.append(pid)

    # Copy updated reports to Results/reports/
    log("COPY", "Collecting final reports → Results/reports/")
    copied = []
    for pid, _ in PROPOSALS:
        report = collect_report(pid)
        if report:
            dest = reports_out / f"{pid}_final_report.md"
            shutil.copy2(report, dest)
            copied.append(pid)
            log("COPY", f"  {pid} → {dest.name}")
        else:
            log("WARN", f"  No report found for pid={pid}")

    log("COPY", f"Copied {len(copied)}/{len(PROPOSALS)} reports")

    # Ensure human scores are in place
    if not HUMAN_XLS.exists():
        log("ERROR", f"ExpertsEvaluations.xlsx not found at {HUMAN_XLS}")
        sys.exit(1)

    EVAL_HUMAN.mkdir(parents=True, exist_ok=True)
    human_dest = EVAL_HUMAN / "human_scores.xlsx"
    if human_dest.exists():
        backup = EVAL_HUMAN / f"human_scores_backup_{int(time.time())}.xlsx"
        shutil.copy2(human_dest, backup)
    shutil.copy2(HUMAN_XLS, human_dest)
    log("EVAL", f"Copied ExpertsEvaluations.xlsx → {human_dest}")

    eval_ok = run_evaluation(eval_out)

    print("\n" + "=" * 60)
    log("DONE", f"Reports copied: {len(copied)}/{len(PROPOSALS)}")
    log("DONE", f"Stage 6+7 failures: {failures if failures else 'none'}")
    log("DONE", f"Evaluation: {'ok' if eval_ok else 'failed'}")
    log("DONE", f"Results → {RESULTS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
