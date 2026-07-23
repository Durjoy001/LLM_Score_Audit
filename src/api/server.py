# -*- coding: utf-8 -*-
"""
FastAPI server for YangtzeDelta Proposal Analysis
"""

import os
import sys
import json
import uuid
import asyncio
import ast
import shutil
import smtplib
import subprocess
import re
import time
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import (
    StreamingResponse, FileResponse, HTMLResponse, JSONResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
SRC_DIR    = BASE_DIR / "src"
DATA_DIR   = SRC_DIR  / "data"
PROPOSALS  = DATA_DIR / "proposals"
REPORTS    = DATA_DIR / "reports"
EVAL_DIR   = DATA_DIR / "evaluations"
HUMAN_EVAL_DIR = EVAL_DIR / "human"
EVAL_RUNS_DIR  = EVAL_DIR / "runs"
FRONTEND   = BASE_DIR / "frontend"
OUTPUT_DIR = BASE_DIR / "output"

PIPELINE_STEPS = [
    ("prepare_proposal_text",   "Extracting text from document"),
    ("extract_facts_by_chunk",  "Extracting key facts"),
    ("build_dimensions_from_facts", "Building analysis dimensions"),
    ("generate_questions",      "Generating evaluation questions"),
    ("llm_answering",           "Running LLM analysis"),
    ("post_processing",         "Post-processing answers"),
    ("ai_expert_opinion",       "Generating expert opinion"),
    ("generate_final_report",   "Compiling final report"),
    ("generate_llm_scores",     "Generating LLM rubric scores"),
]
PROVIDER_RUN_RE = re.compile(r"\[RUN\]\s+provider=(\S+)\s+stage=(\S+)\s+pid=(\S+)")
PROVIDER_LIST_RE = re.compile(r"healthy_providers=(\[[^\]]*\])")

# ─── in-memory job store ───────────────────────────────────────────────────────
# job_id → {"status": str, "pid": str, "steps": [...], "error": str|None}
jobs: dict[str, dict] = {}
eval_jobs: dict[str, dict] = {}
batches: dict[str, dict] = {}

# ─── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="YangtzeDelta Proposal Analyser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── helpers ──────────────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
COMMON_GMAIL_TYPOS = {
    "gm.co",
    "gm.com",
    "gmai.com",
    "gmail.co",
    "gamil.com",
    "gmal.com",
    "gmail.cm",
    "gmail.con",
    "gmail.om",
    "gnail.com",
}


def validate_notification_email(raw: str) -> str:
    """Validate and normalize a notification email address before accepting it."""
    email = (raw or "").strip()
    if not email:
        return ""
    if len(email) > 254:
        raise ValueError("Email address is too long.")
    parsed_name, parsed_email = parseaddr(email)
    if parsed_name or parsed_email != email:
        raise ValueError("Use a plain email address without a display name.")
    if not EMAIL_RE.fullmatch(email):
        raise ValueError("Enter a valid email address, for example name@gmail.com.")
    local, domain = email.rsplit("@", 1)
    domain = domain.lower()
    if local.startswith(".") or local.endswith(".") or ".." in local:
        raise ValueError("The email username part is not valid.")
    if any(part.startswith("-") or part.endswith("-") for part in domain.split(".")):
        raise ValueError("The email domain is not valid.")
    if domain in COMMON_GMAIL_TYPOS:
        raise ValueError("That looks like a Gmail typo. Did you mean gmail.com?")
    return f"{local}@{domain}"


def send_report_email(to: str, report_path: str, original_filename: str) -> dict:
    """Send the finished report — tries Resend API first, falls back to SMTP."""
    try:
        to = validate_notification_email(to)
        report_data = Path(report_path).read_bytes()
    except FileNotFoundError:
        msg = f"Report file not found: {report_path}"
        print(f"[email] {msg}", flush=True)
        return {"ok": False, "provider": None, "error": msg}
    except ValueError as exc:
        msg = str(exc)
        print(f"[email] invalid recipient {to!r}: {msg}", flush=True)
        return {"ok": False, "provider": None, "error": msg}

    report_name = Path(report_path).name
    last_provider_error = ""

    # ── Option 1: Resend API (zero SMTP config needed) ────────────────────────
    resend_key = os.getenv("RESEND_API_KEY", "")
    if resend_key:
        import base64
        import requests as _requests
        try:
            resp = _requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_key}"},
                json={
                    "from":    "Yangtze Delta <onboarding@resend.dev>",
                    "to":      [to],
                    "subject": f"✅ Your Proposal Analysis is Ready — {original_filename}",
                    "text":    (
                        f"Hello,\n\n"
                        f"Great news! Your proposal \"{original_filename}\" has been fully analysed by the Yangtze Delta AI system.\n\n"
                        f"📎 The complete report is attached to this email as a Markdown (.md) file. "
                        f"You can open it with any Markdown viewer, Notion, Obsidian, VS Code, or a plain text editor.\n\n"
                        f"The report covers:\n"
                        f"  • Team & Background\n"
                        f"  • Objectives & Vision\n"
                        f"  • Market Strategy\n"
                        f"  • Innovation & Technology\n"
                        f"  • Feasibility & Risk\n"
                        f"  • Overall Expert Opinion\n\n"
                        f"If you have any questions or would like to run another analysis, visit the platform and upload a new proposal.\n\n"
                        f"Best regards,\n"
                        f"Yangtze Delta AI Analysis System"
                    ),
                    "attachments": [{
                        "filename": report_name,
                        "content":  base64.b64encode(report_data).decode(),
                    }],
                },
                timeout=15,
            )
            if resp.status_code == 200 or resp.status_code == 201:
                print(f"[email] Report sent to {to} via Resend ✓", flush=True)
                return {"ok": True, "provider": "resend", "error": ""}
            else:
                last_provider_error = f"Resend error {resp.status_code}: {resp.text[:300]}"
                print(f"[email] {last_provider_error}", flush=True)
        except Exception as exc:
            last_provider_error = f"Resend exception: {exc}"
            print(f"[email] {last_provider_error}", flush=True)

    # ── Option 2: SMTP fallback ───────────────────────────────────────────────
    host  = os.getenv("SMTP_HOST", "")
    port  = int(os.getenv("SMTP_PORT", "587"))
    user  = os.getenv("SMTP_USER", "")
    pwd   = os.getenv("SMTP_PASS", "")
    from_ = os.getenv("SMTP_FROM", user)

    if not all([host, user, pwd]):
        msg = last_provider_error or "No email provider configured. Set RESEND_API_KEY or SMTP_* environment variables."
        print(f"[email] {msg}", flush=True)
        return {"ok": False, "provider": None, "error": msg}

    msg = EmailMessage()
    msg["Subject"] = f"✅ Your Proposal Analysis is Ready — {original_filename}"
    msg["From"]    = from_
    msg["To"]      = to
    msg.set_content(
        f"Hello,\n\n"
        f"Great news! Your proposal \"{original_filename}\" has been fully analysed by the Yangtze Delta AI system.\n\n"
        f"📎 The complete report is attached to this email as a Markdown (.md) file. "
        f"You can open it with any Markdown viewer, Notion, Obsidian, VS Code, or a plain text editor.\n\n"
        f"The report covers:\n"
        f"  • Team & Background\n"
        f"  • Objectives & Vision\n"
        f"  • Market Strategy\n"
        f"  • Innovation & Technology\n"
        f"  • Feasibility & Risk\n"
        f"  • Overall Expert Opinion\n\n"
        f"If you have any questions or would like to run another analysis, visit the platform and upload a new proposal.\n\n"
        f"Best regards,\n"
        f"Yangtze Delta AI Analysis System"
    )
    msg.add_attachment(report_data, maintype="text", subtype="markdown",
                       filename=report_name)

    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(user, pwd)
            smtp.send_message(msg)
        print(f"[email] Report sent to {to} via SMTP", flush=True)
        return {"ok": True, "provider": "smtp", "error": ""}
    except Exception as exc:
        msg = f"SMTP failed: {exc}"
        print(f"[email] {msg}", flush=True)
        return {"ok": False, "provider": "smtp", "error": msg}


def python_bin() -> str:
    """Return the venv python that is running this server."""
    return sys.executable


def _safe_log_name(value: str) -> str:
    """Return a filesystem-safe short name for session log files."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in value)
    return cleaned[:140] or "proposal"


def _create_session_log_path(pid: str, job_id: str) -> Path:
    """Create a unique log path under output/ for one upload-to-report session."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{stamp}_{_safe_log_name(pid)}_{job_id[:8]}_pipeline_log.txt"


def _append_session_log(log_path: str | Path, message: str) -> None:
    """Append one line/block to the durable per-session log."""
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message)
        if not message.endswith("\n"):
            f.write("\n")


def _log_session_event(log_path: str | Path, level: str, message: str) -> None:
    """Append a timestamped event to the durable per-session log."""
    _append_session_log(log_path, f"[{_utcnow()}] [{level}] {message}")


def _build_step_cmd(script_name: str, pid: str, upload_path: str) -> list[str]:
    """Build the full command for a pipeline script with explicit pid args."""
    qs_file = str(DATA_DIR / "questions" / pid / "generated_questions.json")
    script_file = f"src/tools/{script_name}.py"
    args_map = {
        "prepare_proposal_text":   ["--file", upload_path, "--proposal_id", pid],
        "extract_facts_by_chunk":  ["--proposal_id", pid],
        "build_dimensions_from_facts": ["--proposal_id", pid],
        "generate_questions":      ["--proposal_id", pid],
        "llm_answering":           ["--proposal_id", pid, "--qs_file", qs_file, "--providers", "auto"],
        "post_processing":         ["--pid", pid, "--qs_file", qs_file],
        "ai_expert_opinion":       ["--pid", pid, "--providers", "auto"],
        "generate_final_report":   ["--pid", pid],
        "generate_llm_scores":     ["--pid", pid, "--providers", "auto"],
    }
    if script_name == "generate_llm_scores":
        script_file = "src/tools/generate_multi_provider_scores.py"
    elif script_name == "ai_expert_opinion":
        script_file = "src/tools/generate_multi_provider_expert_opinions.py"
    extra = args_map.get(script_name, [])
    return [python_bin(), script_file] + extra


def _run_step(script_name: str, pid: str, upload_path: str, log_path: str = "") -> tuple[bool, str]:
    """Run one pipeline script, stream output to terminal and session log, return (ok, tail)."""
    cmd = _build_step_cmd(script_name, pid, upload_path)
    header = (
        f"\n{'='*60}\n"
        f"STEP: {script_name}  [pid={pid}]\n"
        f"COMMAND: {' '.join(cmd)}\n"
        f"STARTED_AT: {_utcnow()}\n"
        f"{'='*60}\n"
    )
    print(header, end="", flush=True)
    _append_session_log(log_path, header)
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        _append_session_log(log_path, line)
        lines.append(line)
    proc.wait()
    ok = proc.returncode == 0
    tail = "".join(lines[-50:])
    if not ok:
        msg = f"STEP FAILED exit_code={proc.returncode} script={script_name} finished_at={_utcnow()}"
        print(f"{msg}", flush=True)
        _log_session_event(log_path, "ERROR", msg)
    else:
        msg = f"STEP DONE script={script_name} finished_at={_utcnow()}"
        print(f"{msg}", flush=True)
        _log_session_event(log_path, "OK", msg)
    return ok, tail


def _run_eval_job(eval_id: str, human_path: str, out_dir: str, provider: str) -> tuple[bool, str]:
    cmd = [python_bin(), "src/tools/evaluate_cohens_kappa.py", "--human_xlsx", human_path, "--out_dir", out_dir]
    if provider:
        cmd += ["--provider", provider]
    header = (
        f"\n{'='*60}\n"
        f"EVAL: cohens_kappa  [eval_id={eval_id}]\n"
        f"COMMAND: {' '.join(cmd)}\n"
        f"STARTED_AT: {_utcnow()}\n"
        f"{'='*60}\n"
    )
    print(header, end="", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    ok = proc.returncode == 0
    tail = "".join(lines[-50:])
    return ok, tail


def _run_groundedness_job(eval_id: str, pid: str, out_dir: str, provider: str, model: str) -> tuple[bool, str]:
    cmd = [python_bin(), "src/tools/evaluate_groundedness.py", "--pid", pid, "--out_dir", out_dir]
    if provider:
        cmd += ["--provider", provider]
    if model:
        cmd += ["--model", model]
    header = (
        f"\n{'='*60}\n"
        f"EVAL: groundedness  [eval_id={eval_id}]\n"
        f"COMMAND: {' '.join(cmd)}\n"
        f"STARTED_AT: {_utcnow()}\n"
        f"{'='*60}\n"
    )
    print(header, end="", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    ok = proc.returncode == 0
    tail = "".join(lines[-50:])
    return ok, tail


def _run_sweep_job(eval_id: str, pid: str, runs: int, temperature: float, out_dir: str, provider: str, model: str) -> tuple[bool, str]:
    cmd = [
        python_bin(),
        "src/tools/evaluate_sweep.py",
        "--pid",
        pid,
        "--runs",
        str(runs),
        "--temperature",
        str(temperature),
        "--out_dir",
        out_dir,
    ]
    if provider:
        cmd += ["--provider", provider]
    if model:
        cmd += ["--model", model]
    header = (
        f"\n{'='*60}\n"
        f"EVAL: sweep  [eval_id={eval_id}]\n"
        f"COMMAND: {' '.join(cmd)}\n"
        f"STARTED_AT: {_utcnow()}\n"
        f"{'='*60}\n"
    )
    print(header, end="", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    ok = proc.returncode == 0
    tail = "".join(lines[-50:])
    return ok, tail


def _run_sensitivity_job(eval_id: str, pids: str, baseline_pid: str, out_dir: str, provider: str, model: str) -> tuple[bool, str]:
    cmd = [
        python_bin(), "src/tools/evaluate_sensitivity.py",
        "--pids", pids,
        "--out_dir", out_dir,
    ]
    if baseline_pid:
        cmd += ["--baseline_pid", baseline_pid]
    if provider:
        cmd += ["--provider", provider]
    if model:
        cmd += ["--model", model]
    header = (
        f"\n{'='*60}\n"
        f"EVAL: sensitivity  [eval_id={eval_id}]\n"
        f"COMMAND: {' '.join(cmd)}\n"
        f"STARTED_AT: {_utcnow()}\n"
        f"{'='*60}\n"
    )
    print(header, end="", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    ok = proc.returncode == 0
    tail = "".join(lines[-50:])
    return ok, tail


def _new_provider_progress(provider: str, provider_pid: str = "") -> dict:
    return {
        "provider": provider,
        "provider_pid": provider_pid,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "steps": [
            {"name": script, "label": label, "status": "pending",
             "started_at": None, "finished_at": None, "error": None}
            for script, label in PIPELINE_STEPS
        ],
    }


def _ensure_provider_progress(job: dict, provider: str, provider_pid: str = "") -> dict:
    providers = job.setdefault("provider_steps", [])
    for entry in providers:
        if entry.get("provider") == provider:
            if provider_pid and not entry.get("provider_pid"):
                entry["provider_pid"] = provider_pid
            return entry
    entry = _new_provider_progress(provider, provider_pid)
    providers.append(entry)
    return entry


def _mark_provider_stage_running(job_id: str, provider: str, provider_pid: str, stage: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    job["provider_lifecycle_mode"] = True
    entry = _ensure_provider_progress(job, provider, provider_pid)
    now = _utcnow()
    entry["status"] = "running"
    entry["started_at"] = entry["started_at"] or now
    stage_names = [script for script, _ in PIPELINE_STEPS]
    stage_idx = stage_names.index(stage) if stage in stage_names else -1
    if stage_idx >= 0:
        job["current_step"] = stage_idx
    for idx, step in enumerate(entry.get("steps", [])):
        if idx < stage_idx and step.get("status") == "pending":
            step["status"] = "done"
            step["finished_at"] = step["finished_at"] or now
        if step.get("name") == stage:
            step["status"] = "running"
            step["started_at"] = step["started_at"] or now
            step["finished_at"] = None
            step["error"] = None
        elif step.get("status") == "running":
            step["status"] = "done"
            step["finished_at"] = step["finished_at"] or now


def _mark_provider_lifecycle_declared(job_id: str, pid: str, providers: list[str]) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    job["provider_lifecycle_mode"] = True
    for provider in providers:
        _ensure_provider_progress(job, provider, f"{pid}__{provider}")


def _sync_provider_lifecycle_summary(job_id: str, pid: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    summary_path = DATA_DIR / "provider_lifecycles" / pid / "provider_lifecycle_summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return
    job["provider_lifecycle_summary"] = str(summary_path)
    job["provider_steps"] = []
    now = _utcnow()
    for result in summary.get("results", []):
        provider = result.get("provider", "")
        if not provider:
            continue
        entry = _ensure_provider_progress(job, provider, result.get("provider_pid", ""))
        entry["status"] = "done" if result.get("ok") else "error"
        entry["started_at"] = entry["started_at"] or now
        entry["finished_at"] = now
        stage_results = {s.get("stage"): s for s in result.get("stages", []) if isinstance(s, dict)}
        for step in entry.get("steps", []):
            stage_result = stage_results.get(step["name"])
            if not stage_result:
                if entry["status"] == "done":
                    step["status"] = "done"
                    step["finished_at"] = step["finished_at"] or now
                continue
            step["started_at"] = stage_result.get("started_at") or step.get("started_at")
            step["finished_at"] = stage_result.get("finished_at") or step.get("finished_at")
            if stage_result.get("ok"):
                step["status"] = "done"
            else:
                step["status"] = "error"
                step["error"] = stage_result.get("tail") or result.get("error") or "Provider stage failed."
        if result.get("error"):
            entry["error"] = result.get("error")


def _run_provider_lifecycle_job(job_id: str, pid: str, upload_path: str, log_path: str = "") -> tuple[bool, str]:
    cmd = [
        python_bin(),
        "src/tools/run_multi_provider_lifecycle.py",
        "--file",
        upload_path,
        "--pid",
        pid,
        "--providers",
        "auto",
        "--workers",
        "1",
    ]
    header = (
        f"\n{'='*60}\n"
        f"PIPELINE: provider_lifecycle  [pid={pid}]\n"
        f"COMMAND: {' '.join(cmd)}\n"
        f"STARTED_AT: {_utcnow()}\n"
        f"{'='*60}\n"
    )
    print(header, end="", flush=True)
    _append_session_log(log_path, header)
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        _append_session_log(log_path, line)
        lines.append(line)
        provider_list_match = PROVIDER_LIST_RE.search(line)
        if provider_list_match:
            try:
                declared_providers = ast.literal_eval(provider_list_match.group(1))
            except Exception:
                declared_providers = []
            if isinstance(declared_providers, list):
                _mark_provider_lifecycle_declared(
                    job_id,
                    pid,
                    [str(provider) for provider in declared_providers if str(provider).strip()],
                )
        match = PROVIDER_RUN_RE.search(line)
        if match:
            provider, stage, provider_pid = match.groups()
            _mark_provider_stage_running(job_id, provider, provider_pid, stage)
    proc.wait()
    ok = proc.returncode == 0
    _sync_provider_lifecycle_summary(job_id, pid)
    tail = "".join(lines[-80:])
    return ok, tail


async def _run_pipeline(job_id: str, pid: str):
    """Background coroutine that drives the pipeline and updates job state."""
    job = jobs[job_id]
    job["status"] = "running"
    upload_path = job.get("upload_path", "")
    log_path = job.get("log_path", "")
    progress_file = DATA_DIR / f"step_progress_{pid}.json"
    _log_session_event(log_path, "PIPELINE", f"Starting pipeline job_id={job_id} pid={pid}")
    _log_session_event(log_path, "PIPELINE", f"Uploaded file path={upload_path}")

    try:
        use_provider_lifecycle = os.getenv("USE_PROVIDER_LIFECYCLE", "true").strip().lower() not in {"0", "false", "no"}
        if use_provider_lifecycle:
            job["current_step"] = 0
            job["provider_lifecycle_mode"] = True
            job["provider_steps"] = []
            for step in job["steps"]:
                step["status"] = "pending"
                step["started_at"] = None
                step["finished_at"] = None
                step["error"] = None
            loop = asyncio.get_running_loop()
            ok, tail = await loop.run_in_executor(
                None, _run_provider_lifecycle_job, job_id, pid, upload_path, log_path
            )
            if not ok:
                _sync_provider_lifecycle_summary(job_id, pid)
                for provider in job.get("provider_steps", []):
                    if provider.get("status") == "running":
                        provider["status"] = "error"
                    for step in provider.get("steps", []):
                        if step.get("status") == "running":
                            step["status"] = "error"
                            step["error"] = tail
                            step["finished_at"] = _utcnow()
                job["status"] = "error"
                job["error"] = f"Provider lifecycle failed:\n{tail}"
                _log_session_event(log_path, "ERROR", "Provider lifecycle pipeline failed")
                _append_session_log(log_path, "\n--- Failure Tail ---\n" + tail)
                return
            _sync_provider_lifecycle_summary(job_id, pid)
            for step in job["steps"]:
                step["status"] = "done"
                step["finished_at"] = step["finished_at"] or _utcnow()
            summary_path = DATA_DIR / "provider_lifecycles" / pid / "provider_lifecycle_summary.json"
            if summary_path.exists():
                job["provider_lifecycle_summary"] = str(summary_path)
            _log_session_event(log_path, "PIPELINE", "Provider lifecycle pipeline completed")
        else:
            for i, (script, label) in enumerate(PIPELINE_STEPS):
                job["current_step"] = i
                job["steps"][i]["status"] = "running"
                job["steps"][i]["started_at"] = _utcnow()
                _log_session_event(log_path, "STEP", f"Starting {i + 1}/{len(PIPELINE_STEPS)} {script}: {label}")

                # reset sub-step progress for the new step
                try:
                    progress_file.write_text('{"done":0,"total":0}', encoding="utf-8")
                except Exception:
                    pass

                # run in thread so we don't block the event loop
                loop = asyncio.get_running_loop()
                ok, tail = await loop.run_in_executor(
                    None, _run_step, script, pid, upload_path, log_path
                )

                job["steps"][i]["finished_at"] = _utcnow()
                if ok:
                    job["steps"][i]["status"] = "done"
                    _log_session_event(log_path, "STEP", f"Completed {script}")
                else:
                    job["steps"][i]["status"] = "error"
                    job["steps"][i]["error"] = tail
                    job["status"] = "error"
                    job["error"] = f"Step '{label}' failed:\n{tail}"
                    _log_session_event(log_path, "ERROR", f"Pipeline failed at step={script}")
                    _append_session_log(log_path, "\n--- Failure Tail ---\n" + tail)
                    return

        _log_session_event(log_path, "PIPELINE", "All pipeline steps completed")
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        _log_session_event(log_path, "ERROR", f"Unhandled pipeline exception: {exc}")
        return
    finally:
        # clean up per-pid progress file
        try:
            progress_file.unlink(missing_ok=True)
        except Exception:
            pass

    # locate the report
    report_path = REPORTS / f"{pid}_final_report.md"
    if report_path.exists():
        job["report_path"] = str(report_path)
    else:
        matches = list(REPORTS.glob(f"{pid}*.md"))
        job["report_path"] = str(matches[0]) if matches else ""
    _log_session_event(log_path, "OUTPUT", f"report_path={job.get('report_path', '')}")
    _log_session_event(log_path, "OUTPUT", f"session_log_path={log_path}")

    # send email if address was provided
    recipient = job.get("email", "")
    if recipient and job.get("report_path"):
        job["email_status"] = "sending"
        job["email_error"] = ""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, send_report_email, recipient, job["report_path"], job["filename"]
        )
        job["email_status"] = "sent" if result.get("ok") else "failed"
        job["email_provider"] = result.get("provider")
        job["email_error"] = result.get("error", "")
        _log_session_event(
            log_path,
            "EMAIL",
            f"Email send finished recipient={recipient} status={job['email_status']} "
            f"provider={job.get('email_provider') or '<none>'} error={job.get('email_error') or '<none>'}"
        )
    elif recipient:
        job["email_status"] = "failed"
        job["email_error"] = "Report file was not available for email delivery."
        _log_session_event(log_path, "EMAIL", f"Email send skipped recipient={recipient} error={job['email_error']}")

    job["status"] = "done"
    job["current_step"] = len(PIPELINE_STEPS)


async def _auto_run_kappa(batch_id: str) -> None:
    """Trigger Cohen's kappa agreement eval after a batch finishes, if human scores exist."""
    batch = batches[batch_id]
    human_xlsx = HUMAN_EVAL_DIR / "human_scores.xlsx"
    if not human_xlsx.exists():
        batch["auto_eval_status"] = "skipped_no_human_scores"
        return
    eval_id = str(uuid.uuid4())
    out_dir = str((EVAL_RUNS_DIR / eval_id).resolve())
    eval_jobs[eval_id] = {
        "eval_id": eval_id,
        "status": "running",
        "type": "cohens_kappa",
        "human_path": str(human_xlsx),
        "provider": "",
        "out_dir": out_dir,
        "error": "",
        "created_at": _utcnow(),
        "batch_id": batch_id,
    }
    batch["auto_eval_status"] = "running"
    batch["auto_eval_id"] = eval_id
    loop = asyncio.get_running_loop()
    ok, tail = await loop.run_in_executor(None, _run_eval_job, eval_id, str(human_xlsx), out_dir, "")
    ev = eval_jobs.get(eval_id, {})
    if ok:
        ev["status"] = "done"
        batch["auto_eval_status"] = "done"
    else:
        ev["status"] = "error"
        ev["error"] = tail
        batch["auto_eval_status"] = "error"
        batch["auto_eval_error"] = tail[-500:]
    eval_jobs[eval_id] = ev


async def _run_batch(batch_id: str) -> None:
    """Run all jobs in a batch sequentially, then auto-trigger agreement eval."""
    batch = batches[batch_id]
    batch["status"] = "running"
    batch["started_at"] = _utcnow()
    for idx, job_entry in enumerate(batch["jobs"]):
        job_id = job_entry["job_id"]
        batch["current_index"] = idx
        pid = jobs[job_id]["pid"]
        await _run_pipeline(job_id, pid)
    batch["finished_at"] = _utcnow()
    done_count = sum(1 for j in batch["jobs"] if jobs.get(j["job_id"], {}).get("status") == "done")
    batch["status"] = "done"
    batch["done_count"] = done_count
    batch["error_count"] = len(batch["jobs"]) - done_count
    await _auto_run_kappa(batch_id)


# ─── routes ───────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_proposal(
    file: UploadFile = File(...),
    email: str = Form(""),
):
    """Accept a proposal file, save it, return a job_id."""
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".ppt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    try:
        email = validate_notification_email(email)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    PROPOSALS.mkdir(parents=True, exist_ok=True)

    # derive a clean proposal id from filename — append short job_id for uniqueness
    job_id = str(uuid.uuid4())
    stem = Path(file.filename).stem
    safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    pid = f"{safe_stem or 'proposal'}_{job_id[:8]}"

    dest = PROPOSALS / f"{pid}{suffix}"
    content = await file.read()
    dest.write_bytes(content)
    meta_path = PROPOSALS / f"{pid}.meta.json"
    meta_path.write_text(json.dumps({
        "pid": pid,
        "job_id": job_id,
        "original_filename": file.filename,
        "original_stem": Path(file.filename).stem,
        "uploaded_at": _utcnow(),
        "upload_path": str(dest),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = _create_session_log_path(pid, job_id)
    _log_session_event(log_path, "UPLOAD", "New upload session created")
    _log_session_event(log_path, "UPLOAD", f"job_id={job_id}")
    _log_session_event(log_path, "UPLOAD", f"proposal_id={pid}")
    _log_session_event(log_path, "UPLOAD", f"original_filename={file.filename}")
    _log_session_event(log_path, "UPLOAD", f"saved_upload_path={dest}")
    _log_session_event(log_path, "UPLOAD", f"upload_size_bytes={len(content)}")
    _log_session_event(log_path, "UPLOAD", f"pipeline_steps={', '.join(script for script, _ in PIPELINE_STEPS)}")

    jobs[job_id] = {
        "job_id":       job_id,
        "pid":          pid,
        "filename":     file.filename,
        "upload_path":  str(dest),
        "email":        email,
        "email_status": "pending" if email else "none",
        "email_error":  "",
        "email_provider": None,
        "log_path":     str(log_path),
        "status":       "queued",
        "current_step": -1,
        "steps": [
            {"name": script, "label": label, "status": "pending",
             "started_at": None, "finished_at": None, "error": None}
            for script, label in PIPELINE_STEPS
        ],
        "error":       None,
        "report_path": "",
        "created_at":  _utcnow(),
    }
    return {"job_id": job_id, "pid": pid, "log_path": str(log_path)}


@app.post("/api/run/{job_id}")
async def run_pipeline(job_id: str):
    """Start the pipeline for an already-uploaded job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] not in ("queued",):
        raise HTTPException(400, f"Job is already in state: {job['status']}")

    asyncio.create_task(_run_pipeline(job_id, job["pid"]))
    _log_session_event(job.get("log_path", ""), "PIPELINE", "Run request accepted; background task scheduled")
    return {"started": True}


@app.patch("/api/jobs/{job_id}/email")
async def update_email(job_id: str, payload: dict):
    """Update or add a notification email for an existing job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    try:
        email = validate_notification_email(payload.get("email") or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    jobs[job_id]["email"] = email
    jobs[job_id]["email_status"] = "pending" if email else "none"
    jobs[job_id]["email_error"] = ""
    jobs[job_id]["email_provider"] = None
    return {"ok": True, "email": email}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/events/{job_id}")
async def sse_events(job_id: str):
    """
    Server-Sent Events stream — pushes job state every 1 s until done/error.
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def generator():
        prog_file = DATA_DIR / f"step_progress_{jobs[job_id]['pid']}.json"
        while True:
            job = jobs.get(job_id, {})
            job_copy = json.loads(json.dumps(job, ensure_ascii=False))
            if prog_file.exists():
                try:
                    prog = json.loads(prog_file.read_text(encoding="utf-8"))
                    for step in job_copy["steps"]:
                        if step["status"] == "running":
                            step["sub_progress"] = prog
                            break
                except Exception:
                    pass
            data = json.dumps(job_copy)
            yield f"data: {data}\n\n"
            if job.get("status") in ("done", "error"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/report/{job_id}")
async def get_report(job_id: str):
    """Return the final Markdown report as plain text."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Pipeline not finished yet")
    rp = job.get("report_path", "")
    if not rp or not Path(rp).exists():
        raise HTTPException(404, "Report file not found")
    return JSONResponse({"markdown": Path(rp).read_text(encoding="utf-8"),
                         "pid": job["pid"]})


@app.get("/api/download/{job_id}")
async def download_report(job_id: str):
    """Download the final Markdown report as a file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Pipeline not finished yet")
    rp = job.get("report_path", "")
    if not rp or not Path(rp).exists():
        raise HTTPException(404, "Report file not found")
    return FileResponse(
        path=rp,
        media_type="text/markdown",
        filename=f"{job['pid']}_final_report.md",
    )


@app.post("/api/batch/upload")
async def batch_upload(files: List[UploadFile] = File(...)):
    """Upload multiple proposals for sequential batch processing."""
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".ppt"}
    if not files:
        raise HTTPException(400, "No files provided")
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 files per batch")
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    batch_id = str(uuid.uuid4())
    job_list = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            raise HTTPException(400, f"Unsupported file type: {suffix} ({file.filename})")
        job_id = str(uuid.uuid4())
        stem = Path(file.filename).stem
        safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
        pid = f"{safe_stem or 'proposal'}_{job_id[:8]}"
        dest = PROPOSALS / f"{pid}{suffix}"
        content = await file.read()
        dest.write_bytes(content)
        (PROPOSALS / f"{pid}.meta.json").write_text(json.dumps({
            "pid": pid, "job_id": job_id, "original_filename": file.filename,
            "original_stem": stem, "uploaded_at": _utcnow(),
            "upload_path": str(dest), "batch_id": batch_id,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path = _create_session_log_path(pid, job_id)
        jobs[job_id] = {
            "job_id": job_id, "pid": pid, "filename": file.filename,
            "upload_path": str(dest), "email": "", "email_status": "none",
            "email_error": "", "email_provider": None, "log_path": str(log_path),
            "status": "queued", "current_step": -1,
            "steps": [{"name": s, "label": l, "status": "pending",
                        "started_at": None, "finished_at": None, "error": None}
                       for s, l in PIPELINE_STEPS],
            "error": None, "report_path": "", "created_at": _utcnow(), "batch_id": batch_id,
        }
        job_list.append({"job_id": job_id, "pid": pid, "filename": file.filename})
    batches[batch_id] = {
        "batch_id": batch_id, "status": "pending", "jobs": job_list,
        "current_index": -1, "created_at": _utcnow(), "started_at": None,
        "finished_at": None, "done_count": 0, "error_count": 0,
        "auto_eval_status": None, "auto_eval_id": None, "auto_eval_error": None,
    }
    return {"batch_id": batch_id, "jobs": job_list}


@app.post("/api/batch/run/{batch_id}")
async def run_batch(batch_id: str):
    """Start sequential processing of a batch."""
    if batch_id not in batches:
        raise HTTPException(404, "Batch not found")
    if batches[batch_id]["status"] != "pending":
        raise HTTPException(400, f"Batch already in state: {batches[batch_id]['status']}")
    asyncio.create_task(_run_batch(batch_id))
    return {"started": True}


@app.get("/api/batch/events/{batch_id}")
async def batch_events(batch_id: str):
    """SSE stream — pushes full batch + all job states every 1s until done/error."""
    if batch_id not in batches:
        raise HTTPException(404, "Batch not found")

    async def generator():
        while True:
            batch = batches.get(batch_id, {})
            payload = {
                "batch_id": batch.get("batch_id"),
                "status": batch.get("status"),
                "current_index": batch.get("current_index", -1),
                "started_at": batch.get("started_at"),
                "finished_at": batch.get("finished_at"),
                "done_count": batch.get("done_count", 0),
                "error_count": batch.get("error_count", 0),
                "auto_eval_status": batch.get("auto_eval_status"),
                "auto_eval_id": batch.get("auto_eval_id"),
                "auto_eval_error": batch.get("auto_eval_error"),
                "jobs": [jobs.get(j["job_id"], j) for j in batch.get("jobs", [])],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if batch.get("status") in ("done", "error"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/batch/{batch_id}")
async def get_batch(batch_id: str):
    """Snapshot of a batch and all its job states."""
    if batch_id not in batches:
        raise HTTPException(404, "Batch not found")
    batch = batches[batch_id]
    return {**{k: v for k, v in batch.items() if k != "jobs"},
            "jobs": [jobs.get(j["job_id"], j) for j in batch.get("jobs", [])]}


@app.post("/api/evaluation/human_upload")
async def upload_human_scores(file: UploadFile = File(...)):
    """Upload the human Excel score sheet for evaluation."""
    allowed = {".xlsx"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = HUMAN_EVAL_DIR / "human_scores.xlsx"
    content = await file.read()
    if dest.exists():
        backup = HUMAN_EVAL_DIR / f"human_scores_backup_{int(time.time())}.xlsx"
        shutil.copy2(dest, backup)
    dest.write_bytes(content)
    return {"ok": True, "path": str(dest)}


@app.post("/api/evaluation/run")
async def run_evaluation(payload: dict | None = None):
    """Start the evaluation job (Cohen's kappa)."""
    payload = payload or {}
    human_path = payload.get("human_path") or str(HUMAN_EVAL_DIR / "human_scores.xlsx")
    provider = str(payload.get("provider") or "").strip().lower()
    if not Path(human_path).exists():
        raise HTTPException(404, f"Human score file not found: {human_path}")

    eval_id = str(uuid.uuid4())
    out_dir = str((EVAL_RUNS_DIR / eval_id).resolve())
    eval_jobs[eval_id] = {
        "eval_id": eval_id,
        "status": "running",
        "type": "cohens_kappa",
        "human_path": human_path,
        "provider": provider,
        "out_dir": out_dir,
        "error": "",
        "created_at": _utcnow(),
    }

    async def _run():
        loop = asyncio.get_running_loop()
        ok, tail = await loop.run_in_executor(None, _run_eval_job, eval_id, human_path, out_dir, provider)
        job = eval_jobs.get(eval_id, {})
        if ok:
            job["status"] = "done"
        else:
            job["status"] = "error"
            job["error"] = tail
        eval_jobs[eval_id] = job

    asyncio.create_task(_run())
    return {"eval_id": eval_id, "out_dir": out_dir}


@app.post("/api/evaluation/groundedness/run")
async def run_groundedness(payload: dict | None = None):
    payload = payload or {}
    pid = payload.get("pid") or ""
    provider = str(payload.get("provider") or "").strip().lower()
    model = str(payload.get("model") or "").strip()
    if not pid:
        # fall back to latest report
        from src.tools.evaluate_groundedness import detect_latest_pid
        pid = detect_latest_pid()
    if not pid:
        raise HTTPException(404, "No report found to evaluate groundedness")

    eval_id = str(uuid.uuid4())
    out_dir = str((EVAL_RUNS_DIR / eval_id).resolve())
    eval_jobs[eval_id] = {
        "eval_id": eval_id,
        "status": "running",
        "type": "groundedness",
        "pid": pid,
        "provider": provider,
        "model": model,
        "out_dir": out_dir,
        "error": "",
        "created_at": _utcnow(),
    }

    async def _run():
        loop = asyncio.get_running_loop()
        ok, tail = await loop.run_in_executor(None, _run_groundedness_job, eval_id, pid, out_dir, provider, model)
        job = eval_jobs.get(eval_id, {})
        if ok:
            job["status"] = "done"
        else:
            job["status"] = "error"
            job["error"] = tail
        eval_jobs[eval_id] = job

    asyncio.create_task(_run())
    return {"eval_id": eval_id, "out_dir": out_dir, "pid": pid}


@app.post("/api/evaluation/sweep/run")
async def run_sweep(payload: dict | None = None):
    payload = payload or {}
    pid = payload.get("pid") or ""
    try:
        runs = int(payload.get("runs") or 5)
    except (TypeError, ValueError):
        raise HTTPException(400, "'runs' must be an integer")
    try:
        temperature = float(payload.get("temperature") or 0.2)
    except (TypeError, ValueError):
        raise HTTPException(400, "'temperature' must be a number")
    provider = str(payload.get("provider") or "").strip().lower()
    model = str(payload.get("model") or "").strip()
    if runs < 2:
        raise HTTPException(400, "Sweep runs must be >= 2")
    if not pid:
        from src.tools.generate_llm_scores import detect_latest_pid
        pid = detect_latest_pid()
    if not pid:
        raise HTTPException(404, "No report found to sweep")

    eval_id = str(uuid.uuid4())
    out_dir = str((EVAL_RUNS_DIR / eval_id).resolve())
    eval_jobs[eval_id] = {
        "eval_id": eval_id,
        "status": "running",
        "type": "sweep",
        "pid": pid,
        "runs": runs,
        "temperature": temperature,
        "provider": provider,
        "model": model,
        "out_dir": out_dir,
        "error": "",
        "created_at": _utcnow(),
    }

    async def _run():
        loop = asyncio.get_running_loop()
        ok, tail = await loop.run_in_executor(None, _run_sweep_job, eval_id, pid, runs, temperature, out_dir, provider, model)
        job = eval_jobs.get(eval_id, {})
        if ok:
            job["status"] = "done"
        else:
            job["status"] = "error"
            job["error"] = tail
        eval_jobs[eval_id] = job

    asyncio.create_task(_run())
    return {"eval_id": eval_id, "out_dir": out_dir, "pid": pid}


@app.get("/api/evaluation/status/{eval_id}")
async def get_evaluation_status(eval_id: str):
    if eval_id not in eval_jobs:
        raise HTTPException(404, "Evaluation job not found")
    return eval_jobs[eval_id]


@app.get("/api/evaluation/summary/{eval_id}")
async def get_evaluation_summary(eval_id: str):
    job = eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    summary_path = Path(job.get("out_dir", "")) / "evaluation_summary.json"
    if not summary_path.exists():
        raise HTTPException(404, "Evaluation summary not found")
    return JSONResponse(json.loads(summary_path.read_text(encoding="utf-8")))


@app.get("/api/evaluation/download/{eval_id}")
async def download_evaluation_report(eval_id: str):
    job = eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    report_path = Path(job.get("out_dir", "")) / "evaluation_report.xlsx"
    if not report_path.exists():
        raise HTTPException(404, "Evaluation report not found")
    return FileResponse(
        path=str(report_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="evaluation_report.xlsx",
    )


@app.get("/api/evaluation/groundedness/download/{eval_id}")
async def download_groundedness_report(eval_id: str):
    job = eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    report_path = Path(job.get("out_dir", "")) / "groundedness_result.json"
    if not report_path.exists():
        raise HTTPException(404, "Groundedness result not found")
    return FileResponse(
        path=str(report_path),
        media_type="application/json",
        filename="groundedness_result.json",
    )


@app.get("/api/evaluation/sweep/download/{eval_id}")
async def download_sweep_report(eval_id: str):
    job = eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    report_path = Path(job.get("out_dir", "")) / "sweep_report.xlsx"
    if not report_path.exists():
        raise HTTPException(404, "Sweep report not found")
    return FileResponse(
        path=str(report_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="sweep_report.xlsx",
    )


@app.get("/api/evaluation/sweep/download_zip/{eval_id}")
async def download_sweep_bundle(eval_id: str):
    job = eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    report_path = Path(job.get("out_dir", "")) / "sweep_bundle.zip"
    if not report_path.exists():
        raise HTTPException(404, "Sweep bundle not found")
    return FileResponse(
        path=str(report_path),
        media_type="application/zip",
        filename="sweep_bundle.zip",
    )

def _run_full_pipeline_for_sensitivity(
    pid: str,
    upload_path: str,
    eval_id: str,
    file_idx: int,
    total_files: int,
    label: str,
) -> tuple[bool, str]:
    """Run all 9 pipeline stages for one file within a sensitivity job, updating progress."""
    for script, step_label in PIPELINE_STEPS:
        job = eval_jobs.get(eval_id, {})
        job["progress_text"] = f"[{file_idx}/{total_files}] {label} — {step_label}"
        eval_jobs[eval_id] = job

        ok, tail = _run_step(script, pid, upload_path)
        if not ok:
            return False, f"Pipeline failed at '{step_label}' for '{label}':\n{tail}"
    return True, ""


@app.post("/api/evaluation/sensitivity/upload_and_run")
async def sensitivity_upload_and_run(
    baseline: UploadFile = File(...),
    variants: List[UploadFile] = File(...),
    provider: str = Form(""),
    model: str = Form(""),
):
    """Accept baseline + variant files, run full pipeline on each, then compare."""
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".ppt"}
    all_uploads = [("baseline", baseline, True)] + [
        (f"variant_{i+1}", v, False) for i, v in enumerate(variants)
    ]

    # Validate extensions before saving anything
    for role, uf, _ in all_uploads:
        if Path(uf.filename).suffix.lower() not in allowed:
            raise HTTPException(400, f"Unsupported file type for {role}: {uf.filename}")

    # Save each file and create a pid (same pattern as /api/upload)
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    file_configs = []
    for role, uf, is_baseline in all_uploads:
        suffix = Path(uf.filename).suffix.lower()
        stem = Path(uf.filename).stem
        safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:40]
        pid = f"sens_{safe_stem}_{str(uuid.uuid4())[:8]}"
        dest = PROPOSALS / f"{pid}{suffix}"
        content = await uf.read()
        dest.write_bytes(content)
        (PROPOSALS / f"{pid}.meta.json").write_text(json.dumps({
            "pid": pid,
            "job_id": "",
            "original_filename": uf.filename,
            "original_stem": stem,
            "uploaded_at": _utcnow(),
            "upload_path": str(dest),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        file_configs.append({
            "pid": pid,
            "label": uf.filename,
            "upload_path": str(dest),
            "is_baseline": is_baseline,
        })

    eval_id = str(uuid.uuid4())
    out_dir = str((EVAL_RUNS_DIR / eval_id).resolve())
    eval_jobs[eval_id] = {
        "eval_id": eval_id,
        "status": "running",
        "type": "sensitivity",
        "total_files": len(file_configs),
        "completed_files": 0,
        "progress_text": "Starting pipelines…",
        "provider": provider.strip().lower(),
        "model": model.strip(),
        "out_dir": out_dir,
        "error": "",
        "created_at": _utcnow(),
    }

    async def _run():
        all_pids = [c["pid"] for c in file_configs]
        baseline_pid = next(c["pid"] for c in file_configs if c["is_baseline"])
        loop = asyncio.get_running_loop()

        for idx, cfg in enumerate(file_configs, start=1):
            ok, tail = await loop.run_in_executor(
                None,
                _run_full_pipeline_for_sensitivity,
                cfg["pid"], cfg["upload_path"], eval_id, idx, len(file_configs), cfg["label"],
            )
            if not ok:
                job = eval_jobs.get(eval_id, {})
                job["status"] = "error"
                job["error"] = tail
                eval_jobs[eval_id] = job
                return
            job = eval_jobs.get(eval_id, {})
            job["completed_files"] = idx
            eval_jobs[eval_id] = job

        # All pipelines done — run the sensitivity comparison
        job = eval_jobs.get(eval_id, {})
        job["progress_text"] = "Comparing reports with LLM…"
        eval_jobs[eval_id] = job

        ok, tail = await loop.run_in_executor(
            None,
            _run_sensitivity_job,
            eval_id,
            ",".join(all_pids),
            baseline_pid,
            out_dir,
            provider.strip().lower(),
            model.strip(),
        )
        job = eval_jobs.get(eval_id, {})
        job["status"] = "done" if ok else "error"
        if not ok:
            job["error"] = tail
        eval_jobs[eval_id] = job

    asyncio.create_task(_run())
    return {"eval_id": eval_id, "total_files": len(file_configs)}


@app.post("/api/evaluation/sensitivity/run")
async def run_sensitivity(payload: dict | None = None):
    payload = payload or {}
    pids: list = payload.get("pids", [])
    baseline_pid: str = payload.get("baseline_pid", "")
    provider = str(payload.get("provider") or "").strip().lower()
    model = str(payload.get("model") or "").strip()
    if len(pids) < 2:
        raise HTTPException(400, "At least 2 proposal IDs required for sensitivity analysis.")
    # Validate all PIDs have reports + scores
    for pid in pids:
        report = REPORTS / f"{pid}_final_report.md"
        scores = DATA_DIR / "llm_scores" / pid / "llm_scores.json"
        if not report.exists():
            raise HTTPException(404, f"No final report found for pid={pid}. Run the full pipeline first.")
        if not scores.exists():
            raise HTTPException(404, f"No LLM scores found for pid={pid}. Run the full pipeline first.")

    pids_str = ",".join(pids)
    eval_id = str(uuid.uuid4())
    out_dir = str((EVAL_RUNS_DIR / eval_id).resolve())
    eval_jobs[eval_id] = {
        "eval_id": eval_id,
        "status": "running",
        "type": "sensitivity",
        "pids": pids,
        "baseline_pid": baseline_pid or pids[0],
        "provider": provider,
        "model": model,
        "out_dir": out_dir,
        "error": "",
        "created_at": _utcnow(),
    }

    async def _run():
        loop = asyncio.get_running_loop()
        ok, tail = await loop.run_in_executor(
            None, _run_sensitivity_job, eval_id, pids_str, baseline_pid, out_dir, provider, model
        )
        job = eval_jobs.get(eval_id, {})
        job["status"] = "done" if ok else "error"
        if not ok:
            job["error"] = tail
        eval_jobs[eval_id] = job

    asyncio.create_task(_run())
    return {"eval_id": eval_id, "out_dir": out_dir}


@app.get("/api/evaluation/sensitivity/summary/{eval_id}")
async def get_sensitivity_summary(eval_id: str):
    job = eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    summary_path = Path(job.get("out_dir", "")) / "sensitivity_summary.json"
    if not summary_path.exists():
        raise HTTPException(404, "Sensitivity summary not found")
    return JSONResponse(json.loads(summary_path.read_text(encoding="utf-8")))


@app.get("/api/evaluation/sensitivity/download/{eval_id}")
async def download_sensitivity_report(eval_id: str):
    job = eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    report_path = Path(job.get("out_dir", "")) / "sensitivity_report.xlsx"
    if not report_path.exists():
        raise HTTPException(404, "Sensitivity report not found")
    return FileResponse(
        path=str(report_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="sensitivity_report.xlsx",
    )


@app.get("/api/evaluation/sweep/summary/{eval_id}")
async def get_sweep_summary(eval_id: str):
    job = eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(404, "Evaluation job not found")
    summary_path = Path(job.get("out_dir", "")) / "sweep_summary.json"
    if not summary_path.exists():
        raise HTTPException(404, "Sweep summary not found")
    return JSONResponse(json.loads(summary_path.read_text(encoding="utf-8")))


@app.get("/api/evaluation/pids")
async def list_evaluation_pids():
    """Return available proposal IDs (based on reports) for evaluation selection."""
    pids = []
    if REPORTS.exists():
        for report in REPORTS.glob("*_final_report.md"):
            pid = report.name.replace("_final_report.md", "")
            meta_path = PROPOSALS / f"{pid}.meta.json"
            label = pid
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    label = meta.get("original_filename") or pid
                except Exception:
                    label = pid
            pids.append({"pid": pid, "label": label})
    pids.sort(key=lambda x: x["label"])
    return {"items": pids}


# ─── frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
