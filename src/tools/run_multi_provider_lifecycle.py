# -*- coding: utf-8 -*-
"""Run independent full proposal lifecycles for multiple LLM providers.

Each provider gets its own proposal id:
  <base_pid>__openai
  <base_pid>__gemini
  <base_pid>__deepseek

This keeps Stage 0 through Stage 8 artifacts separate so provider outputs can
be evaluated as independent raters against human scores.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.backend.utils.llm_provider import configured_providers, default_model, probe_provider

DATA_DIR = BASE_DIR / "src" / "data"
PROPOSALS_DIR = DATA_DIR / "proposals"
REPORT_DIR = DATA_DIR / "reports"
LIFECYCLE_DIR = DATA_DIR / "provider_lifecycles"

load_dotenv()


STAGES = [
    "prepare_proposal_text",
    "extract_facts_by_chunk",
    "build_dimensions_from_facts",
    "generate_questions",
    "llm_answering",
    "post_processing",
    "ai_expert_opinion",
    "generate_final_report",
    "generate_llm_scores",
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_pid(value: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in value.strip())
    return cleaned[:140] or "proposal"


def _parse_providers(value: str) -> List[str]:
    value = (value or "auto").strip().lower()
    if value in {"", "auto", "all"}:
        return configured_providers()
    providers = [p.strip().lower() for p in value.split(",") if p.strip()]
    if "auto" in providers or "all" in providers:
        return configured_providers()
    return list(dict.fromkeys(providers))


def _preflight_providers(providers: List[str]) -> tuple[List[str], List[Dict[str, Any]]]:
    healthy: List[str] = []
    results: List[Dict[str, Any]] = []
    for provider in providers:
        model = default_model(provider)
        ok, reason = probe_provider(provider, model=model)
        result = {
            "provider": provider,
            "model": model,
            "ok": bool(ok),
            "reason": reason,
        }
        results.append(result)
        if ok:
            healthy.append(provider)
        else:
            _log("WARN", f"Skipping provider={provider} during lifecycle preflight: {reason}")
    return healthy, results


def _stage_cmd(stage: str, provider_pid: str, upload_path: Path, provider: str) -> List[str]:
    qs_file = DATA_DIR / "questions" / provider_pid / "generated_questions.json"
    if stage == "prepare_proposal_text":
        return [sys.executable, "src/tools/prepare_proposal_text.py", "--file", str(upload_path), "--proposal_id", provider_pid]
    if stage == "extract_facts_by_chunk":
        return [sys.executable, "src/tools/extract_facts_by_chunk.py", "--proposal_id", provider_pid]
    if stage == "build_dimensions_from_facts":
        return [sys.executable, "src/tools/build_dimensions_from_facts.py", "--proposal_id", provider_pid]
    if stage == "generate_questions":
        return [sys.executable, "src/tools/generate_questions.py", "--proposal_id", provider_pid, "--llm_provider", provider]
    if stage == "llm_answering":
        return [
            sys.executable,
            "src/tools/llm_answering.py",
            "--proposal_id",
            provider_pid,
            "--qs_file",
            str(qs_file),
            "--providers",
            provider,
        ]
    if stage == "post_processing":
        return [sys.executable, "src/tools/post_processing.py", "--pid", provider_pid, "--qs_file", str(qs_file)]
    if stage == "ai_expert_opinion":
        return [sys.executable, "src/tools/ai_expert_opinion.py", "--pid", provider_pid, "--provider", provider]
    if stage == "generate_final_report":
        return [sys.executable, "src/tools/generate_final_report.py", "--pid", provider_pid]
    if stage == "generate_llm_scores":
        return [sys.executable, "src/tools/generate_llm_scores.py", "--pid", provider_pid, "--provider", provider, "--use_llm"]
    raise ValueError(f"Unknown stage: {stage}")


def _write_provider_meta(base_pid: str, provider_pid: str, provider: str, upload_path: Path) -> None:
    original_filename = upload_path.name
    meta = {
        "pid": provider_pid,
        "base_pid": base_pid,
        "provider": provider,
        "provider_lifecycle": True,
        "original_filename": original_filename,
        "original_stem": Path(original_filename).stem,
        "uploaded_at": _utcnow(),
        "upload_path": str(upload_path),
    }
    write_json(PROPOSALS_DIR / f"{provider_pid}.meta.json", meta)


def _run_cmd(cmd: List[str], provider: str, provider_pid: str, stage: str, env: Dict[str, str]) -> Dict[str, Any]:
    started_at = _utcnow()
    _log("RUN", f"provider={provider} stage={stage} pid={provider_pid} cmd={' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail = (proc.stdout or "")[-4000:]
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    return {
        "stage": stage,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "started_at": started_at,
        "finished_at": _utcnow(),
        "tail": tail,
    }


def _run_provider_lifecycle(base_pid: str, provider: str, upload_path: Path, stages: List[str]) -> Dict[str, Any]:
    provider_pid = f"{base_pid}__{provider}"
    _write_provider_meta(base_pid, provider_pid, provider, upload_path)

    env = os.environ.copy()
    env["PROVIDER"] = provider
    env["PROVIDER_LIFECYCLE_BASE_PID"] = base_pid
    env["PROVIDER_LIFECYCLE_PROVIDER"] = provider
    env["PROVIDER_LIFECYCLE_PID"] = provider_pid

    provider_result: Dict[str, Any] = {
        "provider": provider,
        "provider_pid": provider_pid,
        "model": default_model(provider),
        "ok": True,
        "stages": [],
        "report_path": str(REPORT_DIR / f"{provider_pid}_final_report.md"),
        "scores_path": str(DATA_DIR / "llm_scores" / provider_pid / "llm_scores.json"),
    }

    for stage in stages:
        cmd = _stage_cmd(stage, provider_pid, upload_path, provider)
        stage_result = _run_cmd(cmd, provider, provider_pid, stage, env)
        provider_result["stages"].append(stage_result)
        if not stage_result["ok"]:
            provider_result["ok"] = False
            provider_result["failed_stage"] = stage
            break

    return provider_result


def _copy_primary_to_base(base_pid: str, provider_result: Dict[str, Any]) -> None:
    provider_pid = provider_result.get("provider_pid", "")
    if not provider_pid:
        return
    copies: List[Tuple[Path, Path]] = [
        (REPORT_DIR / f"{provider_pid}_final_report.md", REPORT_DIR / f"{base_pid}_final_report.md"),
        (
            REPORT_DIR / f"{provider_pid}_stage7_final_report_audit.json",
            REPORT_DIR / f"{base_pid}_stage7_final_report_audit.json",
        ),
        (
            DATA_DIR / "llm_scores" / provider_pid / "llm_scores.json",
            DATA_DIR / "llm_scores" / base_pid / "llm_scores.json",
        ),
        (
            DATA_DIR / "llm_scores" / provider_pid / "llm_scores.prompt.json",
            DATA_DIR / "llm_scores" / base_pid / "llm_scores.prompt.json",
        ),
    ]
    for src, dst in copies:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full independent provider lifecycles from Stage 0 to Stage 8.")
    ap.add_argument("--file", required=True, help="Proposal file path.")
    ap.add_argument("--pid", default="", help="Base proposal ID. Defaults to file stem.")
    ap.add_argument("--providers", default="auto", help="Comma-separated providers, or auto/all for keys found in .env.")
    ap.add_argument("--workers", type=int, default=0, help="Provider lifecycle workers; defaults to provider count.")
    ap.add_argument("--from_stage", default=STAGES[0], choices=STAGES)
    ap.add_argument("--to_stage", default=STAGES[-1], choices=STAGES)
    args = ap.parse_args()

    upload_path = Path(args.file).expanduser().resolve()
    if not upload_path.exists():
        raise FileNotFoundError(f"Proposal file not found: {upload_path}")

    base_pid = _safe_pid(args.pid or upload_path.stem)
    providers = _parse_providers(args.providers)
    if not providers:
        raise RuntimeError("No providers available. Configure keys or pass --providers explicitly.")

    start_idx = STAGES.index(args.from_stage)
    end_idx = STAGES.index(args.to_stage)
    if start_idx > end_idx:
        raise RuntimeError("--from_stage must come before --to_stage.")
    stages = STAGES[start_idx : end_idx + 1]

    _log("LIFECYCLE", f"base_pid={base_pid} configured_providers={providers} provider_count={len(providers)} stages={stages}")
    healthy_providers, preflight_results = _preflight_providers(providers)
    _log("LIFECYCLE", f"healthy_providers={healthy_providers} healthy_count={len(healthy_providers)}")
    if not healthy_providers:
        summary = {
            "base_pid": base_pid,
            "upload_path": str(upload_path),
            "generated_at": _utcnow(),
            "configured_providers": providers,
            "healthy_providers": [],
            "stages": stages,
            "preflight": preflight_results,
            "results": [],
        }
        summary_path = LIFECYCLE_DIR / base_pid / "provider_lifecycle_summary.json"
        write_json(summary_path, summary)
        raise RuntimeError("No healthy providers detected during preflight. Fix the broken API keys or provider access first.")

    workers = max(1, min(args.workers or len(healthy_providers), len(healthy_providers)))
    _log("LIFECYCLE", f"running_providers={healthy_providers} workers={workers}")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_run_provider_lifecycle, base_pid, provider, upload_path, stages): provider
            for provider in healthy_providers
        }
        completed: Dict[str, Dict[str, Any]] = {}
        for future in as_completed(future_map):
            provider = future_map[future]
            try:
                completed[provider] = future.result()
            except Exception as exc:
                completed[provider] = {
                    "provider": provider,
                    "provider_pid": f"{base_pid}__{provider}",
                    "ok": False,
                    "error": str(exc),
                }
        results = [completed[p] for p in healthy_providers if p in completed]

    preferred = os.getenv("PROVIDER", "").strip().lower()
    primary = next((r for r in results if r.get("ok") and r.get("provider") == preferred), None)
    primary = primary or next((r for r in results if r.get("ok")), None)
    # Only copy primary to base when multiple providers ran — for a single provider
    # the __<provider> output is the canonical artifact and copying would create
    # duplicate reports/scores that confuse the UI and evaluation matcher.
    if primary and len(healthy_providers) > 1:
        _copy_primary_to_base(base_pid, primary)

    summary = {
        "base_pid": base_pid,
        "upload_path": str(upload_path),
        "generated_at": _utcnow(),
        "providers": providers,
        "healthy_providers": healthy_providers,
        "stages": stages,
        "preflight": preflight_results,
        "primary_provider_copied_to_base": primary.get("provider") if primary else "",
        "results": results,
    }
    summary_path = LIFECYCLE_DIR / base_pid / "provider_lifecycle_summary.json"
    write_json(summary_path, summary)
    _log("OK", f"provider lifecycle summary written: {summary_path}")

    if not any(r.get("ok") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
