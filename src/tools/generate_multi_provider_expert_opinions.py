# -*- coding: utf-8 -*-
"""Generate Stage 6 expert opinions for all configured LLM providers."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.backend.utils.llm_provider import configured_providers, default_model, probe_provider

DATA_DIR = BASE_DIR / "src" / "data"
EXPERT_DIR = DATA_DIR / "expert_reports"
REFINED_ROOT = DATA_DIR / "refined_answers"

load_dotenv()


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_latest_pid() -> str:
    if not REFINED_ROOT.exists():
        return ""
    cands = []
    for d in REFINED_ROOT.iterdir():
        postproc = d / "postproc"
        if d.is_dir() and (postproc / "metrics.json").exists() and (postproc / "final_payload.json").exists():
            cands.append((d.name, (postproc / "metrics.json").stat().st_mtime))
    if not cands:
        return ""
    return max(cands, key=lambda x: x[1])[0]


def _parse_providers(value: str) -> List[str]:
    value = (value or "auto").strip().lower()
    if value in {"", "auto", "all"}:
        return configured_providers()
    providers = [p.strip().lower() for p in value.split(",") if p.strip()]
    if "auto" in providers or "all" in providers:
        return configured_providers()
    return providers


def _preflight_providers(providers: List[str]) -> tuple[List[str], List[Dict[str, Any]]]:
    healthy: List[str] = []
    results: List[Dict[str, Any]] = []
    for provider in providers:
        model = default_model(provider)
        ok, reason = probe_provider(provider, model=model)
        result = {"provider": provider, "model": model, "ok": bool(ok), "reason": reason}
        results.append(result)
        if ok:
            healthy.append(provider)
        else:
            _log("WARN", f"Skipping provider={provider} during preflight: {reason}")
    return healthy, results


def _run_provider(pid: str, provider: str) -> Dict[str, Any]:
    out_dir = EXPERT_DIR / pid / provider
    cmd = [
        sys.executable,
        "src/tools/ai_expert_opinion.py",
        "--pid",
        pid,
        "--provider",
        provider,
        "--out_dir",
        str(out_dir),
        "--strict",
    ]
    _log("RUN", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    payload_path = out_dir / "ai_expert_opinion.json"
    status: Dict[str, Any] = {
        "provider": provider,
        "ok": result.returncode == 0 and payload_path.exists(),
        "path": str(payload_path),
        "returncode": result.returncode,
    }
    if result.returncode != 0:
        status["error_tail"] = (result.stdout or "")[-1200:]
    if payload_path.exists():
        try:
            data = read_json(payload_path)
            meta = data.get("meta") or {}
            status["mode"] = meta.get("mode")
            status["model"] = meta.get("model")
            status["verdict"] = (data.get("overall_opinion") or {}).get("verdict")
            if status.get("mode") != "llm":
                status["ok"] = False
                status["error"] = f"provider returned non-LLM mode: {status.get('mode')}"
        except Exception as exc:
            status["ok"] = False
            status["error"] = str(exc)
    return status


def _copy_provider_to_root(pid: str, provider_path: Path) -> None:
    provider_dir = provider_path.parent
    root_dir = EXPERT_DIR / pid
    root_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "ai_expert_opinion.prompt.json",
        "ai_expert_opinion.json",
        "ai_expert_opinion.md",
        "stage6_ai_expert_opinion_audit.json",
    ):
        src = provider_dir / name
        if src.exists():
            (root_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage 6 expert opinions for multiple providers.")
    ap.add_argument("--pid", default="", help="Proposal ID; latest postproc payload if omitted.")
    ap.add_argument("--providers", default="auto", help="Comma-separated providers, or auto/all for keys found in .env.")
    ap.add_argument("--workers", type=int, default=0, help="Concurrent provider workers; defaults to provider count.")
    args = ap.parse_args()

    pid = args.pid.strip() or detect_latest_pid()
    if not pid:
        raise RuntimeError("No post-processed proposal found.")
    providers = _parse_providers(args.providers)
    if not providers:
        raise RuntimeError("No providers supplied.")

    _log("PROVIDERS", f"configured={providers} count={len(providers)}")
    healthy_providers, preflight_results = _preflight_providers(providers)
    _log("PROVIDERS", f"healthy={healthy_providers} healthy_count={len(healthy_providers)}")
    if not healthy_providers:
        raise RuntimeError("No healthy providers detected during preflight.")

    workers = max(1, min(args.workers or len(healthy_providers), len(healthy_providers)))
    if workers == 1:
        results = [_run_provider(pid, provider) for provider in healthy_providers]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_run_provider, pid, provider): provider for provider in healthy_providers}
            completed: Dict[str, Dict[str, Any]] = {}
            for future in as_completed(future_map):
                provider = future_map[future]
                try:
                    completed[provider] = future.result()
                except Exception as exc:
                    completed[provider] = {"provider": provider, "ok": False, "error": str(exc)}
            results = [completed[p] for p in healthy_providers if p in completed]

    primary = next((r for r in results if r.get("ok") and r.get("provider") == os.getenv("PROVIDER", "").strip().lower()), None)
    primary = primary or next((r for r in results if r.get("ok")), None)
    if primary and primary.get("path"):
        _copy_provider_to_root(pid, Path(primary["path"]))

    summary = {
        "pid": pid,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "providers": providers,
        "healthy_providers": healthy_providers,
        "preflight": preflight_results,
        "results": results,
        "primary_provider_copied_to_root": primary.get("provider") if primary else "",
    }
    summary_path = EXPERT_DIR / pid / "multi_provider_expert_opinions.json"
    write_json(summary_path, summary)
    _log("OK", f"multi-provider expert summary written: {summary_path}")


if __name__ == "__main__":
    main()
