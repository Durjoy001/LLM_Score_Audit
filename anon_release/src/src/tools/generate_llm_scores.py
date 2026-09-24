# -*- coding: utf-8 -*-
"""
Optional LLM scoring tool.

Asks an LLM to score a proposal report directly on the human rubric scale.
Requires --use_llm flag. This is an optional tool; the evaluation pipeline
(sweep, kappa, sensitivity) reads scores directly from the final report as
0-1 floats and does not require this script.

Outputs:
  - src/data/llm_scores/<pid>/llm_scores.json
  - src/data/llm_scores/<pid>/llm_scores.prompt.json

The scoring aligns with the human rubric:
  team, objective, strategy, advantages, feasibility (1-5)
  overall_ranking (1-20)
  verdict (Y/N)
"""

import os
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.llm_provider import chat_json, default_model

# ----------------- Paths and constants -----------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
REPORT_DIR = DATA_DIR / "reports"
EXPERT_DIR = DATA_DIR / "expert_reports"
SCORES_DIR = DATA_DIR / "llm_scores"
PROPOSALS_DIR = DATA_DIR / "proposals"

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_latest_pid() -> str:
    if not REPORT_DIR.exists():
        return ""
    cands = []
    for p in REPORT_DIR.glob("*_final_report.md"):
        pid = p.name.replace("_final_report.md", "")
        cands.append((pid, p.stat().st_mtime))
    if not cands:
        return ""
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0]


def _trim_text(text: str, max_chars: int = 12000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [TRUNCATED] ..."


def _load_meta(pid: str) -> Dict[str, Any]:
    meta_path = PROPOSALS_DIR / f"{pid}.meta.json"
    if not meta_path.exists():
        return {}
    try:
        return read_json(meta_path)
    except Exception:
        return {}


def _safe_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(round(float(value)))
    except Exception:
        return default
    return max(lo, min(hi, iv))



def _fallback_from_expert(expert_json: Dict[str, Any]) -> Dict[str, Any]:
    dims = expert_json.get("dimensions", {}) or {}
    overall = expert_json.get("overall_opinion", {}) or {}

    def _score_to_5(v: Any) -> int:
        try:
            fv = float(v)
        except Exception:
            return 3
        return _safe_int(1 + fv * 4.0, 1, 5, 3)

    def _score_to_20(v: Any) -> int:
        try:
            fv = float(v)
        except Exception:
            return 10
        return _safe_int(1 + fv * 19.0, 1, 20, 10)

    verdict_raw = (overall.get("verdict") or "").strip().upper()
    verdict = "Y" if verdict_raw == "GO" else "N"

    return {
        "team": _score_to_5((dims.get("team") or {}).get("score_echo")),
        "objective": _score_to_5((dims.get("objectives") or {}).get("score_echo")),
        "strategy": _score_to_5((dims.get("strategy") or {}).get("score_echo")),
        "advantages": _score_to_5((dims.get("innovation") or {}).get("score_echo")),
        "feasibility": _score_to_5((dims.get("feasibility") or {}).get("score_echo")),
        "overall_ranking": _score_to_20(overall.get("overall_score_echo")),
        "verdict": verdict,
        "score_source": "local_fallback",
    }


def call_openai_chat(
    provider: str,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    temperature: float,
) -> Dict[str, Any]:
    _log("LLM", f"Calling provider={provider} model={model}")
    return chat_json(
        provider=provider,
        model=model,
        temperature=float(temperature),
        max_tokens=900,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )


def sanitize_scores(raw: Dict[str, Any]) -> Dict[str, Any]:
    scores = {
        "team": _safe_int(raw.get("team"), 1, 5, 3),
        "objective": _safe_int(raw.get("objective"), 1, 5, 3),
        "strategy": _safe_int(raw.get("strategy"), 1, 5, 3),
        "advantages": _safe_int(raw.get("advantages"), 1, 5, 3),
        "feasibility": _safe_int(raw.get("feasibility"), 1, 5, 3),
        "overall_ranking": _safe_int(raw.get("overall_ranking"), 1, 20, 10),
        "verdict": (str(raw.get("verdict") or "").strip().upper()[:1] or "N"),
    }
    scores["verdict"] = "Y" if scores["verdict"] == "Y" else "N"
    if "score_source" in raw:
        scores["score_source"] = str(raw.get("score_source"))
    if "rationale" in raw:
        scores["rationale"] = raw.get("rationale")
    return scores


def build_system_prompt() -> str:
    return (
        "You are an evaluator. Return only JSON.\n"
        "Calibration context: these proposals have already passed initial screening. "
        "Scores of 1-2 should be reserved for a genuine critical gap or missing evidence. "
        "A score of 3 means common quality for a screened proposal — present but not impressive. "
        "A score of 4 means meaningfully strong evidence on this dimension. "
        "A score of 5 means exceptional and fully documented. "
        "Human expert reviewers in this domain regularly assign 3-5 for most dimensions of screened proposals; "
        "be willing to give 4s and 5s where the evidence clearly supports it.\n\n"
        "Score the proposal on these human rubric categories:\n"
        "- team (1-5): 1 = unclear or weak team signal, 3 = credible team with some gaps, 5 = strong, well-documented, and credible team.\n"
        "- objective (1-5): 1 = unclear objective or weak problem definition, 3 = clear objective with partial market evidence, 5 = crisp, well-supported, with named buyer and quantified need.\n"
        "- strategy (1-5): 1 = vague or unconvincing path, 3 = plausible route with partial evidence, 5 = specific, execution-ready with named partners or signed agreements.\n"
        "- advantages (1-5): 1 = little apparent differentiation, 3 = some advantage claimed with partial proof, 5 = clear, defensible, and documented advantage over alternatives.\n"
        "- feasibility (1-5): 1 = major feasibility gaps, 3 = reasonable plan with missing proof points, 5 = strong feasibility evidence with budget, milestones, and risk mitigation.\n"
        "- overall_ranking (1-20): rank relative to strong investment/diligence candidates; top proposals score 15-20.\n"
        "- verdict (Y or N): Y if at least 3 dimensions score 4 or above and you would advance for next-round diligence.\n\n"
        "Use only the provided report evidence. Do not invent facts. "
        "Prefer the report's own evidence, expert summary, and traceable specifics over generalized impressions. "
        "Output JSON with keys: team, objective, strategy, advantages, feasibility, overall_ranking, verdict."
    )


def build_user_payload(pid: str, report_text: str, expert_json: Dict[str, Any]) -> Dict[str, Any]:
    overall = expert_json.get("overall_opinion", {}) or {}
    return {
        "proposal_id": pid,
        "report_excerpt": report_text,
        "expert_summary": overall.get("summary") or "",
        "expert_verdict": overall.get("verdict") or "",
        "expert_key_risks": (overall.get("key_risks") or [])[:5],
        "expert_key_strengths": (overall.get("key_strengths") or [])[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate rubric scores for evaluation.")
    ap.add_argument("--pid", type=str, default="", help="Proposal ID; latest report if omitted.")
    ap.add_argument("--provider", type=str, default=PROVIDER, help="LLM provider: openai, gemini, or deepseek.")
    ap.add_argument("--model", type=str, default="", help="Model name. Defaults to provider-specific env model.")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (LLM mode only).")
    ap.add_argument("--out_dir", type=str, default="", help="Optional output directory.")
    ap.add_argument("--run_tag", type=str, default="", help="Optional run tag for sweep runs.")
    ap.add_argument("--dry_run", action="store_true", help="Write prompt only; skip LLM call (LLM mode only).")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of falling back to expert scores when LLM call fails (LLM mode only).",
    )
    ap.add_argument(
        "--use_llm",
        action="store_true",
        help="Required. Score via an LLM call. Without this flag the script exits immediately.",
    )
    args = ap.parse_args()

    if not args.use_llm:
        ap.error(
            "--use_llm is required. The evaluation pipeline now reads scores directly from "
            "final_report.md; this script is only needed for optional LLM-based scoring."
        )

    provider = (args.provider or PROVIDER).strip().lower()
    model = args.model.strip() or default_model(provider)

    pid = args.pid.strip() or detect_latest_pid()
    if not pid:
        raise RuntimeError("No report found to score.")

    report_path = REPORT_DIR / f"{pid}_final_report.md"
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    expert_path = EXPERT_DIR / pid / "ai_expert_opinion.json"
    expert_json: Dict[str, Any] = {}
    if expert_path.exists():
        try:
            expert_json = read_json(expert_path)
        except Exception:
            expert_json = {}

    meta = _load_meta(pid)
    out_dir = Path(args.out_dir) if args.out_dir else (SCORES_DIR / pid)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "llm_scores.json"

    result: Dict[str, Any]
    used_mode: str

    report_text = _trim_text(report_path.read_text(encoding="utf-8"))
    prompt_path = out_dir / "llm_scores.prompt.json"
    system_prompt = build_system_prompt()
    user_payload = build_user_payload(pid, report_text, expert_json)
    write_json(prompt_path, {"system": system_prompt, "user": user_payload})

    if args.dry_run:
        _log("OK", f"Prompt written: {prompt_path}")
        return

    can_use_llm = provider in {"openai", "deepseek", "gemini"}
    used_mode = "local_fallback"
    if can_use_llm:
        try:
            raw = call_openai_chat(provider, model, system_prompt, user_payload, args.temperature)
            result = sanitize_scores(raw)
            used_mode = "llm"
        except Exception as exc:
            if args.strict:
                raise RuntimeError(f"LLM scoring failed for provider={provider}: {exc}") from exc
            _log("WARN", f"LLM scoring failed; using fallback. error={exc}")
            result = _fallback_from_expert(expert_json)
    else:
        result = _fallback_from_expert(expert_json)

    payload = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "provider": provider,
            "model": model,
            "temperature": float(args.temperature),
            "mode": used_mode,
            "report_path": str(report_path),
            "expert_path": str(expert_path) if expert_path.exists() else "",
            "source_filename": meta.get("original_filename", ""),
            "source_filename_stem": meta.get("original_stem", ""),
            "job_id": meta.get("job_id", ""),
            "run_tag": args.run_tag,
        },
        "scores": result,
    }
    write_json(json_path, payload)
    _log("OK", f"scores written: {json_path}  mode={used_mode}")


if __name__ == "__main__":
    main()
