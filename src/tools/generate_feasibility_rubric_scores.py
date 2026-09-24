# -*- coding: utf-8 -*-
"""
Feasibility sub-rubric scoring tool.

Scores a proposal on 8 granular feasibility sub-rubrics (R1-R8) using an LLM.
Results are stored separately from the main evaluation pipeline and do not
affect any existing scores or verdicts.

This tool is used to compare human investor judgment vs AI on the feasibility
dimension — specifically to show where AI and humans agree (document-verifiable
rubrics R1-R4) vs diverge (judgment-dependent rubrics R5-R8).

Outputs:
  - src/data/feasibility_rubric_scores/<pid>/feasibility_rubric_scores.json

Usage:
    python src/tools/generate_feasibility_rubric_scores.py --pid <proposal_id> --use_llm
    python src/tools/generate_feasibility_rubric_scores.py --use_llm   # latest proposal
"""

import os
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.llm_provider import chat_json, default_model

# ----------------- Paths -----------------
BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = BASE_DIR / "src" / "data"
REPORT_DIR  = DATA_DIR / "reports"
EXPERT_DIR  = DATA_DIR / "expert_reports"
OUT_DIR     = DATA_DIR / "feasibility_rubric_scores"
PROPOSALS_DIR = DATA_DIR / "proposals"

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

RUBRIC_KEYS = [
    "feasibility_R1_budget_detail",
    "feasibility_R2_funding_secured",
    "feasibility_R3_infrastructure",
    "feasibility_R4_technical_readiness",
    "feasibility_R5_risk_mitigation",
    "feasibility_R6_resource_timeline_fit",
    "feasibility_R7_financial_realism",
    "feasibility_R8_sustainability",
]

RUBRIC_TYPE = {
    "feasibility_R1_budget_detail":         "document-verifiable",
    "feasibility_R2_funding_secured":       "document-verifiable",
    "feasibility_R3_infrastructure":        "document-verifiable",
    "feasibility_R4_technical_readiness":   "document-verifiable",
    "feasibility_R5_risk_mitigation":       "judgment-dependent",
    "feasibility_R6_resource_timeline_fit": "judgment-dependent",
    "feasibility_R7_financial_realism":     "judgment-dependent",
    "feasibility_R8_sustainability":        "judgment-dependent",
}


def _log(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", flush=True)


def _now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(val: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(round(float(val)))
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return default


def _trim_text(text: str, max_chars: int = 12000) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def _detect_latest_pid() -> str:
    paths = sorted(REPORT_DIR.glob("*_final_report.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if paths:
        return paths[0].name.replace("_final_report.md", "")
    return ""


def build_system_prompt() -> str:
    return (
        "You are an investment proposal evaluator. Return only JSON.\n"
        "Score ONLY the feasibility dimension on the 8 sub-rubrics below. "
        "Calibration: these proposals have already passed initial screening. "
        "Scores of 1-2 should be reserved for a genuine critical gap or missing evidence. "
        "A score of 3 means common quality for a screened proposal. "
        "A score of 4 means meaningfully strong evidence. "
        "A score of 5 means exceptional and fully documented.\n\n"
        "Use only the provided report evidence. Do not invent facts.\n\n"
        "Sub-rubrics (score each 1-5):\n"
        "- feasibility_R1_budget_detail: Budget detail. "
        "1 = no budget described. "
        "3 = total budget stated without a breakdown. "
        "5 = itemised budget by category, with amounts and time period.\n"
        "- feasibility_R2_funding_secured: Funding secured. "
        "1 = no funding source described. "
        "3 = funding sources named but not committed. "
        "5 = committed funding documented (grant awarded, investment closed) with amounts.\n"
        "- feasibility_R3_infrastructure: Infrastructure and resource access. "
        "1 = no facilities, equipment, or data access described. "
        "3 = resources mentioned without confirmed access. "
        "5 = named facilities, equipment, data, or suppliers with confirmed access (ownership or agreement).\n"
        "- feasibility_R4_technical_readiness: Technical readiness. "
        "1 = concept only, no prototype or data. "
        "3 = prototype or preliminary data described without figures. "
        "5 = working prototype or validated pilot with stated maturity level and results.\n"
        "- feasibility_R5_risk_mitigation: Risk mitigation. "
        "1 = no risks identified. "
        "3 = risks listed without mitigation. "
        "5 = key risks identified with specific mitigation and contingency plans.\n"
        "- feasibility_R6_resource_timeline_fit: Resource-timeline fit. "
        "1 = timeline clearly unrealistic for the available resources. "
        "3 = timeline plausible but not argued. "
        "5 = timeline explicitly justified by resources, staffing, and dependencies.\n"
        "- feasibility_R7_financial_realism: Financial realism. "
        "1 = budget clearly insufficient for the stated scope. "
        "3 = budget plausible but no comparables given. "
        "5 = budget benchmarked against comparables, with runway to the next milestone.\n"
        "- feasibility_R8_sustainability: Operational sustainability. "
        "1 = no plan beyond the funded period. "
        "3 = sustainability asserted without a mechanism. "
        "5 = concrete mechanism to sustain operations (revenue, follow-on funding, institutional commitment) with evidence.\n\n"
        "Score each rubric INDEPENDENTLY. Do not let strong evidence for one rubric "
        "influence your score for another. If the report contains no evidence relevant "
        "to a specific rubric, that rubric must score 1 or 2 regardless of the overall "
        "proposal quality. R5-R8 are judgment-dependent but anchor each judgment strictly "
        "to signals within that rubric's own scope — do not carry over adjacent evidence.\n\n"
        "Output JSON with exactly these keys: "
        "feasibility_R1_budget_detail, feasibility_R2_funding_secured, feasibility_R3_infrastructure, feasibility_R4_technical_readiness, feasibility_R5_risk_mitigation, feasibility_R6_resource_timeline_fit, feasibility_R7_financial_realism, feasibility_R8_sustainability."
    )


def build_user_payload(pid: str, report_text: str, expert_json: Dict[str, Any]) -> Dict[str, Any]:
    overall = expert_json.get("overall_opinion", {}) or {}
    feasibility_dim = (expert_json.get("dimensions") or {}).get("feasibility") or {}
    return {
        "proposal_id": pid,
        "report_excerpt": report_text,
        "expert_feasibility_summary": feasibility_dim.get("summary") or "",
        "expert_feasibility_strengths": (feasibility_dim.get("strengths") or [])[:5],
        "expert_feasibility_concerns": (feasibility_dim.get("concerns") or [])[:5],
        "expert_overall_summary": overall.get("summary") or "",
    }


def call_llm(
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
        max_tokens=600,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )


def sanitize_rubric_scores(raw: Dict[str, Any]) -> Dict[str, int]:
    return {k: _safe_int(raw.get(k), 1, 5, 3) for k in RUBRIC_KEYS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score feasibility sub-rubrics via LLM.")
    ap.add_argument("--pid",         type=str, default="",          help="Proposal ID; latest report if omitted.")
    ap.add_argument("--provider",    type=str, default=PROVIDER,    help="LLM provider: openai, gemini, or deepseek.")
    ap.add_argument("--model",       type=str, default="",          help="Model name. Defaults to provider-specific env model.")
    ap.add_argument("--temperature", type=float, default=0.0,       help="Sampling temperature.")
    ap.add_argument("--out_dir",     type=str, default="",          help="Optional output directory override.")
    ap.add_argument("--dry_run",     action="store_true",           help="Write prompt only; skip LLM call.")
    ap.add_argument(
        "--use_llm",
        action="store_true",
        help="Required. Score via an LLM call.",
    )
    args = ap.parse_args()

    if not args.use_llm:
        ap.error("--use_llm is required.")

    provider = (args.provider or PROVIDER).strip().lower()
    model    = args.model.strip() or default_model(provider)
    pid      = args.pid.strip() or _detect_latest_pid()

    if not pid:
        raise RuntimeError("No proposal report found.")

    report_path = REPORT_DIR / f"{pid}_final_report.md"
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    expert_path = EXPERT_DIR / pid / "ai_expert_opinion.json"
    expert_json: Dict[str, Any] = {}
    if expert_path.exists():
        try:
            expert_json = json.loads(expert_path.read_text(encoding="utf-8"))
        except Exception:
            expert_json = {}

    out_dir = Path(args.out_dir) if args.out_dir else (OUT_DIR / pid)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_text   = _trim_text(report_path.read_text(encoding="utf-8"))
    system_prompt = build_system_prompt()
    user_payload  = build_user_payload(pid, report_text, expert_json)

    prompt_path = out_dir / "feasibility_rubric_scores.prompt.json"
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump({"system": system_prompt, "user": user_payload}, f, ensure_ascii=False, indent=2)

    if args.dry_run:
        _log("OK", f"Prompt written: {prompt_path}")
        return

    raw = call_llm(provider, model, system_prompt, user_payload, args.temperature)
    rubric_scores = sanitize_rubric_scores(raw)

    payload = {
        "meta": {
            "pid": pid,
            "generated_at": _now_str(),
            "provider": provider,
            "model": model,
            "temperature": float(args.temperature),
            "report_path": str(report_path),
            "expert_path": str(expert_path) if expert_path.exists() else "",
        },
        "feasibility_rubrics": {
            k: {
                "score": rubric_scores[k],
                "type":  RUBRIC_TYPE[k],
            }
            for k in RUBRIC_KEYS
        },
    }

    out_path = out_dir / "feasibility_rubric_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _log("OK", f"Feasibility rubric scores written: {out_path}")
    _log("OK", "Scores:")
    for k in RUBRIC_KEYS:
        entry = payload["feasibility_rubrics"][k]
        _log("  ", f"{k:<30} {entry['score']}  [{entry['type']}]")


if __name__ == "__main__":
    main()
