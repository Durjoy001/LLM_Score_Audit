# -*- coding: utf-8 -*-
"""
Objectives sub-rubric scoring tool.

Scores a proposal on 8 granular objectives sub-rubrics (R1-R8) using an LLM.
Results are stored separately from the main evaluation pipeline and do not
affect any existing scores or verdicts.

This tool is used to compare human investor judgment vs AI on the objectives
dimension — specifically to show where AI and humans agree (document-verifiable
rubrics R1-R4) vs diverge (judgment-dependent rubrics R5-R8).

Outputs:
  - src/data/objectives_rubric_scores/<pid>/objectives_rubric_scores.json

Usage:
    python src/tools/generate_objectives_rubric_scores.py --pid <proposal_id> --use_llm
    python src/tools/generate_objectives_rubric_scores.py --use_llm   # latest proposal
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
OUT_DIR     = DATA_DIR / "objectives_rubric_scores"
PROPOSALS_DIR = DATA_DIR / "proposals"

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

RUBRIC_KEYS = [
    "objectives_R1_problem_clarity",
    "objectives_R2_market_size",
    "objectives_R3_buyer_pathway",
    "objectives_R4_unmet_need_evidence",
    "objectives_R5_why_now",
    "objectives_R6_competitive_context",
    "objectives_R7_problem_solution_fit",
    "objectives_R8_addressability",
]

RUBRIC_TYPE = {
    "objectives_R1_problem_clarity":     "document-verifiable",
    "objectives_R2_market_size":         "document-verifiable",
    "objectives_R3_buyer_pathway":       "document-verifiable",
    "objectives_R4_unmet_need_evidence": "document-verifiable",
    "objectives_R5_why_now":             "judgment-dependent",
    "objectives_R6_competitive_context": "judgment-dependent",
    "objectives_R7_problem_solution_fit":"judgment-dependent",
    "objectives_R8_addressability":      "judgment-dependent",
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
        "Score ONLY the objectives dimension on the 8 sub-rubrics below. "
        "Calibration: these proposals have already passed initial screening. "
        "Scores of 1-2 should be reserved for a genuine critical gap or missing evidence. "
        "A score of 3 means common quality for a screened proposal. "
        "A score of 4 means meaningfully strong evidence. "
        "A score of 5 means exceptional and fully documented.\n\n"
        "Use only the provided report evidence. Do not invent facts.\n\n"
        "Sub-rubrics (score each 1-5):\n"
        "- objectives_R1_problem_clarity: Problem clarity and specificity. "
        "1 = problem is vague or not stated. "
        "3 = problem described but without specific context or data. "
        "5 = problem precisely defined with specific patient/customer context and quantified evidence.\n"
        "- objectives_R2_market_size: Market size quantification. "
        "1 = no market size stated. "
        "3 = market size mentioned but unverified or broad estimate only. "
        "5 = specific, sourced market size figure with addressable segment breakdown.\n"
        "- objectives_R3_buyer_pathway: Buyer/payer pathway definition. "
        "1 = no buyer or payer identified. "
        "3 = buyer type named but no pathway or engagement described. "
        "5 = named buyer/payer with documented pathway, LOI, or pilot evidence.\n"
        "- objectives_R4_unmet_need_evidence: Unmet need evidence and demand validation. "
        "1 = unmet need asserted with no supporting evidence. "
        "3 = clinical/market gap described with partial references. "
        "5 = validated unmet need with published data, regulatory evidence, or patient outcome data.\n"
        "- objectives_R5_why_now: Why-now timing rationale. "
        "1 = no timing rationale given. "
        "3 = why-now mentioned but based only on proposal claims. "
        "5 = strong external catalyst cited (regulation change, technology inflection, validated demand spike).\n"
        "- objectives_R6_competitive_context: Competitive context and positioning. "
        "1 = no mention of existing solutions or competitors. "
        "3 = competitors acknowledged but differentiation is vague. "
        "5 = clear positioning vs named alternatives with specific differentiating evidence.\n"
        "- objectives_R7_problem_solution_fit: Problem-solution fit. "
        "1 = solution does not clearly address the stated problem. "
        "3 = fit is plausible but not explicitly argued. "
        "5 = tight and explicit problem-solution fit with mechanism explained.\n"
        "- objectives_R8_addressability: Market addressability and segment definition. "
        "1 = target segment is undefined or too broad to serve. "
        "3 = segment defined but access strategy is missing. "
        "5 = specific addressable segment with defined entry point and realistic reach estimate.\n\n"
        "Score each rubric INDEPENDENTLY. Do not let strong evidence for one rubric "
        "influence your score for another. If the report contains no evidence relevant "
        "to a specific rubric, that rubric must score 1 or 2 regardless of overall proposal quality. "
        "R5-R8 are judgment-dependent but anchor each judgment strictly to signals within "
        "that rubric's own scope — do not carry over adjacent evidence.\n\n"
        "Output JSON with exactly these keys: "
        "objectives_R1_problem_clarity, objectives_R2_market_size, objectives_R3_buyer_pathway, "
        "objectives_R4_unmet_need_evidence, objectives_R5_why_now, objectives_R6_competitive_context, "
        "objectives_R7_problem_solution_fit, objectives_R8_addressability."
    )


def build_user_payload(pid: str, report_text: str, expert_json: Dict[str, Any]) -> Dict[str, Any]:
    overall = expert_json.get("overall_opinion", {}) or {}
    objectives_dim = (expert_json.get("dimensions") or {}).get("objectives") or {}
    return {
        "proposal_id": pid,
        "report_excerpt": report_text,
        "expert_objectives_summary": objectives_dim.get("summary") or "",
        "expert_objectives_strengths": (objectives_dim.get("strengths") or [])[:5],
        "expert_objectives_concerns": (objectives_dim.get("concerns") or [])[:5],
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
    ap = argparse.ArgumentParser(description="Score objectives sub-rubrics via LLM.")
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

    prompt_path = out_dir / "objectives_rubric_scores.prompt.json"
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
        "objectives_rubrics": {
            k: {
                "score": rubric_scores[k],
                "type":  RUBRIC_TYPE[k],
            }
            for k in RUBRIC_KEYS
        },
    }

    out_path = out_dir / "objectives_rubric_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _log("OK", f"Objectives rubric scores written: {out_path}")
    _log("OK", "Scores:")
    for k in RUBRIC_KEYS:
        entry = payload["objectives_rubrics"][k]
        _log("  ", f"{k:<40} {entry['score']}  [{entry['type']}]")


if __name__ == "__main__":
    main()
