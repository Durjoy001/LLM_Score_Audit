# -*- coding: utf-8 -*-
"""
Team sub-rubric scoring tool.

Scores a proposal on 8 granular team sub-rubrics (R1-R8) using an LLM.
Results are stored separately from the main evaluation pipeline and do not
affect any existing scores or verdicts.

This tool is used to compare human investor judgment vs AI on the team
dimension — specifically to show where AI and humans agree (document-verifiable
rubrics R1-R4) vs diverge (judgment-dependent rubrics R5-R8).

Outputs:
  - src/data/team_rubric_scores/<pid>/team_rubric_scores.json

Usage:
    python src/tools/generate_team_rubric_scores.py --pid <proposal_id> --use_llm
    python src/tools/generate_team_rubric_scores.py --use_llm   # latest proposal
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
OUT_DIR     = DATA_DIR / "team_rubric_scores"
PROPOSALS_DIR = DATA_DIR / "proposals"

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

RUBRIC_KEYS = [
    "team_R1_composition",
    "team_R2_credentials",
    "team_R3_track_record",
    "team_R4_governance",
    "team_R5_capacity",
    "team_R6_complementarity",
    "team_R7_cohesion",
    "team_R8_key_person_risk",
]

RUBRIC_TYPE = {
    "team_R1_composition":     "document-verifiable",
    "team_R2_credentials":     "document-verifiable",
    "team_R3_track_record":    "document-verifiable",
    "team_R4_governance":      "document-verifiable",
    "team_R5_capacity":        "judgment-dependent",
    "team_R6_complementarity": "judgment-dependent",
    "team_R7_cohesion":        "judgment-dependent",
    "team_R8_key_person_risk": "judgment-dependent",
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
        "Score ONLY the team dimension on the 8 sub-rubrics below. "
        "Calibration: these proposals have already passed initial screening. "
        "Scores of 1-2 should be reserved for a genuine critical gap or missing evidence. "
        "A score of 3 means common quality for a screened proposal. "
        "A score of 4 means meaningfully strong evidence. "
        "A score of 5 means exceptional and fully documented.\n\n"
        "Use only the provided report evidence. Do not invent facts.\n\n"
        "Sub-rubrics (score each 1-5):\n"
        "- team_R1_composition: Team composition. "
        "1 = only a single lead is named or roles are unspecified. "
        "3 = core roles named but key technical, domain, or commercial functions are unfilled. "
        "5 = all critical roles filled by named individuals with stated responsibilities.\n"
        "- team_R2_credentials: Domain credentials. "
        "1 = no relevant qualifications stated. "
        "3 = credentials stated only generically (degrees, titles). "
        "5 = specific credentials in the project's domain: named institutions, publications, patents, or prior positions.\n"
        "- team_R3_track_record: Prior execution track record. "
        "1 = no prior projects or ventures cited. "
        "3 = prior roles or projects mentioned without outcomes. "
        "5 = named prior ventures or projects with documented outcomes (launch, approval, exit, funded grant).\n"
        "- team_R4_governance: Governance and advisors. "
        "1 = no governance structure, board, or advisors described. "
        "3 = advisors or board mentioned without names or roles. "
        "5 = named board or advisors, defined decision-making structure, and formal agreements (collaboration, equity).\n"
        "- team_R5_capacity: Execution capacity. "
        "1 = team size or time commitment clearly insufficient for the project scope. "
        "3 = capacity plausible but commitment (full-time vs part-time) is unclear. "
        "5 = dedicated team sized to the project scope, with a hiring plan for gaps.\n"
        "- team_R6_complementarity: Skill complementarity. "
        "1 = skills concentrated in one area with a critical gap. "
        "3 = skills mostly complementary with one notable gap. "
        "5 = balanced technical, domain, and commercial skills with no critical gap.\n"
        "- team_R7_cohesion: Team cohesion. "
        "1 = no evidence the members have worked together. "
        "3 = members linked by a shared institution but no joint delivery. "
        "5 = core members have jointly delivered prior projects with documented outcomes.\n"
        "- team_R8_key_person_risk: Key-person risk. "
        "1 = project depends on one individual with no backup. "
        "3 = some redundancy, but key functions rest on one person. "
        "5 = responsibilities distributed, with backup and retention mechanisms.\n\n"
        "Score each rubric INDEPENDENTLY. Do not let strong evidence for one rubric "
        "influence your score for another. If the report contains no evidence relevant "
        "to a specific rubric, that rubric must score 1 or 2 regardless of the overall "
        "proposal quality. R5-R8 are judgment-dependent but anchor each judgment strictly "
        "to signals within that rubric's own scope — do not carry over adjacent evidence.\n\n"
        "Output JSON with exactly these keys: "
        "team_R1_composition, team_R2_credentials, team_R3_track_record, team_R4_governance, team_R5_capacity, team_R6_complementarity, team_R7_cohesion, team_R8_key_person_risk."
    )


def build_user_payload(pid: str, report_text: str, expert_json: Dict[str, Any]) -> Dict[str, Any]:
    overall = expert_json.get("overall_opinion", {}) or {}
    team_dim = (expert_json.get("dimensions") or {}).get("team") or {}
    return {
        "proposal_id": pid,
        "report_excerpt": report_text,
        "expert_team_summary": team_dim.get("summary") or "",
        "expert_team_strengths": (team_dim.get("strengths") or [])[:5],
        "expert_team_concerns": (team_dim.get("concerns") or [])[:5],
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
    ap = argparse.ArgumentParser(description="Score team sub-rubrics via LLM.")
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

    prompt_path = out_dir / "team_rubric_scores.prompt.json"
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
        "team_rubrics": {
            k: {
                "score": rubric_scores[k],
                "type":  RUBRIC_TYPE[k],
            }
            for k in RUBRIC_KEYS
        },
    }

    out_path = out_dir / "team_rubric_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _log("OK", f"Team rubric scores written: {out_path}")
    _log("OK", "Scores:")
    for k in RUBRIC_KEYS:
        entry = payload["team_rubrics"][k]
        _log("  ", f"{k:<30} {entry['score']}  [{entry['type']}]")


if __name__ == "__main__":
    main()
