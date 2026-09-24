# -*- coding: utf-8 -*-
"""
Evaluation: groundedness of a final report against the source proposal text.

Each factual claim in the report is classified as:
  supported          - explicitly stated in the proposal text
  inferred           - reasonably implied by context but not explicitly stated
  unsupported        - not present in the proposal (hallucination risk)

Score summary:
  groundedness_score  = supported / total                       (strict)
  lenient_score       = (supported + 0.5 * inferred) / total   (lenient)
  hallucination_rate  = unsupported / total

Inputs:
  - proposal text: src/data/prepared/<pid>/full_text.txt
  - final report:  src/data/reports/<pid>_final_report.md

Outputs:
  - src/data/evaluations/runs/<run_id>/groundedness_result.json
"""

import json
import os
import uuid
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.llm_provider import active_provider, chat_json, default_model, provider_api_key

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
REPORT_DIR = DATA_DIR / "reports"
PREPARED_DIR = DATA_DIR / "prepared"
EVAL_DIR = DATA_DIR / "evaluations"
RUNS_DIR = EVAL_DIR / "runs"

load_dotenv()
PROVIDER = active_provider()
MODEL_NAME = os.getenv("GROUNDEDNESS_MODEL", "").strip() or default_model(PROVIDER)

# Truncation limits to control cost and context length
MAX_PROPOSAL_CHARS = 24_000
MAX_REPORT_CHARS = 20_000


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [TRUNCATED] ..."


def _extract_evaluation_sections(report_md: str) -> str:
    """Return only Sections 0 and 1 of the report (before the Q&A evidence block).

    The final report structure is:
      Section 0: Executive Summary  (scores, verdict, overall opinion)
      Section 1: AI Expert Review   (dimension summaries, key risks/strengths)
      Section 2: Dimension Q&A and Scoring Evidence  ← excluded (very long)

    Grounding claims against only Sections 0+1 avoids polluting extraction with
    question-level evidence text that isn't actually part of the proposal.
    """
    for marker in ("\n## 2.", "## 2."):
        idx = report_md.find(marker)
        if idx != -1:
            return report_md[:idx].strip()
    return report_md


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_latest_pid() -> str:
    if not REPORT_DIR.exists():
        return ""
    cands = [(p.name.replace("_final_report.md", ""), p.stat().st_mtime)
              for p in REPORT_DIR.glob("*_final_report.md")]
    if not cands:
        return ""
    return max(cands, key=lambda x: x[1])[0]


def extract_claims(report_md: str, provider: str, model_name: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a claim extraction evaluator.\n\n"
        "Your task: extract factual claims that the PROPOSAL ITSELF explicitly makes, "
        "as reported in the evaluation summary.\n\n"
        "ONLY extract claims in this category:\n"
        "- Named individuals, roles, organisations, or stated team credentials\n"
        "- Stated technologies, products, or methodologies the proposal describes\n"
        "- Specific financial figures, market sizes, revenue targets, or timelines "
        "that the proposal cites\n"
        "- Named partnerships, customers, or regulatory approvals the proposal claims\n"
        "- Concrete milestones or deliverables with specific dates or metrics\n\n"
        "DO NOT extract any of the following:\n"
        "- Statements about the AI evaluation itself: the overall score, ranking, "
        "verdict (GO/HOLD/NO-GO), confidence value, or any numeric output of the "
        "evaluation system — these can never be verified against the proposal.\n"
        "- Evaluator opinion statements framed as absences or deficiencies: "
        "\"the proposal lacks X\", \"does not provide Y\", \"fails to specify Z\", "
        "\"no evidence of\", \"unclear whether\", \"not addressed\" — these are "
        "evaluator judgements, not proposal facts.\n"
        "- Generic evaluator commentary, recommendations, or questions.\n\n"
        "Additional rules:\n"
        "- Split compound statements into separate atomic claims.\n"
        "- Do not verify claims here — just extract them.\n"
        "- Return only valid JSON.\n"
    )
    user_prompt = (
        "Extract factual claims that the PROPOSAL explicitly states, as summarised "
        "in the evaluation report below.\n\n"
        "Return exactly this JSON:\n"
        "{\n"
        "  \"claims\": [\n"
        "    {\"claim_id\": 1, \"claim\": \"string\"}\n"
        "  ]\n"
        "}\n\n"
        f"Report:\n\"\"\"\n{report_md}\n\"\"\"\n"
    )
    try:
        return chat_json(
            provider=provider,
            model=model_name,
            temperature=0.0,
            max_tokens=1800,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            ],
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Model did not return valid JSON for claim extraction.") from exc


def verify_claims(claims_json: Dict[str, Any], proposal_text: str, provider: str, model_name: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a groundedness evaluator.\n\n"
        "Verify whether each claim is grounded in the provided business proposal text.\n\n"
        "Use ONLY the proposal text — no outside knowledge.\n\n"
        "Classify each claim as exactly one of:\n"
        "  \"supported\"   — the claim is explicitly and clearly stated in the proposal.\n"
        "  \"inferred\"    — the claim is a reasonable inference from the proposal context "
        "but is not explicitly stated.\n"
        "  \"unsupported\" — the claim contradicts or is entirely absent from the proposal "
        "(hallucination risk).\n\n"
        "Rules:\n"
        "- If the proposal explicitly states the fact → supported.\n"
        "- If the proposal strongly implies it but does not say it directly → inferred.\n"
        "- If the proposal provides no basis for the claim → unsupported.\n"
        "- For supported/inferred claims: provide the exact supporting quote from the proposal.\n"
        "- For unsupported claims: set supporting_quote to \"\".\n"
        "- Return only valid JSON.\n"
    )
    user_prompt = (
        "Business proposal text:\n"
        f"\"\"\"\n{proposal_text}\n\"\"\"\n\n"
        "Claims to verify:\n"
        f"{json.dumps(claims_json, ensure_ascii=False, indent=2)}\n\n"
        "Return exactly this JSON:\n"
        "{\n"
        "  \"claims\": [\n"
        "    {\n"
        "      \"claim_id\": 1,\n"
        "      \"claim\": \"string\",\n"
        "      \"classification\": \"supported | inferred | unsupported\",\n"
        "      \"supporting_quote\": \"exact quote from proposal, or empty string\",\n"
        "      \"reason\": \"brief explanation\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )
    try:
        result = chat_json(
            provider=provider,
            model=model_name,
            temperature=0.0,
            max_tokens=3000,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            ],
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Model did not return valid JSON for verification.") from exc

    # Normalise classification field and add backward-compat `supported` boolean
    valid_classes = {"supported", "inferred", "unsupported"}
    for claim in result.get("claims", []):
        cls = str(claim.get("classification", "")).strip().lower()
        if cls not in valid_classes:
            cls = "unsupported"
        claim["classification"] = cls
        claim["supported"] = cls in ("supported", "inferred")  # lenient backward compat

    return result


def calculate_groundedness(verified: Dict[str, Any]) -> Dict[str, Any]:
    claims = verified.get("claims", [])
    total = len(claims)
    if total == 0:
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "inferred_claims": 0,
            "unsupported_claims": 0,
            "groundedness_score": None,
            "lenient_score": None,
            "hallucination_rate": None,
        }

    n_supported = sum(1 for c in claims if c.get("classification") == "supported")
    n_inferred = sum(1 for c in claims if c.get("classification") == "inferred")
    n_unsupported = sum(1 for c in claims if c.get("classification") == "unsupported")

    groundedness = round(n_supported / total, 4)
    lenient = round((n_supported + 0.5 * n_inferred) / total, 4)
    hallucination_rate = round(n_unsupported / total, 4)

    return {
        "total_claims": total,
        "supported_claims": n_supported,
        "inferred_claims": n_inferred,
        "unsupported_claims": n_unsupported,
        "groundedness_score": groundedness,
        "lenient_score": lenient,
        "hallucination_rate": hallucination_rate,
    }


def evaluate_groundedness(proposal_text: str, report_md: str, provider: str, model_name: str) -> Dict[str, Any]:
    if not provider_api_key(provider):
        raise RuntimeError(f"{provider.upper()} API key is missing. Configure it in .env.")

    proposal_text = _truncate(proposal_text, MAX_PROPOSAL_CHARS)
    report_md = _truncate(_extract_evaluation_sections(report_md), MAX_REPORT_CHARS)

    _log("INFO", f"Extracting claims from report with provider={provider} model={model_name}...")
    extracted = extract_claims(report_md, provider, model_name)
    n_claims = len(extracted.get("claims", []))
    _log("INFO", f"Extracted {n_claims} claims. Verifying against proposal...")

    verified = verify_claims(extracted, proposal_text, provider, model_name)
    summary = calculate_groundedness(verified)

    _log("INFO", (
        f"Groundedness: {summary['groundedness_score']} strict, "
        f"{summary['lenient_score']} lenient, "
        f"hallucination_rate={summary['hallucination_rate']}"
    ))

    return {
        "score_summary": summary,
        "extracted_claims": extracted,
        "verified_claims": verified,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate groundedness of a proposal report.")
    ap.add_argument("--pid", type=str, default="", help="Proposal ID; latest report if omitted.")
    ap.add_argument("--out_dir", type=str, default="", help="Optional output directory.")
    ap.add_argument("--provider", type=str, default="", help="LLM provider, e.g. openai or gemini.")
    ap.add_argument("--model", type=str, default="", help="Model override. Defaults to provider-specific env model.")
    args = ap.parse_args()
    provider = args.provider.strip().lower() or PROVIDER
    model_name = args.model.strip() or default_model(provider)

    pid = args.pid.strip() or detect_latest_pid()
    if not pid:
        raise RuntimeError("No report found to evaluate groundedness.")

    report_path = REPORT_DIR / f"{pid}_final_report.md"
    proposal_path = PREPARED_DIR / pid / "full_text.txt"
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    if not proposal_path.exists():
        raise FileNotFoundError(f"Proposal text not found: {proposal_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_text = read_text(report_path)
    proposal_text = read_text(proposal_path)

    result = evaluate_groundedness(proposal_text, report_text, provider, model_name)
    # Report actual chars sent to the LLM (post section-extraction, post truncation)
    report_chars_used = len(_truncate(_extract_evaluation_sections(report_text), MAX_REPORT_CHARS))
    payload = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "provider": provider,
            "model": model_name,
            "report_path": str(report_path),
            "proposal_path": str(proposal_path),
            "proposal_chars_used": min(len(proposal_text), MAX_PROPOSAL_CHARS),
            "report_chars_used": report_chars_used,
        },
        **result,
    }

    out_path = out_dir / "groundedness_result.json"
    write_json(out_path, payload)
    _log("OK", f"groundedness result written: {out_path}")


if __name__ == "__main__":
    main()
