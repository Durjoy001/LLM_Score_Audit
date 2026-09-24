# -*- coding: utf-8 -*-
"""
Stage 7: Final Report Generator
(executive summary + expert review + compact Q&A, based on final_payload)
-----------------------------------------------------------------------
Purpose:
  1. Read the final Stage 5 and Stage 6 artifacts:
       - refined_answers/<pid>/postproc/final_payload.json
       - expert_reports/<pid>/ai_expert_opinion.json
       - expert_reports/<pid>/ai_expert_opinion.md
  2. Generate one integrated Markdown report:
       - top-level report title, generation time, and system note
       - executive summary with verdict, score/confidence, top strengths, top risks, and next actions
       - AI expert review from ai_expert_opinion.md, with headings shifted down by one level
       - dimension-level Q&A traceability based on final_payload

Usage:
  cd to the project root that contains src/
  python -m src.tools.generate_final_report
  python -m src.tools.generate_final_report --pid example_proposal_id
"""

import json
import os
import re
import time
import argparse
from pathlib import Path
from datetime import datetime

# ========= Paths =========
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
REFINED_ROOT = DATA_DIR / "refined_answers"
EXPERT_ROOT = DATA_DIR / "expert_reports"
REPORT_ROOT = DATA_DIR / "reports"
PROGRESS_FILE = DATA_DIR / "step_progress.json"


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass

DIM_ORDER = ["team", "objectives", "strategy", "innovation", "feasibility"]
DIM_LABELS = {
    "team": "Team and governance",
    "objectives": "Project objectives",
    "strategy": "Implementation path and strategy",
    "innovation": "Technology and product innovation",
    "feasibility": "Resources and feasibility",
}

# Maximum selected Q&A rows shown per dimension, ranked by confidence plus alignment.
# Default to 10 so the report keeps full traceability for normal 6-10 question sets.
MAX_QA_PER_DIM = int(os.getenv("STAGE7_MAX_QA_PER_DIM", "10"))


# ========= Utilities =========
def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean_report_text(text: str) -> str:
    text = str(text or "")
    replacements = [
        ("which with", "which align with"),
        ("that with", "that align with"),
        ("validation of with", "validation against"),
        ("indicating with", "indicating"),
        ("for project.", "for project execution."),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.replace("and stakeholders.", "and stakeholder coordination.")
    text = text.replace("stakeholder .", "stakeholders.")
    return text


def detect_latest_pid() -> str:
    """Return the latest pid with final_payload.json and the Stage 6 expert Markdown."""
    if not REFINED_ROOT.exists():
        return ""
    cands = []
    for d in REFINED_ROOT.iterdir():
        if not d.is_dir():
            continue
        postproc_dir = d / "postproc"
        fp_path = postproc_dir / "final_payload.json"
        expert_md_path = EXPERT_ROOT / d.name / "ai_expert_opinion.md"
        if fp_path.exists() and expert_md_path.exists():
            cands.append((d.name, fp_path.stat().st_mtime))
    if not cands:
        return ""
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0]


def adjust_expert_markdown(md_text: str) -> str:
    """
    Shift headings from ai_expert_opinion.md down by two levels:
      #   -> ###
      ##  -> ####
      ### -> #####
    This keeps the imported expert review nested under Section 1.
    """
    lines = md_text.splitlines()
    adjusted = []
    for line in lines:
        if line.startswith("### "):
            adjusted.append("##### " + line[4:])
        elif line.startswith("## "):
            adjusted.append("#### " + line[3:])
        elif line.startswith("# "):
            adjusted.append("### " + line[2:])
        else:
            adjusted.append(line)
    return "\n".join(adjusted)


def _fmt_float(v, ndigits=1, default="0.0"):
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return default


def build_diligence_focus(expert_json: dict) -> list[str]:
    """
    Create a concise scale-up/investment diligence checklist from expert gaps.
    This is deterministic and avoids adding project facts not present in the report.
    """
    dims = expert_json.get("dimensions", {}) or {}
    corpus_parts = []
    for block in dims.values():
        if not isinstance(block, dict):
            continue
        for key in ("concerns", "recommendations"):
            for item in block.get(key) or []:
                corpus_parts.append(str(item))
    corpus = " ".join(corpus_parts).lower()

    checks = [
        (
            ("problem", "pain", "need", "beneficiary", "use case", "user", "demand"),
            "Problem and demand proof: add customer/user/beneficiary discovery, pain-point evidence, willingness-to-pay or adoption signals, and clear target-segment definition.",
        ),
        (
            ("performance", "benchmark", "metric", "efficiency", "prototype", "validation", "test", "quality", "service"),
            "Product/service proof: provide benchmark tests, prototype or service validation data, quality metrics, and acceptance thresholds against current alternatives.",
        ),
        (
            ("customer", "market", "adoption", "pilot", "loi", "feedback", "commercial", "channel", "procurement", "payer", "buyer"),
            "Route-to-market proof: add buyer/payer/funder evidence, pilot/LOI status, channel or procurement path, pricing assumptions, and adoption risks.",
        ),
        (
            ("budget", "funding", "financial", "cost", "unit", "cash", "fundraising", "margin", "pricing", "revenue"),
            "Financial proof: show use-of-funds, unit economics or cost model, pricing/margin logic, committed funding, and follow-on financing plan.",
        ),
        (
            ("implementation", "timeline", "milestone", "owner", "resource", "capacity", "production", "staff", "supply", "delivery", "operation"),
            "Operating proof: include owner-level milestones, staffing/capacity assumptions, dependencies, delivery timeline, supply or partner constraints, and scale-up bottlenecks.",
        ),
        (
            ("risk", "mitigation", "regulatory", "compliance", "governance", "legal", "security", "privacy", "permit"),
            "Risk and governance proof: add a risk register with mitigations, decision owners, legal/compliance/security path, and escalation process.",
        ),
        (
            ("patent", "ip", "competitive", "competitor", "defensibility", "brand", "data", "network", "moat", "alternative"),
            "Defensibility proof: include competitor/substitute comparison, IP/brand/data/network advantages when relevant, freedom-to-operate risks, and moat strategy.",
        ),
        (
            ("impact", "beneficiary", "outcome", "public", "social", "environment", "training", "education"),
            "Impact proof: define the theory of change, measurable outcomes, beneficiary evidence, implementation partners, and funding durability.",
        ),
    ]

    out = []
    for keywords, text in checks:
        if any(k in corpus for k in keywords):
            out.append(text)
    return out[:5]


# ========= Executive summary =========
def build_executive_summary(expert_json: dict) -> str:
    """
    Build the executive summary Markdown from ai_expert_opinion.json.
    Return an empty string if the JSON is unavailable.
    """
    if not expert_json:
        return ""

    overall = expert_json.get("overall_opinion", {}) or {}
    score = overall.get("overall_score_echo", 0.0)
    conf = overall.get("confidence_echo", 0.0)
    verdict = (overall.get("verdict") or "").strip()
    summary = (overall.get("summary") or "").strip()
    key_strengths = overall.get("key_strengths") or []
    key_risks = overall.get("key_risks") or []
    recs = overall.get("recommendations") or []
    basis = overall.get("basis") or []

    verdict_note = ""
    if verdict == "GO":
        verdict_note = "(advance if risks remain controlled)"
    elif verdict == "HOLD":
        verdict_note = "(supplement materials before a final decision)"
    elif verdict == "NO-GO":
        verdict_note = "(do not proceed under the current evidence set)"

    # One-line decision basis: prefer basis[0], then fall back to the summary.
    brief_reason = ""
    if basis:
        brief_reason = clean_report_text(basis[0])
    elif summary:
        brief_reason = clean_report_text(summary)

    lines = []
    lines.append("## 0. Executive Summary")
    lines.append("")
    if verdict:
        lines.append(f"- **Overall verdict:** {verdict} {verdict_note}".strip())
    else:
        lines.append("- **Overall verdict:** no clear verdict available")
    if brief_reason:
        lines.append(f"- **Brief decision basis:** {brief_reason}")
    lines.append(f"- **Overall score:** {_fmt_float(score, 3)} (0-1 range)")
    lines.append(f"- **Confidence:** {_fmt_float(conf, 3)}")
    lines.append("")
    lines.append("> **Score band guide for non-technical reviewers:**")
    lines.append("> - >= 0.62: conditions look relatively strong; proceed only with controlled risk.")
    lines.append("> - 0.45-0.62: evidence is incomplete or mixed; supplement materials before deciding.")
    lines.append("> - < 0.45: key dimensions have clear weaknesses or high uncertainty; approval is generally not recommended.")
    lines.append("")

    if key_strengths:
        lines.append("**Top Project Strengths (up to 3)**")
        for s in key_strengths[:3]:
            lines.append(f"- {clean_report_text(s)}")
        lines.append("")
    if key_risks:
        lines.append("**Top Risks / Gaps (up to 3)**")
        for r in key_risks[:3]:
            lines.append(f"- {clean_report_text(r)}")
        lines.append("")

    if recs:
        lines.append("**Key Next Actions / Materials to Add (3-5 selected items)**")
        for r in recs[:5]:
            lines.append(f"- {clean_report_text(r)}")
        lines.append("")

    diligence_focus = build_diligence_focus(expert_json)
    if diligence_focus:
        lines.append("**Scale-Up Diligence Focus**")
        for item in diligence_focus:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)


# ========= Dimension Q&A section =========
def build_qa_section_from_final_payload(final_payload: dict) -> tuple[str, dict]:
    """
    Build the dimension Q&A traceability section from postproc/final_payload.json.
    Each dimension shows:
    - aggregate score
    - rationale summary
    - dimension-level general insights
    - selected Q&A rows, capped at MAX_QA_PER_DIM
    """
    dims = final_payload.get("dimensions", {}) or {}
    section_stats = {"dimensions": {}, "total_qas_available": 0, "total_qas_shown": 0}

    lines = []
    lines.append("## 2. Dimension Q&A and Scoring Evidence")
    lines.append("")
    lines.append("> This section is generated from the Stage 5 selected Q&A results and supports traceability for the expert review.")
    lines.append("> For a quick read, focus on Section 0 and Section 1; this section is mainly for technical review and audit.")
    lines.append("")

    for dim in DIM_ORDER:
        block = dims.get(dim) or {}
        qas = block.get("qas") or []
        if not qas:
            section_stats["dimensions"][dim] = {
                "qas_available": 0,
                "qas_shown": 0,
                "rationales": 0,
                "general_insights": 0,
            }
            continue

        # Rank by confidence plus alignment, then cap to the top rows.
        def _qa_key(qa_item):
            try:
                conf = float(qa_item.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            try:
                align = float(qa_item.get("alignment") or 0.0)
            except Exception:
                align = 0.0
            return (conf + align)

        qas_sorted = sorted(qas, key=_qa_key, reverse=True)
        qas_sorted = qas_sorted[:MAX_QA_PER_DIM]

        label = DIM_LABELS.get(dim, dim)
        score = block.get("score", 0.0)
        rationales = block.get("rationales") or []
        gi_dim = block.get("general_insights") or []
        section_stats["dimensions"][dim] = {
            "qas_available": len(qas),
            "qas_shown": len(qas_sorted),
            "rationales": len(rationales),
            "general_insights": len(gi_dim),
        }
        section_stats["total_qas_available"] += len(qas)
        section_stats["total_qas_shown"] += len(qas_sorted)

        lines.append(f"### Dimension: {label} ({dim}) - Aggregate Score: {_fmt_float(score, 1)} / 100")
        lines.append("")

        if rationales:
            lines.append("**Scoring Rationale Summary**")
            for r in rationales:
                r = str(r).strip()
                if r:
                    lines.append(f"- {r}")
            lines.append("")

        if gi_dim:
            lines.append("**Industry Context Points (reference only, not proof of project achievement)**")
            for g in gi_dim[:8]:
                g = str(g).strip()
                if g:
                    lines.append(f"- {g}")
            lines.append("")

        for idx, qa in enumerate(qas_sorted, start=1):
            q_text = (qa.get("q") or "").strip()
            ans = (qa.get("answer") or "").strip()
            provider = (qa.get("provider") or "").strip()
            model = (qa.get("model") or "").strip()
            conf = qa.get("confidence", 0.0)
            align = qa.get("alignment", 0.0)
            drift = qa.get("dimension_drift", 0.0)
            claims = qa.get("claims") or []
            evids = qa.get("evidence_hints") or []

            lines.append("")
            lines.append(f"#### Q{idx}. {q_text}")
            lines.append("")
            meta_parts = []
            if provider:
                meta_parts.append(f"provider: {provider}")
            if model:
                meta_parts.append(model)
            meta_parts.append(f"confidence: {_fmt_float(conf, 2)}")
            meta_parts.append(f"alignment: {_fmt_float(align, 2)}")
            meta_parts.append(f"dimension_drift: {_fmt_float(drift, 2)}")

            lines.append("_" + " | ".join(meta_parts) + "_")
            lines.append("")

            if claims:
                lines.append("**Key Claims**")
                for c in claims:
                    c = str(c).strip()
                    if c:
                        lines.append(f"- {c}")
                lines.append("")

            lines.append("**Selected Answer**")
            lines.append("")
            if ans:
                lines.append(ans)
            else:
                lines.append("_No valid answer._")
            lines.append("")

            if evids:
                lines.append("**Evidence Hints**")
                for e in evids:
                    e = str(e).strip()
                    if e:
                        lines.append(f"- {e}")
                lines.append("")
    return "\n".join(lines), section_stats


# ========= Main flow =========
def main():
    stage_start = time.perf_counter()
    ap = argparse.ArgumentParser(description="Generate final integrated markdown report (executive summary + expert opinion + Q&A, v3).")
    ap.add_argument("--pid", type=str, default="", help="Proposal ID. If omitted, the latest available project is used.")
    args = ap.parse_args()

    _log("STAGE_7", "Starting Stage 7: final report compilation")
    _log("STAGE_7", "Purpose: compile executive summary, expert opinion, and selected Q&A into one Markdown report.")

    pid = args.pid.strip() or detect_latest_pid()
    if not pid:
        raise RuntimeError(
            "No available project detected. Expected refined_answers/<pid>/postproc/final_payload.json "
            "and expert_reports/<pid>/ai_expert_opinion.md."
        )

    refined_dir = REFINED_ROOT / pid
    postproc_dir = refined_dir / "postproc"
    fp_path = postproc_dir / "final_payload.json"

    expert_dir = EXPERT_ROOT / pid
    expert_md_path = expert_dir / "ai_expert_opinion.md"
    expert_json_path = expert_dir / "ai_expert_opinion.json"

    if not fp_path.exists():
        raise FileNotFoundError(f"final_payload.json not found: {fp_path}")
    if not expert_md_path.exists():
        raise FileNotFoundError(f"expert review Markdown not found: {expert_md_path}")

    _log("INPUT", f"proposal_id={pid}")
    _log("INPUT", f"final_payload_path={fp_path} size_bytes={fp_path.stat().st_size}")
    _log("INPUT", f"expert_markdown_path={expert_md_path} size_bytes={expert_md_path.stat().st_size}")

    # Missing ai_expert_opinion.json is allowed; the report will skip the executive summary.
    expert_json = {}
    if expert_json_path.exists():
        _log("INPUT", f"expert_json_path={expert_json_path} size_bytes={expert_json_path.stat().st_size}")
        expert_json = load_json(expert_json_path)
    else:
        _log("WARN", f"expert_json_path missing; executive summary will be skipped. path={expert_json_path}")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_ROOT / f"{pid}_final_report.md"
    audit_path = REPORT_ROOT / f"{pid}_stage7_final_report_audit.json"

    read_start = time.perf_counter()
    final_payload = load_json(fp_path)
    expert_md_raw = expert_md_path.read_text(encoding="utf-8")
    _log("TIMING", f"read_inputs elapsed_sec={time.perf_counter() - read_start:.3f}")

    dims = final_payload.get("dimensions", {}) or {}
    for dim in DIM_ORDER:
        block = dims.get(dim, {}) or {}
        _log(
            "DIM_INPUT",
            f"{dim}: qas={len(block.get('qas') or [])} "
            f"rationales={len(block.get('rationales') or [])} "
            f"general_insights={len(block.get('general_insights') or [])} "
            f"score={_fmt_float(block.get('score', 0.0), 1)}"
        )

    build_start = time.perf_counter()
    _write_progress(0, 4, pid)
    report_lines = []

    report_lines.append(f"# Integrated Project Review Report - {pid}")
    report_lines.append("")
    generated_at = now_str()
    report_lines.append(f"_Generated at: {generated_at}_")
    report_lines.append("")
    report_lines.append("_This report was generated by the AI-assisted review system for internal expert and decision-committee reference._")
    report_lines.append("")

    summary_start = time.perf_counter()
    exec_summary_md = build_executive_summary(expert_json)
    if exec_summary_md:
        report_lines.append(exec_summary_md)
        report_lines.append("")
        _log("SECTION", f"executive_summary included=True chars={len(exec_summary_md)} elapsed_sec={time.perf_counter() - summary_start:.3f}")
    else:
        _log("SECTION", f"executive_summary included=False elapsed_sec={time.perf_counter() - summary_start:.3f}")
    _write_progress(1, 4, pid)

    expert_start = time.perf_counter()
    adjusted_expert_md = clean_report_text(adjust_expert_markdown(expert_md_raw))
    # The imported Stage 6 markdown already includes detailed dimension sections.
    # Remove the compact "By dimension" echo when present because it is prone to
    # abbreviation/truncation artifacts and duplicates the full sections below.
    adjusted_expert_md = re.sub(
        r"\nBy dimension:\n(?:- .+\n)+",
        "\n",
        adjusted_expert_md,
    )
    report_lines.append("## 1. AI Expert Review (Detailed)")
    report_lines.append("")
    report_lines.append(adjusted_expert_md)
    report_lines.append("")
    _log("SECTION", f"expert_review chars={len(adjusted_expert_md)} elapsed_sec={time.perf_counter() - expert_start:.3f}")
    _write_progress(2, 4, pid)

    qa_start = time.perf_counter()
    qa_section, qa_stats = build_qa_section_from_final_payload(final_payload)
    report_lines.append(qa_section)
    report_lines.append("")
    _log(
        "SECTION",
        f"qa_traceability chars={len(qa_section)} qas_available={qa_stats['total_qas_available']} "
        f"qas_shown={qa_stats['total_qas_shown']} elapsed_sec={time.perf_counter() - qa_start:.3f}"
    )
    _write_progress(3, 4, pid)

    final_md = "\n".join(report_lines)
    _log("TIMING", f"assemble_report elapsed_sec={time.perf_counter() - build_start:.3f}")

    write_start = time.perf_counter()
    out_path.write_text(final_md, encoding="utf-8")
    _write_progress(4, 4, pid)
    _log("TIMING", f"write_report elapsed_sec={time.perf_counter() - write_start:.3f}")

    audit = {
        "stage": "stage_7_final_report_compilation",
        "purpose": "Compile the final Markdown report from Stage 5 final_payload and Stage 6 expert opinion.",
        "proposal_id": pid,
        "generated_at": generated_at,
        "elapsed_sec": round(time.perf_counter() - stage_start, 3),
        "inputs": {
            "final_payload_path": str(fp_path),
            "final_payload_size_bytes": fp_path.stat().st_size,
            "expert_markdown_path": str(expert_md_path),
            "expert_markdown_size_bytes": expert_md_path.stat().st_size,
            "expert_json_path": str(expert_json_path) if expert_json_path.exists() else "",
            "expert_json_size_bytes": expert_json_path.stat().st_size if expert_json_path.exists() else 0,
        },
        "outputs": {
            "report_path": str(out_path),
            "report_size_bytes": out_path.stat().st_size,
            "audit_path": str(audit_path),
        },
        "sections": {
            "executive_summary_included": bool(exec_summary_md),
            "executive_summary_chars": len(exec_summary_md),
            "expert_review_chars": len(adjusted_expert_md),
            "qa_traceability_chars": len(qa_section),
        },
        "qa_stats": qa_stats,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    _log("OUTPUT", f"report_path={out_path} size_bytes={out_path.stat().st_size}")
    _log("OK", f"stage7_final_report_audit written path={audit_path}")
    _log(
        "SUMMARY",
        f"sections=executive_summary:{bool(exec_summary_md)}, expert_review:True, qa_traceability:True "
        f"qas_shown={qa_stats['total_qas_shown']} report_chars={len(final_md)}"
    )
    _log("STAGE_7", f"Completed Stage 7 elapsed_sec={time.perf_counter() - stage_start:.3f}")


if __name__ == "__main__":
    main()
