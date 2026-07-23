# -*- coding: utf-8 -*-
"""
Stage 6 - AI Expert Opinion v4.1
(dimension-first, QA-grounded, general-insights aware, with local fallback)
-----------------------------------------------------------------------
Goals:
- Use only Stage 5 metrics.json and final_payload.json selected answers.
- Generate expert commentary for each of the five dimensions, including evidence direction
  and general industry context, then compose an overall opinion locally.
- Do not expose numeric scores to the LLM; provide only qualitative signals such as
  strong, medium, or weak to avoid score leakage.
- Use OpenAI by default when configured. If the LLM call fails, fall back to a local
  rule-based expert review that does not depend on an LLM.
- Produce an overall opinion as one summary plus dimension-specific bullets.
"""

import os
import re
import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.llm_provider import chat_json, default_model

# ----------------- Paths and constants -----------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
REFINED_ROOT = DATA_DIR / "refined_answers"
EXPERT_DIR = DATA_DIR / "expert_reports"
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
    "feasibility": "Resources and feasibility"
}

# ----------------- Environment -----------------
load_dotenv()
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()
OPENAI_MODEL = default_model(PROVIDER)
TIMEOUT_CONNECT = int(os.getenv("HTTP_TIMEOUT_CONNECT", "12"))
TIMEOUT_READ = int(os.getenv("HTTP_TIMEOUT_READ", "60"))


# ----------------- Small utilities -----------------
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
    """Return the most recently updated pid that has postproc metrics and final payload."""
    if not REFINED_ROOT.exists():
        return ""
    cands: List[Tuple[str, float]] = []
    for d in REFINED_ROOT.iterdir():
        if not d.is_dir():
            continue
        postproc_dir = d / "postproc"
        if (postproc_dir / "metrics.json").exists() and (postproc_dir / "final_payload.json").exists():
            cands.append((d.name, (postproc_dir / "metrics.json").stat().st_mtime))
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0] if cands else ""


# ----------------- Investment-grade scoring (separate LLM call) -----------------
INVEST_SCORE_SYSTEM = (
    "You are a senior venture/corporate-innovation investment analyst. "
    "Score each dimension 0.0–1.0 based ONLY on the proposal facts provided. "
    "Be discriminating: most proposals should land in 0.40–0.70. Scores above 0.80 require strong, specific evidence. "
    "Do NOT be generous. Do NOT reward a proposal for sounding ambitious — score only what the facts prove.\n\n"

    "=== HOW TO SCORE: CATEGORIZE FIRST, THEN SCORE WITHIN THE RANGE ===\n"
    "For each dimension, you MUST first decide which category letter applies (A/B/C/D), "
    "then assign a score strictly within that category's numeric range. "
    "Do NOT assign a score outside the category range. "
    "Resist the urge to cluster scores near 0.65-0.75 — categories exist to force real separation.\n\n"

    "=== TEAM ===\n"
    "Category T-A (score 0.83–0.95) — ANY of the following evidence sets qualifies:\n"
    "  SET 1 (Academic research leader):\n"
    "    • PI/lead researcher from CAS, Chinese National Academies, or world-class institution "
    "(MIT/Harvard/Stanford/Moderna-founder caliber)\n"
    "    • AND: significant research output (50+ publications OR major invention patents OR landmark papers cited)\n"
    "  SET 2 (Deep industry domain leader):\n"
    "    • Senior technical leader with 25+ years hands-on domain expertise "
    "(semiconductor design, pharma R&D, advanced manufacturing, etc.)\n"
    "    • AND: demonstrable commercial success (led product 0→mass production, OR company with recognized "
    "market scale, OR led team that shipped >1B units in the domain)\n"
    "  SET 3 (World-class operational excellence):\n"
    "    • Team with documented production at top industry standard "
    "(DPPM <10, >1B units shipped, >20 tier-1/Fortune-500-caliber customers in production)\n"
    "    • AND: cross-functional leadership represented (R&D + manufacturing + commercial all present in team)\n"
    "  → To qualify, the facts must explicitly support the SET being claimed. "
    "Cite the specific evidence that determines which SET applies.\n"
    "Category T-B (score 0.65–0.80) — strong team meeting ONE major condition:\n"
    "  • Major university professor (national key university, 985/211 tier) with documented research record\n"
    "  • Senior executive (10+ years as CXO/VP/Director at a recognized company in the domain)\n"
    "  • Experienced founder with a prior successful exit in an adjacent domain\n"
    "  • Company with 15+ years of domain-specific operational history in this sector\n"
    "Category T-C (score 0.45–0.62) — credible practitioners, limited proof of exceptional performance:\n"
    "  • Clinical/hospital expert with domain authority but limited publication or production-scale evidence\n"
    "  • Industry team with relevant experience but no documented large-scale execution track record\n"
    "  • Mixed team where neither academic nor industry credentials reach T-B standard individually\n"
    "Category T-D (score 0.20–0.42) — weak or unverifiable:\n"
    "  • Credentials vague or primarily AI/software applied to a domain without deep domain expertise\n"
    "  • No verifiable domain expertise cited in the provided facts\n\n"

    "=== INNOVATION ===\n"
    "CRITICAL PRELIMINARY RULES — APPLY BEFORE READING CATEGORIES:\n\n"
    "RULE 1 — AI-APPLIED-TO-WORKFLOWS: If the PRIMARY commercialized product is an AI SOFTWARE "
    "SYSTEM for clinical decision support, hospital management, disease monitoring, or industrial "
    "process optimization (i.e., AI IS the product) → automatically I-C (0.42-0.56). "
    "Engineering cleverness (bandwidth reduction, multimodal fusion) does NOT elevate above I-C.\n\n"
    "RULE 2 — AI AS A DESIGN TOOL: If the proposal uses AI/ML as a computational DESIGN TOOL "
    "to engineer a PHYSICAL product (enzyme, protein, drug molecule, material, hardware device), "
    "but the PRIMARY commercialized product is the PHYSICAL OUTPUT (not the AI software), "
    "then DO NOT apply the I-C cap. Evaluate the biological/chemical/physical innovation on its own merits "
    "using the I-A through I-D categories below.\n\n"
    "EXAMPLES:\n"
    "  → AI heart failure decision support system (AI IS the product) → I-C, score 0.42-0.56\n"
    "  → Enzyme designed using AI-driven computational chemistry (enzyme IS the product, AI is design tool) "
    "→ evaluate the enzyme's novelty, NOT the AI tool\n"
    "  → Non-natural amino acid platform (no AI involved, new molecular mechanism) → evaluate under I-A criteria\n\n"
    "Category I-A (score 0.83–0.95) — ALL three must be EXPLICITLY present in the facts:\n"
    "  1. A new mechanistic class: non-natural amino acids as molecular building blocks, "
    "a new enzyme class not previously used for this reaction type, a new material platform, "
    "or a fundamentally new physical mechanism. "
    "KEY: 'first-in-class mechanism' means the HOW (mechanism) is new, not just the WHERE (application). "
    "Example qualifying: 'non-natural amino acid technology' — these are synthetic building blocks "
    "not found in nature, representing a new class of molecular engineering tool. "
    "Example NOT qualifying: 'AI-powered optimization' — AI/ML is an existing tool being applied.\n"
    "  2. IP: explicitly claims patents filed/granted OR states 'complete independent IP rights' with specifics\n"
    "  3. Quantitative proof: specific performance numbers cited (binding affinity in nM, "
    "heat stability temperature, efficacy vs. control, benchmark comparison with exact figures)\n"
    "  → Missing ANY one of the three → cannot be I-A\n"
    "Category I-B (score 0.63–0.80) — novel but missing one I-A condition:\n"
    "  • Globally unique hardware/device with verified test results but mechanism is refinement not new class\n"
    "  • 'First' in a category with partial evidence (strong claims, some validation, IP filed)\n"
    "  • Strong IP portfolio but quantitative comparison data absent\n"
    "  NOTE: A novel enzyme with improved performance (vs existing enzymes of same class) → I-B not I-A\n"
    "Category I-C (score 0.40–0.60) — application-layer innovation (INCLUDES most AI proposals):\n"
    "  • AI/ML/software applied to any existing domain (clinical, manufacturing, logistics)\n"
    "  • Novel claims without specific quantitative validation data\n"
    "  • Enzyme or biological process improvement within the same enzyme class\n"
    "  HARD CAP: Any proposal described as 'AI-powered', 'intelligent platform', 'data-driven', "
    "'multimodal data integration', or 'machine learning for [clinical/industrial task]' → "
    "score 0.42–0.56, NO EXCEPTIONS\n"
    "Category I-D (score 0.20–0.38) — incremental or commodity:\n"
    "  • No novel mechanism; existing approach applied in new geography or market\n\n"

    "=== OBJECTIVES ===\n"
    "Category O-A (score 0.78–0.92): Quantified market gap with specific numbers, "
    "named buyer/payer pathway, documented unmet need with regulatory or clinical evidence\n"
    "Category O-B (score 0.58–0.75): Clear problem with market size estimate, "
    "but buyer pathway or demand validation is vague or aspirational\n"
    "Category O-C (score 0.38–0.55): Problem is real but market size or addressable "
    "segment is undefined or speculative; no buyer evidence\n\n"

    "=== STRATEGY ===\n"
    "Score the SOUNDNESS and DEFENSIBILITY of the strategy, NOT documentation completeness. "
    "A strategy with a strong competitive moat and realistic execution path scores higher than "
    "one with signed LOIs but poor defensibility or a team that cannot realistically execute it.\n"
    "Category S-A (score 0.78–0.92) — Strong on BOTH defensibility AND execution realism:\n"
    "  • Clear competitive moat: first-mover IP, regulatory approval already in progress, "
    "exclusive distribution channel, or network effects already demonstrated\n"
    "  • AND: strategy matches team capabilities (the team can realistically execute this plan "
    "given their documented backgrounds and resources)\n"
    "  • AND: at least ONE market validation signal: named customer already in production, "
    "OR regulatory submission filed, OR existing paying customers cited in facts\n"
    "Category S-B (score 0.58–0.75): Coherent strategy with partial moat or partial execution evidence:\n"
    "  • Plausible differentiation (technical, regulatory, cost, or relationship-based) with some evidence\n"
    "  • OR: Early partnerships or LOIs in hand but commercialization path not yet validated\n"
    "  • OR: Clear phase-gate strategy with realistic milestones matched to the team's capabilities\n"
    "Category S-C (score 0.38–0.55): Aspirational or generic:\n"
    "  • Market entry plan is standard ('partner with hospitals', 'sell to enterprises') without differentiation\n"
    "  • Competitive advantage is claimed but not evidenced in the provided facts\n"
    "  • No validation of market demand or competitive positioning beyond market size estimates\n\n"

    "=== FEASIBILITY ===\n"
    "Category F-A (score 0.78–0.92): Itemized budget, realistic multi-year milestones, "
    "regulatory pathway named, existing infrastructure or confirmed manufacturing partner\n"
    "Category F-B (score 0.58–0.75): Reasonable budget with partial detail; milestones exist\n"
    "Category F-C (score 0.38–0.55): Budget stated without breakdown; timeline vague; risks acknowledged but not mitigated\n"
    "Category F-D (score 0.20–0.35): Budget unrealistic or missing; critical dependencies unresolved\n\n"

    "Return ONLY a valid JSON object with this exact structure: "
    "{\"scores\": {\"team\": <float>, \"objectives\": <float>, \"strategy\": <float>, \"innovation\": <float>, \"feasibility\": <float>}, "
    "\"categories\": {\"team\": \"T-A|T-B|T-C|T-D\", \"objectives\": \"O-A|O-B|O-C\", "
    "\"strategy\": \"S-A|S-B|S-C\", \"innovation\": \"I-A|I-B|I-C|I-D\", \"feasibility\": \"F-A|F-B|F-C|F-D\"}, "
    "\"rationale\": {\"team\": \"<cite specific evidence that determined the category>\", \"objectives\": \"...\", "
    "\"strategy\": \"...\", \"innovation\": \"...\", \"feasibility\": \"...\"}}"
)


def investment_score_from_dims(pid: str, provider: str, model: str,
                                dims_v2_path: Path) -> Dict[str, float]:
    """
    Call LLM with raw dimension facts from dimensions_v2.json to get
    investment-grade scores per dimension (0-1). Returns empty dict on failure.
    """
    if not dims_v2_path.exists():
        _log("WARN", f"dimensions_v2.json not found for {pid}, skipping investment scoring")
        return {}
    try:
        dims_data = read_json(dims_v2_path)
    except Exception as e:
        _log("WARN", f"Could not read dimensions_v2.json for {pid}: {e}")
        return {}

    # Build a compact fact summary per dimension
    fact_blocks: Dict[str, str] = {}
    for dim in DIM_ORDER:
        d = dims_data.get(dim, {}) or {}
        parts = []
        if d.get("summary"):
            parts.append(f"Summary: {d['summary'][:600]}")
        kps = d.get("key_points") or []
        if kps:
            parts.append("Key facts: " + " | ".join(str(k)[:150] for k in kps[:10]))
        numbers = d.get("numbers") or []
        if numbers:
            parts.append("Numbers/metrics: " + " | ".join(str(n)[:100] for n in numbers[:6]))
        risks = d.get("risks") or []
        if risks:
            parts.append("Risks: " + " | ".join(str(r)[:100] for r in risks[:5]))
        mitigations = d.get("mitigations") or []
        if mitigations:
            parts.append("Mitigations: " + " | ".join(str(m)[:100] for m in mitigations[:3]))
        fact_blocks[dim] = "\n".join(parts) if parts else "(no facts available)"

    user_payload = {
        "pid": pid,
        "task": "investment_grade_scoring",
        "proposal_facts_by_dimension": fact_blocks,
        "instruction": (
            "Score each dimension 0.0–1.0 using the criteria in the system prompt. "
            "Base scores ONLY on the provided facts, not on document quality or answer length."
        )
    }

    try:
        resp = call_openai_chat(
            provider=provider,
            model=model,
            system_prompt=INVEST_SCORE_SYSTEM,
            user_payload=user_payload,
            temperature=0.0,
            max_tokens=600,
            seed=42,
        )
        raw_scores = resp.get("scores", {}) or {}
        rationale = resp.get("rationale", {}) or {}
        categories = resp.get("categories", {}) or {}
        result = {}
        for dim in DIM_ORDER:
            s = raw_scores.get(dim)
            if s is not None:
                try:
                    result[dim] = max(0.0, min(1.0, float(s)))
                except Exception:
                    pass
        if categories:
            _log("INVEST", f"investment categories: {json.dumps(categories, ensure_ascii=False)}")
        if rationale:
            _log("INVEST", f"investment rationale: {json.dumps(rationale, ensure_ascii=False)}")
        _log("INVEST", f"investment scores: {result}")
        return result
    except Exception as e:
        _log("WARN", f"Investment scoring LLM call failed: {e}")
        return {}


# ----------------- Dimension score signals -> qualitative hints -----------------
def _score_hint(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "Score signal is unclear; information may be insufficient."
    if v >= 0.75:
        return "Strong signal; this dimension appears relatively strong."
    if v >= 0.62:
        return "Moderately strong signal; clear advantages exist, with room to improve."
    if v >= 0.50:
        return "Medium-to-weak signal; several weaknesses or information gaps may exist."
    if v >= 0.35:
        return "Weak signal; this dimension appears to have clear limitations or limited evidence."
    return "Very weak signal; this dimension is a major concern and needs focused remediation."


def _align_hint(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "Cross-candidate consistency signal is unclear."
    if v >= 0.8:
        return "Candidate answers are highly consistent; conclusions are relatively stable."
    if v >= 0.6:
        return "Candidate answers are broadly consistent, with some differences."
    if v >= 0.4:
        return "Candidate answers show notable disagreement; interpret with caution."
    return "Candidate answers differ substantially; conclusions for this dimension are unstable."


def _drift_hint(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "Content drift signal is unclear."
    if v <= 0.18:
        return "Answers stay close to the same core topic; drift is low."
    if v <= 0.30:
        return "Some drift exists, but answers mostly stay on topic."
    if v <= 0.45:
        return "Clear drift exists; stable conclusions should be separated from unstable points."
    return "High drift; this dimension has semantic instability risk."

def _split_keywords(text: str) -> List[str]:
    """
    Lightweight token splitting for rough matching between general_insights and QA content.
    """
    if not text:
        return []
    tokens = re.split(r"[,.;:/\\()\s]+", text)
    tokens = [t.strip().lower() for t in tokens if len(t.strip()) >= 3]
    return tokens

def build_dim_inputs(metrics: Dict[str, Any],
                     final_payload: Dict[str, Any],
                     max_qas: int = 6,
                     max_answer_chars: int = 800) -> Dict[str, Any]:
    """
    Build dimension-level input for the LLM:
    - no numeric scores, only qualitative signals
    - include top_evidence_phrases and general_insights from Stage 5
    """
    dim_inputs: Dict[str, Any] = {}
    dim_metrics = metrics.get("dimensions", {}) or {}
    fp_dims = final_payload.get("dimensions", {}) or {}

    for dim in DIM_ORDER:
        m = dim_metrics.get(dim, {}) or {}
        f = fp_dims.get(dim, {}) or {}
        qas = f.get("qas", []) or []

        # Dimension-level general insights.
        dim_general_insights = f.get("general_insights") or []
        # Evidence phrases from Stage 5.
        top_evid_phrases = m.get("top_evidence_phrases") or []
        redlined_samples = m.get("redlined_samples") or []

        # Aggregate QA text to roughly match against general insights.
        corpus_parts: List[str] = []
        for qa in qas:
            corpus_parts.append((qa.get("q") or ""))
            corpus_parts.append((qa.get("answer") or ""))
            for c in qa.get("claims") or []:
                corpus_parts.append(c)
            for h in qa.get("evidence_hints") or []:
                corpus_parts.append(h)
        corpus_text = " ".join(corpus_parts).lower()

        # Split general insights into partially covered vs. missing.
        dim_general_insights_covered: List[str] = []
        dim_general_insights_missing: List[str] = []
        for gi in dim_general_insights:
            if not gi:
                continue
            gi_tokens = _split_keywords(gi)
            # Empty-token insights are treated as missing to avoid false coverage.
            if not gi_tokens:
                dim_general_insights_missing.append(gi)
                continue
            hit = any(tok in corpus_text for tok in gi_tokens)
            if hit:
                dim_general_insights_covered.append(gi)
            else:
                dim_general_insights_missing.append(gi)

        samples = []
        for qa in qas[:max_qas]:
            ans = (qa.get("answer") or "").strip()
            if len(ans) > max_answer_chars:
                ans = ans[:max_answer_chars] + "..."
            samples.append({
                "question": (qa.get("q") or "").strip(),
                "answer": ans,
                "key_claims": (qa.get("claims") or [])[:6],
                "evidence_hints": (qa.get("evidence_hints") or [])[:6],
                "provider": qa.get("provider", ""),
                "selection_mode": qa.get("selection_mode", ""),
                "fallback_used": bool(qa.get("fallback_used", False)),
                "general_insights": (qa.get("general_insights") or [])[:6],
            })

        dim_inputs[dim] = {
            "dimension": dim,
            "label": DIM_LABELS.get(dim, dim),
            "score_hint": _score_hint(m.get("avg")),
            "alignment_hint": _align_hint(m.get("avg_alignment")),
            "drift_hint": _drift_hint(m.get("avg_drift")),
            "metric_strength_phrases": (m.get("strengths") or [])[:6],
            "metric_risk_phrases": (m.get("risks") or [])[:6],
            "metric_top_evidence_phrases": top_evid_phrases[:6],
            "metric_redlined_samples": redlined_samples[:6],
            "dim_general_insights": dim_general_insights[:10],
            "dim_general_insights_covered": dim_general_insights_covered[:10],
            "dim_general_insights_missing": dim_general_insights_missing[:10],
            "qa_samples": samples
        }
    return dim_inputs


# ----------------- Provider Chat call -----------------
def call_openai_chat(provider: str,
                     model: str,
                     system_prompt: str,
                     user_payload: Dict[str, Any],
                     temperature: float = 0.25,
                     max_tokens: int = 2600,
                     seed: int = None,
                     max_retries: int = 3,
                     backoff: float = 1.8) -> Dict[str, Any]:
    """
    Call the configured provider with the given parameters.
    Provider selection and fallback decisions are handled by the caller.
    """
    last_err = None
    call_start = time.perf_counter()
    _log(
        "LLM_CALL",
        f"start model={model} max_tokens={max_tokens} temperature={temperature} "
        f"payload_chars={len(json.dumps(user_payload, ensure_ascii=False))}"
    )
    for attempt in range(1, max_retries + 1):
        try:
            attempt_start = time.perf_counter()
            parsed = chat_json(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            )
            _log(
                "LLM_CALL",
                f"done attempts={attempt} elapsed_sec={time.perf_counter() - call_start:.2f}"
            )
            return parsed
        except Exception as e:
            last_err = str(e)
            _log("LLM_CALL", f"attempt={attempt} failed elapsed_sec={time.perf_counter() - attempt_start:.2f} error={e}")
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"{provider} Chat call failed after {max_retries} retries: {last_err}")


# ----------------- Prompt construction -----------------
def build_dim_system_prompt() -> str:
    return (
        "You are a senior venture, product, technical, financial, and operating diligence advisor for world-class startup, corporate innovation, product, service, platform, and operating-business reviews. "
        "Before writing any assessment, infer the closest business archetype from the evidence: SaaS/software, AI/data product, marketplace, consumer brand, B2B service, professional services, education/training, healthcare/life science, industrial/manufacturing, hardware/device, energy/climate, logistics/supply-chain, fintech/financial service, real estate/infrastructure, agriculture/food, media/content, nonprofit/public-sector program, research commercialization, internal corporate transformation, or mixed model. "
        "Use that archetype only to choose relevant diligence standards; do not output an archetype label unless it is useful and clearly supported. "
        "The system will provide five dimensions: team, objectives, strategy, innovation, and feasibility. "
        "For each dimension you will receive: a label; qualitative score/consistency/drift signals; "
        "strength/risk phrases extracted by metrics; top_evidence_phrases from post-processing, which are evidence directions rather than complete evidence; "
        "dimension-level and question-level general_insights, which are industry-context benchmarks and not project achievements; "
        "and sample-level provenance flags such as selection_mode and fallback_used, which indicate whether the selected answer came from the normal filtered path or a fallback rescue path; "
        "dim_general_insights_covered, meaning general industry points partly reflected in the selected QA; "
        "dim_general_insights_missing, meaning important industry points barely covered in the selected QA; "
        "and several QA samples. "
        "Your task:\n"
        "1. For each dimension, write an investment/scale-up-grade assessment based on QA content, metric phrases, evidence directions, and general insights. "
        "The standard of proof must match the inferred archetype: software needs adoption, retention, security, integration, and support evidence; services need repeatability, staffing, service quality, utilization, and margin evidence; consumer businesses need channel economics, repeat purchase, brand differentiation, and working-capital evidence; marketplaces need liquidity, trust, acquisition, take-rate, and disintermediation evidence; manufacturing/hardware needs BOM, yield, reliability, certification, supply chain, inventory, and warranty evidence; public/nonprofit programs need theory of change, funding durability, implementation partners, and measurable outcomes; research commercialization needs IP rights, validation milestones, partner route, and capital-intensity evidence. "
        "Strengths must connect concrete proposal facts to a business or execution implication: why this could improve defensibility, speed, cost, customer adoption, compliance readiness, operating readiness, delivery readiness, or capital efficiency. "
        "Concerns must explain the decision impact, not just say information is limited. Specify whether the gap affects problem proof, customer/user proof, buyer/payer/funder proof, product/service proof, technical proof, commercial proof, operating proof, financial proof, legal/compliance proof, impact proof, or governance proof. "
        "Recommendations must be diligence-ready asks: name the exact material to add, such as customer interviews, pilot results, LOIs, usage/retention data, service quality metrics, benchmark test results, IP claim charts, channel economics, unit-economics assumptions, capacity/resource plan, staffing model, supply-chain plan, budget use-of-funds, risk register, owner/timeline/RACI, procurement path, compliance pathway, impact measurement plan, or partner agreements when applicable.\n"
        "2. Each strength/concern must be causal and explanatory: say why it is good or risky, not just what exists. Vary sentence openings and avoid generic phrases such as 'aligns with industry expectations' unless you also name the concrete project fact and decision implication.\n"
        "3. You may cite key QA information, but do not invent organizations, registration numbers, customer data, production data, or concrete data.\n"
        "Do not assume any sector, customer type, business model, regulated pathway, or delivery model unless the QA content supports it; use validation, quality, customer qualification, operating scaling, adoption, security, procurement, funding, implementation, impact, or compliance language when that better fits the project.\n"
        "4. Do not output question IDs, numeric scores, percentages, or internal metric names such as alignment, coverage, authority, drift, overall_score, confidence, or Jaccard.\n"
        "5. If a dimension has limited information or many fallback-selected samples, provide conservative conclusions and separate 'what is promising' from 'what must be proven before scale-up or investment'.\n"
        "Return strict JSON. For each dimension provide summary (2-4 sentences), strengths (3-5 items), concerns (3-5 items), and recommendations (3-5 items)."
    )

def build_dim_user_payload(pid: str,
                           dim_inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pid": pid,
        "task": "dimension_level_expert_opinion",
        "note": "Return only the dimensions field. Other fields are added by the system.",
        "dimensions": dim_inputs,
        "output_schema_hint": {
            "type": "object",
            "required": ["dimensions"],
            "properties": {
                "dimensions": {
                    "type": "object",
                    "properties": {
                        dim: {
                            "type": "object",
                            "required": ["summary", "strengths", "concerns", "recommendations"],
                            "properties": {
                                "summary": {"type": "string"},
                                "strengths": {"type": "array", "items": {"type": "string"}},
                                "concerns": {"type": "array", "items": {"type": "string"}},
                                "recommendations": {"type": "array", "items": {"type": "string"}}
                            }
                        } for dim in DIM_ORDER
                    }
                }
            }
        }
    }


# ----------------- Text cleanup and aggregation -----------------
FORBID_PATTERNS = [
    r"\bQ\d+\b",
    r"\balign(?:ment)?\b",
    r"\bcoverage\b",
    r"\bauth(?:ority)?\b",
    r"\bdrift\b",
    r"\boverall[_ ]?score\b",
    r"\bconfidence\b",
    r"\bjaccard\b",
    r"\d+(\.\d+)?\s*%+",
]


def clean_text(s: str) -> str:
    s = (s or "").replace("\u0000", "").strip()
    for pat in FORBID_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.replace("()", "")
    s = re.sub(r"\bwhich with\b", "which align with", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthat with\b", "that align with", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthat do not with\b", "that do not include", s, flags=re.IGNORECASE)
    s = re.sub(r"\bvalidation of with\b", "validation against", s, flags=re.IGNORECASE)
    s = re.sub(r"\bindicating with\b", "indicating", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfor project\.", "for project execution.", s, flags=re.IGNORECASE)
    s = re.sub(r"\band stakeholders\.", "and stakeholder coordination.", s, flags=re.IGNORECASE)
    s = re.sub(r"\bstakeholder\s+\.", "stakeholders.", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s.strip()


def clean_list(items: List[str]) -> List[str]:
    out: List[str] = []
    for it in items or []:
        t = clean_text(it)
        if t:
            out.append(t)
    return out


def dedup_soft(items: List[str], thresh: float = 0.85) -> List[str]:
    """Simple character-level Jaccard deduplication."""
    def to_set(x: str):
        return set((x or "").lower())

    uniq: List[str] = []
    for s in items or []:
        keep = True
        a = to_set(s)
        for t in uniq:
            b = to_set(t)
            if not a or not b:
                continue
            j = len(a & b) / len(a | b)
            if j >= thresh:
                keep = False
                break
        if keep:
            uniq.append(s)
    return uniq


def _shorten_sentence(text: str, max_len: int = 120) -> str:
    """For overall summary dimension bullets: take the first sentence or truncate."""
    text = clean_text(text or "")
    if not text:
        return ""
    protected = (
        text.replace("Dr.", "Dr<dot>")
        .replace("Mr.", "Mr<dot>")
        .replace("Ms.", "Ms<dot>")
        .replace("Prof.", "Prof<dot>")
    )
    parts = re.split(r"[!?.]", protected)
    for p in parts:
        p = p.replace("<dot>", ".").strip()
        if p:
            text = p
            break
    if len(text) > max_len:
        cut = text[:max_len].rstrip()
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0].rstrip(" ,;:")
        return cut + "..."
    return text


# ----------------- Local dimension expert blocks (LLM fallback) -----------------
def build_local_dim_blocks(metrics: Dict[str, Any],
                           final_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the LLM cannot be called, build dimension commentary directly from metrics and final_payload.
    The logic is intentionally simple and avoids extra speculation.
    """
    dims_metrics = metrics.get("dimensions", {}) or {}
    fp_dims = final_payload.get("dimensions", {}) or {}

    dim_blocks: Dict[str, Any] = {}

    for dim in DIM_ORDER:
        m = dims_metrics.get(dim, {}) or {}
        f = fp_dims.get(dim, {}) or {}

        score = float(m.get("avg", 0.0) or 0.0)
        align = float(m.get("avg_alignment", 0.0) or 0.0)
        drift = float(m.get("avg_drift", 0.0) or 0.0)
        strengths_phr = (m.get("strengths") or [])[:5]
        risks_phr = (m.get("risks") or [])[:5]
        dim_gi = (f.get("general_insights") or [])[:8]

        label = DIM_LABELS.get(dim, dim)

        summary_parts: List[str] = []
        if strengths_phr:
            summary_parts.append(
                f"In the {label} dimension, the selected answers show several strengths, including: " +
                "; ".join(strengths_phr[:2])
            )
        if risks_phr:
            summary_parts.append(
                "They also show potential issues or risks, including: " +
                "; ".join(risks_phr[:2])
            )
        if not summary_parts:
            summary_parts.append(
                f"The current selected answers contain limited effective information about the {label} dimension; conclusions are provisional and should be supported with more detailed facts and metrics."
            )
        summary = " ".join(summary_parts)

        # Strengths: use metric strengths first; if absent, use general insights as context.
        strengths_out = strengths_phr[:]
        if not strengths_out and dim_gi:
            strengths_out = [
                f"From industry experience, this dimension would be strengthened if the project can demonstrate the following practice: {dim_gi[0]}"
            ]

        # Concerns: use metric risks directly.
        concerns_out = risks_phr[:]

        # Recommendations: use general insights first, then fallback recommendations.
        recs_out: List[str] = []
        for g in dim_gi:
            recs_out.append(g)
        if risks_phr and not dim_gi:
            recs_out.append(
                "To address the risks above, supplement the next version with a more specific implementation plan, milestones, and quantitative metrics."
            )
        if not recs_out:
            recs_out.append(
                f"For the {label} dimension, systematically document relevant experience, resource support, and implementation path, then align missing information with industry best practice."
            )

        dim_blocks[dim] = {
            "score_echo": score,
            "alignment_echo": align,
            "drift_echo": drift,
            "summary": summary,
            "strengths": strengths_out,
            "concerns": concerns_out,
            "recommendations": recs_out
        }

    return dim_blocks


# ----------------- Overall opinion (local composition) -----------------
def build_overall_from_dims(dim_blocks: Dict[str, Any],
                            metrics_overall: Dict[str, Any],
                            metrics_dims: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compose the overall opinion from dimension commentary and scores.
    This stage is local and does not call the LLM.
    """
    overall_score = float(metrics_overall.get("overall_score", 0.0) or 0.0)
    overall_conf = float(metrics_overall.get("overall_confidence", 0.0) or 0.0)

    def verdict_rule(score: float, conf: float,
                     dims: Dict[str, Any]) -> (str, str):
        inv = float(dims.get("innovation", {}).get("avg", 0.0) or 0.0)
        fea = float(dims.get("feasibility", {}).get("avg", 0.0) or 0.0)

        if score >= 0.725 or conf >= 0.811:
            return "GO", "Overall score and confidence are relatively strong, key dimensions appear solid, and risks look controllable enough to proceed."

        if score < 0.40 or inv < 0.30 or fea < 0.30:
            return (
                "NO-GO",
                "Overall score or key dimensions, especially innovation or feasibility, are clearly weak; key information gaps and weaknesses are substantial, so this round should not proceed without additional materials."
            )

        return (
            "HOLD",
            "The project has credible signals, but the current evidence package is not strong enough for a scale-up or investment-grade decision. Clarify proof milestones, customer or market validation, execution risks, and resource requirements before making a firm go/no-go decision."
        )

    verdict, verdict_reason = verdict_rule(overall_score, overall_conf, metrics_dims)

    if verdict == "GO":
        head = (
            "Overall, the project appears relatively solid in this review round: key assumptions are reasonably clear, the implementation path has some operational basis, and the project may be worth advancing if risks remain controlled."
        )
    elif verdict == "HOLD":
        head = (
            "Overall, the project shows promise in its technical direction and application scenario, but the current evidence package is not yet investment- or scale-up-grade. It should move to a focused diligence round rather than an immediate large-scale commitment."
        )
    else:  # NO-GO
        head = (
            "Overall, the project may contain interesting technical or application ideas, but the current materials do not support a robust risk-benefit judgment. Weaknesses and uncertainties remain too prominent for direct approval in this round."
        )

    dim_snippets: List[str] = []
    for dim in DIM_ORDER:
        blk = dim_blocks.get(dim, {}) or {}
        dim_sum = (blk.get("summary") or "").strip()
        if not dim_sum:
            continue
        label = DIM_LABELS.get(dim, dim)
        short = _shorten_sentence(dim_sum, max_len=140)
        if not short:
            continue
        dim_snippets.append(f"- {label}: {short}")

    lines: List[str] = [head]
    if dim_snippets:
        lines.append("By dimension:")
        lines.extend(dim_snippets)

    summary_text = "\n".join(lines)

    key_strengths: List[str] = []
    key_risks: List[str] = []
    for dim in DIM_ORDER:
        blk = dim_blocks.get(dim, {}) or {}
        label = DIM_LABELS.get(dim, dim)
        for s in (blk.get("strengths") or [])[:2]:
            key_strengths.append(f"[{label}] {s}")
        for r in (blk.get("concerns") or [])[:2]:
            key_risks.append(f"[{label}] {r}")

    key_strengths = dedup_soft(clean_list(key_strengths))[:6]
    key_risks = dedup_soft(clean_list(key_risks))[:6]

    recs: List[str] = []
    for dim in DIM_ORDER:
        blk = dim_blocks.get(dim, {}) or {}
        label = DIM_LABELS.get(dim, dim)
        for r in (blk.get("recommendations") or [])[:2]:
            recs.append(f"[{label}] {r}")
    recs = dedup_soft(clean_list(recs))[:8]

    return {
        "summary": summary_text,
        "overall_score_echo": overall_score,
        "confidence_echo": overall_conf,
        "key_strengths": key_strengths,
        "key_risks": key_risks,
        "recommendations": recs,
        "verdict": verdict,
        "basis": [
            f"Decision basis: {verdict_reason}",
            "This conclusion is based only on selected QA results and automatic scoring signals; no external materials were introduced."
        ]
    }


# ----------------- Markdown rendering -----------------
def _bar(v: float, n: int = 20) -> str:
    try:
        v = float(v)
    except Exception:
        v = 0.0
    v = max(0.0, min(1.0, v))
    k = int(round(v * n))
    return "#" * k + "-" * (n - k)


def render_markdown(opinion: Dict[str, Any]) -> str:
    meta = opinion.get("meta", {}) or {}
    overall = opinion.get("overall_opinion", {}) or {}
    dims = opinion.get("dimensions", {}) or {}
    scoring = opinion.get("scoring_explainer", {}) or {}
    metrics_path = (meta.get("sources") or {}).get("metrics_path", "")

    lines: List[str] = []
    lines.append(f"# AI Expert Review - {meta.get('pid', '')}")
    lines.append("")
    lines.append(f"- Generated at: {meta.get('generated_at', '')}")
    lines.append(f"- Mode: {meta.get('mode', '')}")
    lines.append(f"- Model/engine: {meta.get('model', '')} (provider={meta.get('provider', '')})")
    lines.append("")

    # Overall opinion.
    lines.append("## Overall Opinion")
    lines.append(f"- Overall score echo: {overall.get('overall_score_echo', 0.0):.3f}  {_bar(overall.get('overall_score_echo', 0.0))}")
    lines.append(f"- Overall confidence echo: {overall.get('confidence_echo', 0.0):.3f}  {_bar(overall.get('confidence_echo', 0.0))}")
    lines.append("")
    if overall.get("summary"):
        lines.append(overall["summary"])
        lines.append("")
    if overall.get("key_strengths"):
        lines.append("**Project Strengths**")
        for s in overall["key_strengths"]:
            lines.append(f"- {s}")
        lines.append("")
    if overall.get("key_risks"):
        lines.append("**Project Weaknesses / Potential Risks**")
        for r in overall["key_risks"]:
            lines.append(f"- {r}")
        lines.append("")
    if overall.get("recommendations"):
        lines.append("**Overall Recommendations**")
        for r in overall["recommendations"]:
            lines.append(f"- {r}")
        lines.append("")
    if overall.get("verdict"):
        lines.append(f"**Overall verdict:** {overall['verdict']}")
        lines.append("")
    if overall.get("basis"):
        lines.append("**Decision Basis (System Generated)**")
        for b in overall["basis"]:
            lines.append(f"- {b}")
        lines.append("")

    # Dimension score table.
    lines.append("## Dimension Score Overview")
    lines.append("")
    lines.append("| Dimension | Score |")
    lines.append("|---|---:|")
    for dim in DIM_ORDER:
        blk = dims.get(dim, {}) or {}
        label = DIM_LABELS.get(dim, dim)
        lines.append(f"| {label} ({dim}) | {blk.get('score_echo', 0.0):.3f} |")
    lines.append("")

    # Dimension details.
    lines.append("## Dimension-Level Expert Commentary")
    lines.append("")
    for dim in DIM_ORDER:
        label = DIM_LABELS.get(dim, dim)
        blk = dims.get(dim, {}) or {}
        lines.append(f"### {label} ({dim})")
        lines.append(f"- Score echo: {blk.get('score_echo', 0.0):.3f}  {_bar(blk.get('score_echo', 0.0))}")
        lines.append("")
        if blk.get("summary"):
            lines.append(blk["summary"])
            lines.append("")
        if blk.get("strengths"):
            lines.append("**Strengths**")
            for s in blk["strengths"]:
                lines.append(f"- {s}")
            lines.append("")
        if blk.get("concerns"):
            lines.append("**Concerns / Risks**")
            for r in blk["concerns"]:
                lines.append(f"- {r}")
            lines.append("")
        if blk.get("recommendations"):
            lines.append("**Recommendations**")
            for r in blk["recommendations"]:
                lines.append(f"- {r}")
            lines.append("")

    # Echo only the compact scoring configuration used for auditability.
    if scoring:
        lines.append("## Scoring Rule Echo (From Post-Processing Config)")
        lines.append(f"- consistency_weight: {scoring.get('consistency_weight', 0.0):.2f}")
        if scoring.get("dimension_weight"):
            dw = scoring["dimension_weight"]
            order_str = ", ".join([f"{d}:{dw.get(d, 0.0):.2f}" for d in DIM_ORDER if d in dw])
            lines.append(f"- dimension_weight: {order_str}")
        lines.append("")

    if metrics_path:
        lines.append("## Traceability")
        lines.append(f"- metrics.json: {metrics_path}")
        lines.append("")

    return "\n".join(lines)


# ----------------- Main flow -----------------
def main():
    stage_start = time.perf_counter()
    ap = argparse.ArgumentParser(description="Generate AI expert opinion (dimension-first, QA-grounded).")
    ap.add_argument("--pid", type=str, default="", help="Proposal ID. If omitted, the latest available project is used.")
    ap.add_argument("--provider", type=str, default=PROVIDER, help="LLM provider: openai, gemini, or deepseek.")
    ap.add_argument("--model", type=str, default="", help="Model name. Defaults to provider-specific env model.")
    ap.add_argument("--out_dir", type=str, default="", help="Optional output directory.")
    ap.add_argument("--dry_run", action="store_true", help="Write the prompt only; do not call the LLM.")
    ap.add_argument("--no_markdown", action="store_true", help="Write JSON only; skip Markdown output.")
    ap.add_argument("--force_local", action="store_true", help="Force local rule-based mode; do not call the LLM.")
    ap.add_argument("--strict", action="store_true", help="Fail instead of writing local fallback when provider call fails.")
    args = ap.parse_args()
    provider = (args.provider or PROVIDER).strip().lower()
    model = args.model.strip() or default_model(provider)

    _log("STAGE_6", "Starting Stage 6: AI expert opinion")
    _log("STAGE_6", "Purpose: turn Stage 5 post-processed answers and metrics into a final expert-style review.")

    pid = args.pid.strip() or detect_latest_pid()
    if not pid:
        raise RuntimeError("No available project detected. Expected refined_answers/<pid>/postproc/metrics.json and final_payload.json.")

    postproc_dir = REFINED_ROOT / pid / "postproc"
    metrics_path = postproc_dir / "metrics.json"
    payload_path = postproc_dir / "final_payload.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    if not payload_path.exists():
        raise FileNotFoundError(f"final_payload.json not found: {payload_path}")

    _log("INPUT", f"proposal_id={pid}")
    _log("INPUT", f"metrics_path={metrics_path} size_bytes={metrics_path.stat().st_size}")
    _log("INPUT", f"final_payload_path={payload_path} size_bytes={payload_path.stat().st_size}")
    _log("CONFIG", f"provider={provider} model={model} force_local={args.force_local} dry_run={args.dry_run}")

    metrics = read_json(metrics_path)
    final_payload = read_json(payload_path)

    build_start = time.perf_counter()
    dim_inputs = build_dim_inputs(metrics, final_payload)
    _log("TIMING", f"build_dim_inputs elapsed_sec={time.perf_counter() - build_start:.3f}")
    for dim in DIM_ORDER:
        di = dim_inputs.get(dim, {}) or {}
        _log(
            "DIM_INPUT",
            f"{dim}: qa_samples={len(di.get('qa_samples') or [])} "
            f"general_insights={len(di.get('dim_general_insights') or [])} "
            f"covered={len(di.get('dim_general_insights_covered') or [])} "
            f"missing={len(di.get('dim_general_insights_missing') or [])} "
            f"score_hint='{di.get('score_hint', '')}'"
        )

    out_dir = Path(args.out_dir) if args.out_dir else (EXPERT_DIR / pid)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "ai_expert_opinion.prompt.json"
    json_path = out_dir / "ai_expert_opinion.json"
    md_path = out_dir / "ai_expert_opinion.md"
    audit_path = out_dir / "stage6_ai_expert_opinion_audit.json"

    # Save the prompt even for local mode so the exact Stage 6 input can be audited.
    system_prompt = build_dim_system_prompt()
    user_payload = build_dim_user_payload(pid, dim_inputs)
    write_json(prompt_path, {
        "system": system_prompt,
        "user": user_payload
    })
    _log("OK", f"prompt written path={prompt_path}")

    # Dimension-level commentary: prefer LLM, then fall back to local rules.
    dim_op_raw: Dict[str, Any] = {}
    used_model = ""
    used_mode = ""
    llm_attempted = False
    fallback_reason = ""

    use_llm = (not args.force_local) and provider in {"openai", "deepseek", "gemini"}

    if args.dry_run:
        _log("STAGE_6", f"Dry run complete; prompt written path={prompt_path}")
        return

    if use_llm:
        llm_attempted = True
        try:
            resp = call_openai_chat(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_payload=user_payload,
                temperature=0.25,
                max_tokens=2600,
                seed=None,
            )
            if "dimensions" not in resp:
                raise RuntimeError("LLM response JSON is missing the 'dimensions' field.")
            dim_op_raw = resp["dimensions"] or {}
            used_model = model
            used_mode = "llm"
        except Exception as e:
            if args.strict:
                raise
            fallback_reason = str(e)
            _log("WARN", f"LLM dimension commentary failed; using local rule fallback. error={e}")
            dim_op_raw = {}
            used_model = "local_rules"
            used_mode = "local_fallback"
    else:
        used_model = "local_rules"
        used_mode = "local_forced"
        reason = "force_local=True" if args.force_local else f"provider/key configuration does not enable {provider}"
        _log("MODE", f"Using local rule-based mode because {reason}.")

    if not dim_op_raw:
        local_start = time.perf_counter()
        dim_op_raw = build_local_dim_blocks(metrics, final_payload)
        _log("TIMING", f"build_local_dim_blocks elapsed_sec={time.perf_counter() - local_start:.3f}")

    # Investment-grade scoring: independent LLM evaluation from raw proposal facts.
    # This bypasses QA document-quality bias and uses explicit VC-grade criteria.
    INVEST_WEIGHT = 0.90  # weight on investment score vs QA score (calibrated)
    dims_v2_path = DATA_DIR / "extracted" / pid / "dimensions_v2.json"
    use_llm_invest = (not args.force_local) and provider in {"openai", "deepseek", "gemini"}
    invest_scores: Dict[str, float] = {}
    if use_llm_invest and not args.dry_run:
        invest_start = time.perf_counter()
        invest_scores = investment_score_from_dims(pid, provider, model, dims_v2_path)
        _log("TIMING", f"investment_scoring elapsed_sec={time.perf_counter() - invest_start:.3f}")
        if invest_scores:
            _log("INVEST", f"investment scoring succeeded for {len(invest_scores)}/{len(DIM_ORDER)} dims")
        else:
            _log("INVEST", "investment scoring returned no scores; using QA scores only")

    # Ensure all dimensions exist, clean text, deduplicate bullets, and cap lengths.
    clean_start = time.perf_counter()
    cleaned_dims: Dict[str, Any] = {}
    metrics_dims = metrics.get("dimensions", {}) or {}
    _write_progress(0, len(DIM_ORDER), pid)
    for dim_idx, dim in enumerate(DIM_ORDER):
        blk = dim_op_raw.get(dim, {}) or {}
        m = metrics_dims.get(dim, {}) or {}

        strengths = dedup_soft(clean_list(blk.get("strengths") or []))[:3]
        concerns = dedup_soft(clean_list(blk.get("concerns") or []))[:3]
        recs = dedup_soft(clean_list(blk.get("recommendations") or []))[:4]
        summary = clean_text(blk.get("summary") or "")

        # If a dimension has almost no information, add a conservative fallback summary.
        if not summary and not strengths and not concerns:
            label = DIM_LABELS.get(dim, dim)
            summary = (
                f"The current effective QA and evidence signals for the {label} dimension are very limited. "
                "The conclusion is unstable, and the project team should add core facts, quantitative metrics, "
                "and implementation details in the next version."
            )

        qa_score = float(m.get("avg", 0.0) or 0.0)
        if dim in invest_scores:
            blended = INVEST_WEIGHT * invest_scores[dim] + (1 - INVEST_WEIGHT) * qa_score
        else:
            blended = qa_score

        cleaned_dims[dim] = {
            "score_echo": round(blended, 4),
            "qa_score_echo": round(qa_score, 4),
            "invest_score_echo": round(invest_scores.get(dim, qa_score), 4),
            "alignment_echo": float(m.get("avg_alignment", 0.0) or 0.0),
            "drift_echo": float(m.get("avg_drift", 0.0) or 0.0),
            "summary": summary,
            "strengths": strengths,
            "concerns": concerns,
            "recommendations": recs
        }
        _log(
            "DIM_OUTPUT",
            f"{dim}: summary_chars={len(summary)} strengths={len(strengths)} concerns={len(concerns)} "
            f"recommendations={len(recs)} score={blended:.3f} (qa={qa_score:.3f} invest={invest_scores.get(dim, qa_score):.3f})"
        )
        _write_progress(dim_idx + 1, len(DIM_ORDER), pid)
    _log("TIMING", f"clean_dimension_blocks elapsed_sec={time.perf_counter() - clean_start:.3f}")

    # Build blended metrics_dims so overall verdict reflects investment scores.
    blended_metrics_dims: Dict[str, Any] = {}
    for dim in DIM_ORDER:
        orig = metrics_dims.get(dim, {}) or {}
        blended_metrics_dims[dim] = dict(orig)
        blended_metrics_dims[dim]["avg"] = cleaned_dims.get(dim, {}).get("score_echo", orig.get("avg", 0.0))

    # Compute blended overall score (dimension-weighted average of blended dim scores).
    dim_weights = (metrics.get("config_used") or {}).get("dimension_weight") or {}
    blended_dim_scores = [cleaned_dims[d]["score_echo"] for d in DIM_ORDER if d in cleaned_dims]
    blended_weights = [float(dim_weights.get(d, 1.0)) for d in DIM_ORDER if d in cleaned_dims]
    blended_overall = (sum(s * w for s, w in zip(blended_dim_scores, blended_weights)) /
                       max(1e-9, sum(blended_weights))) if blended_dim_scores else 0.0
    blended_metrics_overall = dict(metrics.get("overall", {}) or {})
    blended_metrics_overall["overall_score"] = round(blended_overall, 4)

    overall_start = time.perf_counter()
    overall_block = build_overall_from_dims(
        dim_blocks=cleaned_dims,
        metrics_overall=blended_metrics_overall,
        metrics_dims=blended_metrics_dims
    )
    _log("TIMING", f"build_overall_from_dims elapsed_sec={time.perf_counter() - overall_start:.3f}")

    elapsed = time.perf_counter() - stage_start

    # Build final JSON.
    opinion: Dict[str, Any] = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "model": used_model,
            "mode": used_mode,
            "provider": provider,
            "elapsed_sec": round(elapsed, 3),
            "sources": {
                "metrics_path": str(metrics_path),
                "final_payload_path": str(payload_path)
            }
        },
        "overall_opinion": overall_block,
        "dimensions": cleaned_dims,
        "scoring_explainer": {
            # Echo only the key scoring knobs needed to explain how the report relates to Stage 5.
            "consistency_weight": float((metrics.get("config_used") or {}).get("consistency_weight", 0.20) or 0.20),
            "dimension_weight": {
                k: float(v) for k, v in ((metrics.get("config_used") or {}).get("dimension_weight") or {}).items()
            }
        }
    }

    write_start = time.perf_counter()
    write_json(json_path, opinion)
    if not args.no_markdown:
        try:
            md_text = render_markdown(opinion)
            md_path.write_text(md_text, encoding="utf-8")
        except Exception as _e:
            _log("WARN", f"render_markdown failed, .md not written: {_e}")
    _log("TIMING", f"write_outputs elapsed_sec={time.perf_counter() - write_start:.3f}")

    audit = {
        "stage": "stage_6_ai_expert_opinion",
        "purpose": "Generate an expert-style review from Stage 5 metrics and selected answers.",
        "proposal_id": pid,
        "started_at": opinion["meta"]["generated_at"],
        "elapsed_sec": round(elapsed, 3),
        "provider": provider,
        "model": used_model,
        "mode": used_mode,
        "llm_attempted": llm_attempted,
        "fallback_reason": fallback_reason,
        "inputs": {
            "metrics_path": str(metrics_path),
            "metrics_size_bytes": metrics_path.stat().st_size,
            "final_payload_path": str(payload_path),
            "final_payload_size_bytes": payload_path.stat().st_size,
        },
        "outputs": {
            "prompt_path": str(prompt_path),
            "json_path": str(json_path),
            "markdown_path": "" if args.no_markdown else str(md_path),
            "audit_path": str(audit_path),
        },
        "dimension_summary": {
            dim: {
                "score_echo": cleaned_dims.get(dim, {}).get("score_echo", 0.0),
                "summary_chars": len(cleaned_dims.get(dim, {}).get("summary", "")),
                "strengths": len(cleaned_dims.get(dim, {}).get("strengths", []) or []),
                "concerns": len(cleaned_dims.get(dim, {}).get("concerns", []) or []),
                "recommendations": len(cleaned_dims.get(dim, {}).get("recommendations", []) or []),
                "qa_samples_supplied": len(dim_inputs.get(dim, {}).get("qa_samples", []) or []),
                "general_insights_supplied": len(dim_inputs.get(dim, {}).get("dim_general_insights", []) or []),
            }
            for dim in DIM_ORDER
        },
        "overall": {
            "verdict": overall_block.get("verdict", ""),
            "overall_score_echo": overall_block.get("overall_score_echo", 0.0),
            "confidence_echo": overall_block.get("confidence_echo", 0.0),
            "key_strengths": len(overall_block.get("key_strengths", []) or []),
            "key_risks": len(overall_block.get("key_risks", []) or []),
            "recommendations": len(overall_block.get("recommendations", []) or []),
        },
    }
    write_json(audit_path, audit)

    _log("OUTPUT", f"expert_output_dir={out_dir}")
    _log("OK", f"ai_expert_opinion.json written path={json_path}")
    if not args.no_markdown:
        _log("OK", f"ai_expert_opinion.md written path={md_path}")
    _log("OK", f"stage6_ai_expert_opinion_audit.json written path={audit_path}")
    _log(
        "SUMMARY",
        f"mode={used_mode} llm_attempted={llm_attempted} verdict={overall_block.get('verdict', '')} "
        f"overall_score={overall_block.get('overall_score_echo', 0.0):.3f} "
        f"confidence={overall_block.get('confidence_echo', 0.0):.3f}"
    )
    _log("STAGE_6", f"Completed Stage 6 elapsed_sec={time.perf_counter() - stage_start:.3f}")


if __name__ == "__main__":
    main()
