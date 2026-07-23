# -*- coding: utf-8 -*-
"""
Stage 4 - Proposal-aware LLM answering (ChatGPT + DeepSeek, no web search)

Main responsibilities:
- Use dimension facts produced by the extraction pipeline plus the generated question set.
- Generate structured, proposal-grounded answers for every question in every dimension.
- Do not rely on external web search. Use only:
    - proposal dimension facts such as summary/key_points/risks/mitigations/numbers
    - general domain knowledge for explanation, without inventing new experiments,
      numbers, organizations, registrations, or evidence.
- Keep output compatibility:
    data/refined_answers/{pid}/all_refined_items.json
    data/refined_answers/{pid}/chatgpt_raw.json
    data/refined_answers/{pid}/deepseek_raw.json

This version asks each answer to include baseline comparison and common evidence
requirements as general expert context, clearly separated from proposal facts.
"""

import os
import sys
import json
import time
import re
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.llm_provider import (
    UnifiedChatClient,
    active_provider,
    configured_providers,
    default_model,
    provider_api_key,
)

from prompt_templates import self_consistency_section, cot_reasoning_section, react_reasoning_section

# ========== Environment and paths ==========
load_dotenv()
ROOT = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT / "data"
PROGRESS_FILE = DATA_DIR / "step_progress.json"


def _write_llm_progress(done: int, total: int, pid: str = "") -> None:
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass
EXTRACTED_DIR = DATA_DIR / "extracted"
PARSED_DIR = DATA_DIR / "parsed"
CONFIG_QS_DEFAULT = DATA_DIR / "config" / "question_sets" / "generated_questions.json"
OUT_REFINED = DATA_DIR / "refined_answers"

DIM_ORDER = ["team", "objectives", "strategy", "innovation", "feasibility"]

# ========== SDK ==========
try:
    from openai import OpenAI as OpenAIClient
except Exception:  # pragma: no cover
    OpenAIClient = None

# ========== Variants and provider capabilities ==========
VARIANTS = ["default", "risk", "implementation"]
TEMP_BY_VARIANT = {"default": 0.25, "risk": 0.35, "implementation": 0.30}

PROVIDER_CAPS = {
    "openai": {"json_mode": True, "batch_ok": True},
    "deepseek": {"json_mode": False, "batch_ok": False},
    "gemini": {"json_mode": True, "batch_ok": True},
}

ACTIVE_PROVIDER = active_provider()
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
STAGE4_DIM_WORKERS = int(os.getenv("STAGE4_DIM_WORKERS", "2"))
STAGE4_VARIANT_WORKERS = int(os.getenv("STAGE4_VARIANT_WORKERS", "3"))
STAGE4_REFINE_WORKERS = int(os.getenv("STAGE4_REFINE_WORKERS", "3"))


def provider_caps(provider: str) -> Dict[str, Any]:
    return PROVIDER_CAPS.get(provider, {"json_mode": True, "batch_ok": True})


CONF_MIN, CONF_MAX = 0.40, 0.92
MAX_LIST_LEN = 10
LLM_TIMING: Dict[str, Dict[str, float]] = {}


def detect_proposal_language(pid: str) -> str:
    """Return 'zh' if >30% of proposal text chars are CJK, else 'en'."""
    txt_path = DATA_DIR / "prepared" / pid / "full_text.txt"
    if not txt_path.exists():
        return "en"
    sample = txt_path.read_text(encoding="utf-8", errors="ignore")[:3000]
    if not sample:
        return "en"
    cjk = sum(1 for c in sample if "一" <= c <= "鿿")
    return "zh" if cjk / max(1, len(sample)) > 0.15 else "en"
LLM_TIMING_LOCK = Lock()

# ========== Utility functions ==========
def read_json(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def detect_latest_pid() -> str:
    if not EXTRACTED_DIR.exists():
        return "unknown"
    cands = [d for d in EXTRACTED_DIR.iterdir() if d.is_dir()]
    if not cands:
        return "unknown"
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0].name


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def _preview_text(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return "<empty>"
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def _record_timing(name: str, elapsed_sec: float) -> None:
    with LLM_TIMING_LOCK:
        bucket = LLM_TIMING.setdefault(name, {"count": 0.0, "total_sec": 0.0, "max_sec": 0.0})
        bucket["count"] += 1
        bucket["total_sec"] += float(elapsed_sec)
        bucket["max_sec"] = max(bucket["max_sec"], float(elapsed_sec))


def _timing_summary() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    with LLM_TIMING_LOCK:
        snapshot = {name: dict(bucket) for name, bucket in LLM_TIMING.items()}
    for name, bucket in snapshot.items():
        count = int(bucket.get("count", 0))
        total = float(bucket.get("total_sec", 0.0))
        out[name] = {
            "count": count,
            "total_sec": round(total, 2),
            "avg_sec": round(total / count, 2) if count else 0.0,
            "max_sec": round(float(bucket.get("max_sec", 0.0)), 2),
        }
    return out


def _bounded_worker_count(value: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = minimum
    return max(minimum, min(parsed, maximum))

def _flatten_list_field(block: Dict[str, Any], keys: List[str], limit: int = 12) -> List[str]:
    items: List[str] = []
    for k in keys:
        v = block.get(k)
        if not v:
            continue
        if isinstance(v, str):
            items.extend([x.strip() for x in re.split(r"[;\n]", v) if x.strip()])
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    s = x.strip()
                    if s:
                        items.append(s)
    uniq, seen = [], set()
    for x in items:
        key = x.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)
        if len(uniq) >= limit:
            break
    return uniq


def _build_dim_context_text(dim: str, block: Dict[str, Any]) -> str:
    """
    Convert one dimension block from the facts pipeline into prompt-ready context.
    Keep the context compact and prioritize summary, key_points, risks, mitigations, and numbers.
    """
    if not isinstance(block, dict):
        block = {}

    summary = str(block.get("summary") or "").strip()
    key_points = _flatten_list_field(block, ["key_points", "keypoints", "key_facts", "bullets"], limit=10)
    risks = _flatten_list_field(block, ["risks", "risk_points"], limit=8)
    mitigations = _flatten_list_field(block, ["mitigations", "mitigation_points"], limit=8)
    numbers = _flatten_list_field(block, ["numbers", "key_numbers"], limit=8)

    parts: List[str] = []
    if summary:
        parts.append(f"[{dim} overview] {summary}")
    if key_points:
        parts.append("[key points] " + "; ".join(key_points))
    if risks:
        parts.append("[main risks or uncertainties] " + "; ".join(risks))
    if mitigations:
        parts.append("[existing mitigations] " + "; ".join(mitigations))
    if numbers:
        parts.append("[key quantitative information] " + "; ".join(numbers))

    text = "\n".join(parts)
    if len(text) > 2000:
        text = text[:2000]
    return text


def load_dimension_context(pid: str, dim_file: Optional[Path]) -> Dict[str, str]:
    """
    Return {dim: context_text}.
    If the file is missing or malformed, return empty context for every dimension.
    """
    ctx: Dict[str, str] = {d: "" for d in DIM_ORDER}
    if dim_file is None or not dim_file.exists():
        return ctx

    try:
        raw = read_json(dim_file)
    except Exception:
        return ctx

    if isinstance(raw, dict) and "dimensions" in raw and isinstance(raw["dimensions"], dict):
        root = raw["dimensions"]
    else:
        root = raw if isinstance(raw, dict) else {}

    for dim in DIM_ORDER:
        block = root.get(dim) or {}
        ctx[dim] = _build_dim_context_text(dim, block)
    return ctx


# ========== Question set and optional regulatory hints ==========
def get_q_list(block: Any) -> List[str]:
    if isinstance(block, dict) and isinstance(block.get("questions"), list):
        return [q for q in block["questions"] if isinstance(q, str)]
    if isinstance(block, list):
        return [q for q in block if isinstance(q, str)]
    return []


def _load_reg_hints(qs_cfg: Dict[str, Any], dim: str, limit: int = 8) -> List[str]:
    """
    Pull a few regulatory/terminology hints from the question set as optional guidance.
    """
    block = qs_cfg.get(dim, {}) or {}
    hints = block.get("search_hints") or block.get("reg_hints") or []
    out: List[str] = []
    if isinstance(hints, list):
        for h in hints:
            if not isinstance(h, str):
                continue
            s = h.strip()
            if not s:
                continue
            out.append(s)
    uniq, seen = [], set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
        if len(uniq) >= limit:
            break
    return uniq

# ========== Prompt construction ==========
SYSTEM_EN = (
    "You are a rigorous sector-agnostic venture, business, product, technical, operating, and financial diligence expert. "
    "Your task is to read the provided proposal facts and answer each question with analysis that is strongly tied to this proposal and useful for startup, investment, grant, procurement, partnership, launch, scale-up, and operating-readiness diligence. "
    "Principles: "
    "1. First infer the business or operating archetype from the proposal facts, then choose the right diligence standard for that archetype. "
    "Do not use the same evidence standard for a SaaS product, professional service, consumer brand, public program, manufacturing operation, research commercialization project, infrastructure asset, and internal transformation initiative. "
    "2. Base conclusions primarily on the proposal facts. "
    "3. Do not invent experiments, data, organizations, registrations, studies, customers, contracts, approvals, pilots, or specific numbers not present in the proposal. "
    "4. When a category of information is not visible in the proposal facts, state that the current materials do not show specific details about it. "
    "Prefer precise information-gap wording such as 'the proposal states A/B, but still lacks detail on C/D'; avoid absolute claims such as "
    "'there is no analysis' or 'no information is provided'. "
    "5. You may use general industry knowledge to explain significance, but do not fabricate certification IDs, contract IDs, registration IDs, patent IDs, sample sizes, customer data, operating data, production data, impact data, or other concrete evidence. "
    "6. Answers must be structured and suitable for an expert review report. Each answer should separate what the proposal already supports from what still needs proof before scale-up, financing, launch, procurement, partnership, or operating commitment. "
    "7. Include baseline comparison and common evidence requirements as general expert context, but clearly label them as general industry practice or recommendations, not facts achieved by the project. "
    "8. Do not assume any sector, customer type, business model, regulated pathway, or delivery model unless the proposal facts support it; use validation, quality, customer qualification, operating scaling, adoption, security, procurement, funding, implementation, impact, or compliance language when that better fits the project."
)

BUSINESS_ADAPTATION_FRAMEWORK = """
Business adaptation framework:
- Silently classify the proposal into the closest archetype or mixed archetype: SaaS/software, AI/data product,
  marketplace, consumer brand/product, B2B service, professional services, education/training, healthcare/life science,
  industrial/manufacturing, hardware/device, energy/climate, logistics/supply-chain, fintech/financial service,
  real estate/infrastructure, agriculture/food, media/content, nonprofit/public-sector program, research commercialization,
  internal corporate transformation, or other.
- Tailor evidence requirements to that archetype:
  - Software/platform: activation, retention, security, data rights, integration, support burden, cloud/unit costs.
  - Service/business operations: repeatability, delivery capacity, staffing model, service quality, utilization, margin, customer references.
  - Consumer/brand: customer insight, channel economics, brand differentiation, repeat purchase, distribution, working capital.
  - Marketplace/network: liquidity, supply-demand acquisition, take rate, trust/safety, disintermediation risk.
  - Manufacturing/hardware: bill of materials, yield, reliability, certification, supply chain, inventory, warranty.
  - Regulated product/program: approval pathway, safety, compliance, quality system, reimbursement/procurement when relevant.
  - Infrastructure/real estate: asset control, permits, capex, utilization, financing, operating partner, location risk.
  - Public/nonprofit/impact: beneficiaries, theory of change, implementation partners, funding durability, measurable outcomes.
  - Research commercialization: translational path, IP rights, validation milestones, partner route, capital intensity.
- If the proposal does not reveal the archetype clearly, say so and keep recommendations modular rather than forcing a sector template.
"""

def _schema_structured() -> str:
    return """
Return exactly one JSON object with these fixed keys:
{
  "answer": "Main answer as numbered English bullets, covering project status, issues, and improvement directions",
  "claims": ["Key verifiable conclusion 1", "Key verifiable conclusion 2", "..."],
  "evidence_hints": ["Proposal fact, section, or supporting material that supports the conclusion, or a follow-up evidence trail to check"],
  "general_insights": ["Baseline practices, common pitfalls, and evidence requirements for similar projects, written as general context rather than project facts"],
  "topic_tags": ["Topic tags within the dimension"],
  "confidence": 0.0,
  "caveats": "Limitations or cautions, or empty if none"
}
""".strip()


def _schema_batch_structured() -> str:
    return """
Return exactly one JSON object in this batch shape:
{
  "answers": [
    {
      "answer": "Main answer as numbered English bullets, covering project status, issues, and improvement directions",
      "claims": ["Key verifiable conclusion 1", "Key verifiable conclusion 2", "..."],
      "evidence_hints": ["Proposal fact, section, or supporting material that supports the conclusion, or a follow-up evidence trail to check"],
      "general_insights": ["Baseline practices, common pitfalls, and evidence requirements for similar projects, written as general context rather than project facts"],
      "topic_tags": ["Topic tags within the dimension"],
      "confidence": 0.0,
      "caveats": "Limitations or cautions, or empty if none"
    }
  ]
}
The answers array length must exactly match the question list length and preserve question order.
""".strip()


def _variant_instructions(variant_id: str) -> str:
    if variant_id == "risk":
        return (
            "Variant: risk perspective.\n"
            "- Prioritize uncertainties, missing information, and risks that fit the inferred business archetype.\n"
            "- For each major risk, explain the risk source, possible impact, and recommended supporting materials.\n"
            "- If proposal facts do not cover a key step, explicitly identify the information gap.\n"
            "- In general_insights, summarize common risk patterns, quality/compliance/regulatory/security/procurement concerns when applicable, and evidence requirements for similar initiatives, "
            "clearly stating that these are general industry considerations rather than confirmed project achievements.\n"
        )
    if variant_id == "implementation":
        return (
            "Variant: implementation perspective.\n"
            "- List 4-8 concrete actions that could be executed over the next 3-12 months, each with actor/object, specific step, and expected output.\n"
            "- Actions must match the proposal's current state and must not assume completed work.\n"
            "- You may mention evidence or document types that need to be collected, rather than inventing conclusions.\n"
            "- In general_insights, summarize common implementation paths, milestone breakdowns, and pitfalls for similar projects.\n"
        )
    return (
        "Variant: default integrated perspective.\n"
        "- Evaluate the dimension by combining the current approach, strengths, issues, and recommendations.\n"
        "- Mention both strengths and limitations or uncertainties.\n"
        "- Cover at least four content types: current status, strengths, main issues, and improvement directions.\n"
        "- Where relevant, name the diligence evidence that would materially change the decision: customer/user discovery, buyer/payer/funder proof, pilots or LOIs, retention/adoption data, benchmark data, service quality metrics, unit economics, operating capacity, supply-chain readiness, IP/brand/data defensibility, use-of-funds, risk register, owner-level milestones, or compliance/procurement path.\n"
        "- In answer, clearly distinguish: "
        "a) project status and issues based on proposal facts; "
        "b) baseline comparison for what similar projects generally need; "
        "c) common pitfalls and evidence, quality, compliance, regulatory, or adoption requirements for similar projects.\n"
        "- In claims, prioritize what the proposal has stated and where information remains limited. Avoid absolute negatives; if detail is limited, say the current materials do not show more detailed information about X.\n"
        "- In general_insights, summarize b) and c) as general industry context, using wording such as 'generally', 'typically', or 'in similar projects' to avoid implying project achievement.\n"
    )

def build_single_prompt(
    dimension: str,
    question: str,
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
    lang: str = "en",
) -> str:
    reg_txt = "; ".join(reg_hints) if reg_hints else "No special hints."
    return f"""
Dimension: {dimension}

{BUSINESS_ADAPTATION_FRAMEWORK.strip()}

[Proposal facts]
Use only this section for proposal-specific details. If relevant information is absent, say so precisely:
{proposal_context or "The proposal facts for this dimension are empty; only general analysis is allowed."}

[Optional regulatory or terminology hints]
Use only if relevant:
{reg_txt}

Question:
{question.strip()}

{_variant_instructions(variant_id)}

{react_reasoning_section()}

Answer requirements:
- Language: {'Chinese' if lang == 'zh' else 'English'}.
- Structure: answer must contain 3-8 numbered bullets, using "1. 2. 3." style. Keep each bullet to one or two sentences where possible.
- Suggested content coverage:
  1. Start from the proposal facts and summarize current status, strengths, and main issues for this dimension.
  2. Use the inferred archetype to choose comparable baseline expectations. Clearly mark this as general industry context, not project achievement.
  3. Add 1-3 bullets about common pitfalls, evidence requirements, quality/compliance/security/procurement concerns when applicable, adoption concerns, operating constraints, or economics for similar initiatives.
- Relevance: analyze the project primarily from proposal facts. For information not present in the facts, describe it as an information gap or required follow-up; do not assume it exists.
- If the proposal already provides partial information, write in a nuanced way such as "the proposal provides A/B, but lacks detail on C/D." Do not use broad absolute negatives.
- Separate general knowledge from project facts. When using industry experience, use wording such as "typically", "generally", or "in similar projects".
- general_insights: list 3-8 baseline, pitfall, or evidence-requirement points that do not depend on project-specific facts. The final item should be a general disclaimer such as "These are general industry considerations and do not mean the project has already met these requirements."
- Be cautious and evidence-oriented. Avoid overly absolute conclusions, and name the supporting material types that would be needed.
- Output only one JSON object. Do not add extra text or Markdown fences.

{_schema_structured()}
""".strip()


def build_batch_prompt(
    dimension: str,
    questions: List[str],
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
    lang: str = "en",
) -> str:
    reg_txt = "; ".join(reg_hints) if reg_hints else "No special hints."
    q_block = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    return f"""
You will answer multiple questions for the same dimension using the same proposal facts.
Return exactly one JSON object in this shape: {{"answers":[<object1>,<object2>,...]}}.
The answers array length must match the number of questions.

Dimension: {dimension}

{BUSINESS_ADAPTATION_FRAMEWORK.strip()}

[Proposal facts]
Use only this section for proposal-specific details. If relevant information is absent, say so precisely:
{proposal_context or "The proposal facts for this dimension are empty; only general analysis is allowed."}

[Optional regulatory or terminology hints]
Use only if relevant:
{reg_txt}

Question list:
{q_block}

{_variant_instructions(variant_id)}

Unified answer requirements:
- Language: {'Chinese' if lang == 'zh' else 'English'}.
- Each answer: 3-8 numbered bullets. If there is limited information, write 3-4 solid bullets rather than filler.
- In answer, explicitly cover:
  a) project status, strengths, and issues based only on proposal facts;
  b) baseline comparison for the inferred business archetype, clearly marked as general industry context;
  c) common pitfalls, evidence requirements, quality/compliance/security/procurement concerns when applicable, adoption concerns, operating constraints, or economics for similar initiatives.
- claims: 2-6 verifiable conclusions, primarily based on proposal facts. If information is insufficient, include the information gap itself as a claim.
- evidence_hints: identify what type of proposal fact, section, or supporting document could support the conclusion. Avoid empty templates.
- general_insights: 3-8 baseline, common-pitfall, or evidence-requirement points that are independent of project-specific facts. Write them as general recommendations only.
- If the proposal provides partial information, evaluate it as partial detail rather than an absolute absence.
- Do not invent new experiments, data, organizations, registration IDs, contracts, customers, pilots, revenue, approvals, or exact details not present in the proposal.
- Output only one JSON object. Do not add extra text or Markdown fences.

{_schema_batch_structured()}
""".strip()


def build_refine_prompt(candidate_obj: Dict[str, Any], proposal_context: str, dimension: str, lang: str = "en") -> str:
    original = json.dumps(candidate_obj, ensure_ascii=False, indent=2)
    return f"""
Review the structured answer below without introducing any information beyond the proposal facts:
- Remove or soften overly strong conclusions that lack support.
- If a conclusion is not supported by proposal facts, rewrite it as material or information that needs to be supplemented.
- Improve the numbered answer bullets so each item is more specific and actionable, without inventing new experiments or data.
- Check general_insights: it must contain only baseline practices, common approaches, pitfalls, or evidence requirements for similar initiatives with the same inferred business archetype, not project-specific facts.
- Keep the JSON structure and field names exactly unchanged, including general_insights.
- Preserve the original response language ({'Chinese' if lang == 'zh' else 'English'}).

Dimension: {dimension}

[Proposal facts]:
{proposal_context or "The proposal facts for this dimension are empty; only light general cleanup is allowed."}

Original candidate:
{original}
""".strip()


# ========== LLM call basics ==========
def _error_text(exc: Any) -> str:
    return str(exc or "").lower()


def is_permanent_provider_error(exc: Any) -> bool:
    """Return True for auth/permission errors that should not be retried."""
    text = _error_text(exc)
    permanent_markers = [
        "401",
        "403",
        "authentication",
        "authenticates",
        "invalid api key",
        "invalid_api_key",
        "incorrect api key",
        "permission denied",
        "access denied",
        "unauthorized",
        "forbidden",
    ]
    return any(marker in text for marker in permanent_markers)


def _with_retry(fn, max_tries: int = 4, base: float = 0.8):
    for i in range(max_tries):
        try:
            return fn()
        except Exception as e:
            if is_permanent_provider_error(e):
                raise
            if i == max_tries - 1:
                raise
            time.sleep(base * (2 ** i) + random.random() * 0.2)


ERR_PATTERNS = [
    r"\[?ERROR[:\]]",
    r"\bHTTP\s*4\d{2}\b",
    r"\bHTTP\s*5\d{2}\b",
    r"insufficient[_\s-]?quota",
    r"invalid[_\s-]?api[_\s-]?key",
    r"request\s+timed\s*out",
    r"rate\s*limit",
    r"payment\s*required",
    r"bad gateway",
    r"service unavailable",
    r"connection (?:reset|refused)",
]
_err_re = re.compile("|".join(ERR_PATTERNS), re.I)


def is_error_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return bool(_err_re.search(t))


def _chat_completion_json(
    client,
    model: str,
    system_text: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
    force_json: bool,
):
    call_start = time.perf_counter()
    user_chars = len(user_text or "")
    system_chars = len(system_text or "")
    _log(
        "LLM_CALL",
        f"start model={model} force_json={force_json} max_tokens={max_tokens} "
        f"temperature={temperature} system_chars={system_chars} user_chars={user_chars}"
    )

    def call(json_mode: bool):
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    try:
        if force_json:
            resp = _with_retry(lambda: call(json_mode=True))
            out = (resp.choices[0].message.content or "").strip()
            if is_error_text(out):
                raise RuntimeError("provider_error_json_mode")
            elapsed = time.perf_counter() - call_start
            _record_timing("chat_completion", elapsed)
            _record_timing("chat_completion_json_mode", elapsed)
            _log(
                "LLM_CALL",
                f"done model={model} mode=json elapsed_sec={elapsed:.2f} raw_chars={len(out)}"
            )
            return out
        else:
            resp = _with_retry(lambda: call(json_mode=False))
            out = (resp.choices[0].message.content or "").strip()
            if is_error_text(out):
                raise RuntimeError("provider_error_text_mode")
            elapsed = time.perf_counter() - call_start
            _record_timing("chat_completion", elapsed)
            _record_timing("chat_completion_text_mode", elapsed)
            _log(
                "LLM_CALL",
                f"done model={model} mode=text elapsed_sec={elapsed:.2f} raw_chars={len(out)}"
            )
            return out
    except Exception as first_error:
        if force_json:
            _log("LLM_CALL", f"json_mode_failed model={model}; retrying text mode error={first_error}")
            resp = _with_retry(lambda: call(json_mode=False))
            out = (resp.choices[0].message.content or "").strip()
            if is_error_text(out):
                raise RuntimeError("provider_error_text_mode")
            elapsed = time.perf_counter() - call_start
            _record_timing("chat_completion", elapsed)
            _record_timing("chat_completion_json_fallback_to_text", elapsed)
            _log(
                "LLM_CALL",
                f"done model={model} mode=json_fallback_to_text elapsed_sec={elapsed:.2f} raw_chars={len(out)}"
            )
            return out
        elapsed = time.perf_counter() - call_start
        _record_timing("chat_completion_failed", elapsed)
        _log("LLM_CALL", f"failed model={model} elapsed_sec={elapsed:.2f} error={first_error}")
        raise


def _safe_parse_json_plus(txt: str) -> Optional[Any]:
    if is_error_text(txt):
        return None

    t = (txt or "").replace("\ufeff", "").strip()
    t = re.sub(r"^\s*```(?:json)?\s*\n?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\n?\s*```\s*$", "", t, flags=re.IGNORECASE)
    t = t.replace("\xa0", " ")

    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return {"answers": obj}
        return obj
    except Exception:
        pass

    m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
    if m:
        frag = m.group(1)
        try:
            tmp = json.loads(frag)
            if isinstance(tmp, list):
                return {"answers": tmp}
            return tmp
        except Exception:
            pass

    # Try adding double quotes around common field names, including general_insights.
    cand = re.sub(
        r"(\banswer|claims|evidence_hints|general_insights|topic_tags|confidence|caveats\b)\s*:",
        r'"\1":',
        t,
    )
    try:
        return json.loads(cand)
    except Exception:
        return None


# ========== Model initialization ==========
def _looks_like_placeholder_key(key: str) -> bool:
    key_l = (key or "").strip().lower()
    if not key_l:
        return True
    placeholder_fragments = [
        "your_",
        "replace",
        "placeholder",
        "example",
        "changeme",
        "here",
        "todo",
        "xxx",
    ]
    return any(fragment in key_l for fragment in placeholder_fragments)


def init_openai():
    if OpenAIClient is None:
        return None, None
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None, None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    client = OpenAIClient(api_key=key, timeout=OPENAI_TIMEOUT_SECONDS)
    _log("OK", f"Loaded OpenAI model: {model}")
    return client, model


def init_gemini():
    key = provider_api_key("gemini")
    if not key:
        return None, None
    model = default_model("gemini")
    client = UnifiedChatClient(provider="gemini", model=model)
    _log("OK", f"Loaded Gemini model: {model}")
    return client, model


def init_deepseek():
    if OpenAIClient is None:
        return None, None
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None, None
    if _looks_like_placeholder_key(key):
        _log("WARN", "Skipping DeepSeek because DEEPSEEK_API_KEY looks like a placeholder.")
        return None, None
    base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
    client = OpenAIClient(api_key=key, base_url=base, timeout=OPENAI_TIMEOUT_SECONDS)
    _log("OK", f"Loaded DeepSeek model: {model}")
    return client, model


def init_provider(provider: str):
    provider = (provider or "").strip().lower()
    if provider == "openai":
        return init_openai()
    if provider == "gemini":
        return init_gemini()
    if provider == "deepseek":
        return init_deepseek()
    return None, None


def provider_raw_filename(provider: str) -> str:
    if provider == "openai":
        return "chatgpt_raw.json"
    return f"{provider}_raw.json"


def provider_display_name(provider: str) -> str:
    return {
        "openai": "OpenAI",
        "gemini": "Gemini",
        "deepseek": "DeepSeek",
    }.get(provider, provider.title())


def parse_provider_list(value: str) -> List[str]:
    value = (value or "auto").strip().lower()
    if value in {"", "auto", "all"}:
        return configured_providers()
    providers = [p.strip().lower() for p in value.split(",") if p.strip()]
    if "auto" in providers or "all" in providers:
        return configured_providers()
    return providers


def validate_chat_provider(provider: str, client, model: str) -> tuple[bool, str]:
    """
    Make one cheap health-check call before the expensive answer loop.
    This prevents wasting many calls on permanent auth/model-access failures.
    """
    if client is None or not model:
        return False, "client_or_model_missing"
    start = time.perf_counter()
    _log("PROVIDER_CHECK", f"start provider={provider} model={model}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return the single word OK."},
                {"role": "user", "content": "Health check."},
            ],
            temperature=0,
            max_tokens=2,
        )
        content = (resp.choices[0].message.content or "").strip()
        _log(
            "PROVIDER_CHECK",
            f"ok provider={provider} model={model} elapsed_sec={time.perf_counter() - start:.2f} raw={content[:20]!r}"
        )
        return True, "ok"
    except Exception as exc:
        reason = str(exc)
        level = "ERROR" if is_permanent_provider_error(exc) else "WARN"
        _log(
            "PROVIDER_CHECK",
            f"{level.lower()} provider={provider} model={model} elapsed_sec={time.perf_counter() - start:.2f} error={reason}"
        )
        return False, reason


# ========== Answer normalization and post-processing ==========
def _norm_str(s: Any) -> str:
    return " ".join(str(s or "").strip().split())


def _uniq_cut(lst: List[Any], k: int = MAX_LIST_LEN) -> List[str]:
    seen, out = set(), []
    for x in lst:
        t = _norm_str(x)
        if not t:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= k:
            break
    return out


def _dedupe_casefold(items: List[Any], k: int = MAX_LIST_LEN) -> List[str]:
    seen, out = set(), []
    for x in items:
        t = _norm_str(x)
        if not t:
            continue
        key = re.sub(r"\W+", " ", t.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= k:
            break
    return out


RE_DATE = re.compile(r"\b(20\d{2}|19\d{2})([-/.])\d{1,2}([-/\.])\d{1,2}\b|\b(Q[1-4]\s*-\s*20\d{2})\b", re.I)
RE_MONEY = re.compile(
    r"\b(\$|USD|EUR|CNY|RMB|CAD)\s*\d{2,}(,\d{3})*(\.\d+)?\b|\b\d+(\.\d+)?\s*(million|billion)\b",
    re.I,
)
RE_TRIAL = re.compile(r"\bNCT\d{8}\b|\bEUCTR-\d{4}-\d{6}-\d{2}\b", re.I)
RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
RE_PATENT = re.compile(r"\b(US|EP|CN)\d{5,}\b|\bWO\d{7,}\b", re.I)
RE_ISO = re.compile(r"\bISO\s?\d{4,5}(-\d+)?\b", re.I)
RE_STDNUM = re.compile(r"\bEN\s?\d{3,5}\b|\bASTM\s?[A-Z]?\d{2,5}\b", re.I)
RE_ID_ANY = re.compile(r"\b(registration|approval|filing)\s+(number|id)\b", re.I)


def _is_redline(text: str) -> bool:
    t = text or ""
    if RE_DATE.search(t):
        return True
    if RE_MONEY.search(t):
        return True
    if RE_TRIAL.search(t):
        return True
    if RE_DOI.search(t):
        return True
    if RE_PATENT.search(t):
        return True
    if RE_ISO.search(t):
        return True
    if RE_STDNUM.search(t):
        return True
    if RE_ID_ANY.search(t):
        return True
    return False


def _scrub_claims_to_hints(claims: List[Any], hints: List[Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Move sentences with concrete IDs or monetary amounts from claims to evidence_hints.
    This only cleans and reclassifies content; it does not add filler templates.
    """
    new_claims: List[str] = []
    new_hints: List[str] = [str(x).strip() for x in (hints or []) if str(x).strip()]
    moved_facts: List[str] = []

    for c in claims or []:
        s = _norm_str(c)
        if not s:
            continue
        if _is_redline(s):
            new_hints.append(s)
            moved_facts.append(s)
        else:
            new_claims.append(s)

    return _dedupe_casefold(new_claims, MAX_LIST_LEN), _dedupe_casefold(new_hints, MAX_LIST_LEN), _dedupe_casefold(moved_facts, MAX_LIST_LEN)


def _scrub_general_insights_to_hints(general_insights: List[Any], hints: List[Any]) -> Tuple[List[str], List[str], List[str]]:
    """
    Keep general_insights free of concrete IDs, dates, money, or other redline-style facts.
    Move those items into evidence_hints so the general_insights field stays truly general.
    """
    new_gi: List[str] = []
    new_hints: List[str] = [str(x).strip() for x in (hints or []) if str(x).strip()]
    moved_facts: List[str] = []

    for item in general_insights or []:
        s = _norm_str(item)
        if not s:
            continue
        if _is_redline(s):
            new_hints.append(s)
            moved_facts.append(s)
        else:
            new_gi.append(s)

    return _dedupe_casefold(new_gi, MAX_LIST_LEN), _dedupe_casefold(new_hints, MAX_LIST_LEN), _dedupe_casefold(moved_facts, MAX_LIST_LEN)


def _to_bullets(answer: str) -> Tuple[str, int]:
    """
    Convert free-form text into numbered bullets.
    Existing numbering or bullet markers are removed before applying "1. 2. 3." numbering.
    """
    raw = (answer or "").strip()
    if not raw:
        return "", 0

    text = raw.replace("\r\n", "\n").strip()

    def _normalize_item(line: str) -> str:
        s = line.strip()
        # Remove Markdown bullets.
        s = re.sub(r"^\s*[-*]+\s*", "", s)
        # Remove leading numbering such as 1. or 1).
        s = re.sub(r"^\s*\d+[\.\)]\s*", "", s)
        # Compress whitespace.
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        if not s:
            return ""

        # Drop content that is only numbers.
        s_nospace = s.replace(" ", "")
        if s_nospace.isdigit() and len(s_nospace) <= 4:
            return ""

        return s

    # First split by lines to preserve model structure.
    lines = [ln for ln in text.split("\n") if ln.strip()]
    items: List[str] = []
    for ln in lines:
        norm = _normalize_item(ln)
        if norm:
            items.append(norm)

    # If line splitting produced too little structure, split by sentence punctuation.
    if len(items) <= 1:
        chunks = re.split(r"[;.!?]\s*", text)
        items = []
        for ch in chunks:
            norm = _normalize_item(ch)
            if norm:
                items.append(norm)

    if not items:
        # If no usable split is found, keep the original text.
        return raw, 0

    # Cap bullet count to avoid overly long answers.
    items = items[:8]

    numbered = [f"{i+1}. {seg}" for i, seg in enumerate(items)]
    return "\n".join(numbered), len(items)

def _calibrate_conf(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.65
    if v < CONF_MIN:
        v = CONF_MIN
    if v > CONF_MAX:
        v = CONF_MAX
    return round(v, 2)


def _normalize_candidate_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    out: Dict[str, Any] = {}

    # Preserve original line breaks so model-provided structure is not destroyed.
    raw_answer = obj.get("answer", "")

    # Handle list/dict answer forms so they do not become a single JSON dump.
    if isinstance(raw_answer, list):
        # The model may return nested list-like answer content.
        pieces: List[str] = []
        for elem in raw_answer:
            if elem is None:
                continue
            # Handle an extra nested list in edge cases.
            if isinstance(elem, list):
                for sub in elem:
                    s = str(sub or "").strip()
                    if s:
                        pieces.append(s)
            else:
                s = str(elem or "").strip()
                if s:
                    pieces.append(s)
        raw_answer = "\n".join(pieces)

    elif isinstance(raw_answer, dict):
        # Prefer structured bullet-like fields if the model returns a dict.
        for key in ("bullets", "points", "items"):
            val = raw_answer.get(key)
            if isinstance(val, list):
                pieces = [str(x or "").strip() for x in val if str(x or "").strip()]
                raw_answer = "\n".join(pieces)
                break
        else:
            # Fallback to a JSON string when no structured list exists.
            raw_answer = json.dumps(raw_answer, ensure_ascii=False)

    # Other values are converted directly to strings.
    out["answer"] = str(raw_answer or "")

    raw_claims = obj.get("claims", [])
    raw_hints = obj.get("evidence_hints", [])
    raw_tags = obj.get("topic_tags", [])
    raw_gi = obj.get("general_insights", [])

    if isinstance(raw_claims, (str, int, float)):
        raw_claims = [raw_claims]
    if isinstance(raw_hints, (str, int, float)):
        raw_hints = [raw_hints]
    if isinstance(raw_tags, (str, int, float)):
        raw_tags = [raw_tags]
    if isinstance(raw_gi, (str, int, float)):
        raw_gi = [raw_gi]

    out["claims"] = [str(x).strip() for x in (raw_claims or []) if str(x).strip()]
    out["evidence_hints"] = [str(x).strip() for x in (raw_hints or []) if str(x).strip()]
    out["topic_tags"] = [str(x).strip() for x in (raw_tags or []) if str(x).strip()]
    out["general_insights"] = [str(x).strip() for x in (raw_gi or []) if str(x).strip()]
    try:
        out["confidence"] = float(obj.get("confidence", 0.65))
    except Exception:
        out["confidence"] = 0.65
    out["caveats"] = _norm_str(obj.get("caveats", ""))
    return out


def _validate_candidate_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("answer"), str) or not obj.get("answer").strip():
        return False
    if not isinstance(obj.get("claims"), list):
        return False
    if not isinstance(obj.get("evidence_hints"), list):
        return False
    if not isinstance(obj.get("topic_tags"), list):
        return False
    if not isinstance(obj.get("general_insights"), list):
        return False
    return True


def _build_topic_tags(tags: List[Any], dimension: str, answer: str) -> List[str]:
    base = [str(t).lower().strip() for t in (tags or []) if str(t).strip()]
    extra: List[str] = []

    dim_token = dimension.lower().strip()
    if dim_token:
        extra.append(dim_token)

    words = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]{2,8}", answer)
    freq: Dict[str, int] = {}
    for w in words:
        wl = w.lower()
        if len(wl) < 2:
            continue
        freq[wl] = freq.get(wl, 0) + 1
    common = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:6]
    extra.extend([w for w, _ in common])

    return _dedupe_casefold(base + extra, MAX_LIST_LEN)


def _quick_score(cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a lightweight quality score for later selection:
    - numbered bullet count
    - claim count
    - confidence
    general_insights is kept for downstream use but not scored here.
    """
    bullets = len([ln for ln in str(cand.get("answer", "")).splitlines() if ln.strip()])
    claims_n = len(cand.get("claims") or [])
    conf = float(cand.get("confidence", 0.65))

    score = 0.0
    if bullets >= 3:
        score += 0.25
    if 3 <= bullets <= 8:
        score += 0.20
    if claims_n >= 2:
        score += 0.25
    if claims_n >= 4:
        score += 0.10
    score += max(0.0, min(0.20, (conf - CONF_MIN) / (CONF_MAX - CONF_MIN + 1e-6) * 0.20))

    cand["quick_score"] = round(score, 3)
    return cand


def _finalize_candidate(
    obj: Dict[str, Any],
    provider: str,
    model: str,
    variant_id: str,
    sample_id: int,
    dimension: str,
) -> Dict[str, Any]:
    base = _normalize_candidate_obj(obj)
    answer_bullets, n_bullets = _to_bullets(base["answer"])
    base["answer"] = answer_bullets

    new_claims, new_hints, moved_facts = _scrub_claims_to_hints(base.get("claims", []), base.get("evidence_hints", []))
    base["claims"] = new_claims
    base["evidence_hints"] = new_hints
    base["facts_redlined"] = moved_facts

    # Keep general_insights truly general by pushing concrete redline items into evidence_hints.
    base["general_insights"], base["evidence_hints"], gi_redlined = _scrub_general_insights_to_hints(
        base.get("general_insights", []),
        base.get("evidence_hints", []),
    )
    base["facts_redlined"].extend(gi_redlined)
    base["facts_redlined"] = _uniq_cut(base["facts_redlined"], MAX_LIST_LEN)

    base["topic_tags"] = _build_topic_tags(base.get("topic_tags", []), dimension, base["answer"])
    base["confidence"] = _calibrate_conf(base.get("confidence", 0.65))

    if not base.get("caveats"):
        base["caveats"] = "Conclusions should be checked against the original proposal text and supporting materials."

    base["provider"] = provider
    base["model"] = model
    base["variant_id"] = variant_id
    base["sample_id"] = int(sample_id)
    base["generated_at"] = now_str()

    base["diag"] = {
        "bullet_count": n_bullets,
        "claims_count": len(base["claims"]),
        "hints_count": len(base["evidence_hints"]),
        "general_insights_count": len(base["general_insights"]),
    }

    base = _quick_score(base)
    return base


def _simhash_key(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return s[:256]


def dedup_nearby(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        key = _simhash_key(c.get("answer", ""))
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(c)
    return kept


# ========== Main answering logic (batch / single) ==========
def ask_model_batch(
    provider: str,
    client,
    model: str,
    dimension: str,
    q_list: List[str],
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
    max_tokens: int,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    For providers that support batch mode, answer multiple questions in one call.
    If parsing fails or the array length is wrong, the caller falls back to single-question mode.
    Return a candidate list whose length equals len(q_list).
    """
    total_start = time.perf_counter()
    prompt = build_batch_prompt(dimension, q_list, proposal_context, reg_hints, variant_id, lang=lang)
    _log(
        "BATCH_CALL",
        f"start provider={provider} model={model} dimension={dimension} variant={variant_id} "
        f"questions={len(q_list)} prompt_chars={len(prompt)} max_tokens={max_tokens}"
    )
    try:
        call_start = time.perf_counter()
        txt = _chat_completion_json(
            client,
            model,
            SYSTEM_EN,
            prompt,
            max_tokens=max_tokens,
            temperature=TEMP_BY_VARIANT.get(variant_id, 0.3),
            force_json=provider_caps(provider)["json_mode"],
        )
        _record_timing("batch_llm_call", time.perf_counter() - call_start)
    except Exception as e:
        elapsed = time.perf_counter() - total_start
        _record_timing("batch_failed", elapsed)
        _log(
            "BATCH_CALL",
            f"failed provider={provider} dimension={dimension} variant={variant_id} "
            f"questions={len(q_list)} elapsed_sec={elapsed:.2f} error={e}"
        )
        return []

    low = (txt or "").strip().lower()
    if any(k in low for k in ("incorrect api key", "invalid api key", "rate limit", "quota", "access denied")):
        elapsed = time.perf_counter() - total_start
        _record_timing("batch_provider_error_text", elapsed)
        _log(
            "BATCH_CALL",
            f"provider_error_text provider={provider} dimension={dimension} variant={variant_id} elapsed_sec={elapsed:.2f}"
        )
        return []

    parse_start = time.perf_counter()
    obj = _safe_parse_json_plus(txt)
    _record_timing("batch_parse", time.perf_counter() - parse_start)
    if isinstance(obj, list):
        arr = obj
    elif isinstance(obj, dict):
        arr = obj.get("answers")
        if not isinstance(arr, list):
            for key in ("items", "results", "responses"):
                val = obj.get(key)
                if isinstance(val, list):
                    arr = val
                    break
        if not isinstance(arr, list) and len(q_list) == 1 and obj.get("answer"):
            arr = [obj]
    else:
        elapsed = time.perf_counter() - total_start
        _record_timing("batch_parse_failed", elapsed)
        _log(
            "BATCH_CALL",
            f"parse_failed provider={provider} dimension={dimension} variant={variant_id} "
            f"elapsed_sec={elapsed:.2f} raw_chars={len(txt or '')}"
        )
        return []
    if isinstance(arr, list) and len(arr) > len(q_list):
        elapsed = time.perf_counter() - total_start
        _record_timing("batch_length_overflow_trimmed", elapsed)
        _log(
            "BATCH_CALL",
            f"length_overflow_trimmed provider={provider} dimension={dimension} variant={variant_id} "
            f"expected={len(q_list)} got={len(arr)} elapsed_sec={elapsed:.2f}"
        )
        arr = arr[:len(q_list)]

    if not isinstance(arr, list) or len(arr) != len(q_list):
        elapsed = time.perf_counter() - total_start
        _record_timing("batch_length_mismatch", elapsed)
        _log(
            "BATCH_CALL",
            f"length_mismatch provider={provider} dimension={dimension} variant={variant_id} "
            f"expected={len(q_list)} got={len(arr) if isinstance(arr, list) else 'not_list'} "
            f"elapsed_sec={elapsed:.2f}"
        )
        return []

    finalize_start = time.perf_counter()
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(arr, 1):
        cand_norm = _normalize_candidate_obj(it)
        if not _validate_candidate_dict(cand_norm):
            out.append({})
            continue
        finalized = _finalize_candidate(
            cand_norm,
            provider=provider,
            model=model,
            variant_id=variant_id,
            sample_id=idx,
            dimension=dimension,
        )
        out.append(finalized)
    finalize_elapsed = time.perf_counter() - finalize_start
    total_elapsed = time.perf_counter() - total_start
    _record_timing("batch_finalize", finalize_elapsed)
    _record_timing("batch_total", total_elapsed)
    _log(
        "BATCH_CALL",
        f"done provider={provider} dimension={dimension} variant={variant_id} "
        f"questions={len(q_list)} valid={len([x for x in out if x])} "
        f"finalize_sec={finalize_elapsed:.2f} total_sec={total_elapsed:.2f}"
    )
    return out


def ask_model_single(
    provider: str,
    client,
    model: str,
    dimension: str,
    question: str,
    proposal_context: str,
    reg_hints: List[str],
    variant_id: str,
    max_tokens: int,
    lang: str = "en",
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    prompt = build_single_prompt(dimension, question, proposal_context, reg_hints, variant_id, lang=lang)
    _log(
        "SINGLE_CALL",
        f"start provider={provider} model={model} dimension={dimension} variant={variant_id} "
        f"question_chars={len(question or '')} prompt_chars={len(prompt)} max_tokens={max_tokens}"
    )
    try:
        call_start = time.perf_counter()
        txt = _chat_completion_json(
            client,
            model,
            SYSTEM_EN,
            prompt,
            max_tokens=max_tokens,
            temperature=TEMP_BY_VARIANT.get(variant_id, 0.3),
            force_json=provider_caps(provider)["json_mode"],
        )
        _record_timing("single_llm_call", time.perf_counter() - call_start)
    except Exception as e:
        elapsed = time.perf_counter() - total_start
        _record_timing("single_failed", elapsed)
        _log(
            "SINGLE_CALL",
            f"failed provider={provider} dimension={dimension} variant={variant_id} "
            f"elapsed_sec={elapsed:.2f} error={e}"
        )
        return {"error": True, "answer": ""}

    parse_start = time.perf_counter()
    obj = _safe_parse_json_plus(txt)
    _record_timing("single_parse", time.perf_counter() - parse_start)
    if not isinstance(obj, dict):
        elapsed = time.perf_counter() - total_start
        _record_timing("single_parse_failed", elapsed)
        _log(
            "SINGLE_CALL",
            f"parse_failed provider={provider} dimension={dimension} variant={variant_id} "
            f"elapsed_sec={elapsed:.2f} raw_chars={len(txt or '')}"
        )
        return {"error": True, "answer": ""}

    finalize_start = time.perf_counter()
    cand_norm = _normalize_candidate_obj(obj)
    if not _validate_candidate_dict(cand_norm):
        elapsed = time.perf_counter() - total_start
        _record_timing("single_invalid_candidate", elapsed)
        _log(
            "SINGLE_CALL",
            f"invalid_candidate provider={provider} dimension={dimension} variant={variant_id} elapsed_sec={elapsed:.2f}"
        )
        return {"error": True, "answer": ""}

    finalized = _finalize_candidate(
        cand_norm,
        provider=provider,
        model=model,
        variant_id=variant_id,
        sample_id=1,
        dimension=dimension,
    )
    finalize_elapsed = time.perf_counter() - finalize_start
    total_elapsed = time.perf_counter() - total_start
    _record_timing("single_finalize", finalize_elapsed)
    _record_timing("single_total", total_elapsed)
    _log(
        "SINGLE_CALL",
        f"done provider={provider} dimension={dimension} variant={variant_id} "
        f"finalize_sec={finalize_elapsed:.2f} total_sec={total_elapsed:.2f}"
    )
    return finalized


def refine_candidate(
    candidate: Dict[str, Any],
    client,
    model: str,
    dimension: str,
    proposal_context: str,
    provider: str,
    max_tokens: int = 600,
    lang: str = "en",
) -> Dict[str, Any]:
    total_start = time.perf_counter()
    try:
        rp = build_refine_prompt(candidate, proposal_context, dimension, lang=lang)
        _log(
            "REFINE_CALL",
            f"start provider={provider} model={model} dimension={dimension} "
            f"variant={candidate.get('variant_id', 'default')} prompt_chars={len(rp)} max_tokens={max_tokens}"
        )
        call_start = time.perf_counter()
        txt = _chat_completion_json(
            client,
            model,
            SYSTEM_EN,
            rp,
            max_tokens=max_tokens,
            temperature=0.2,
            force_json=provider_caps(provider)["json_mode"],
        )
        _record_timing("refine_llm_call", time.perf_counter() - call_start)
        parse_start = time.perf_counter()
        obj = _safe_parse_json_plus(txt)
        _record_timing("refine_parse", time.perf_counter() - parse_start)
        if isinstance(obj, dict):
            cand_norm = _normalize_candidate_obj(obj)
            if not _validate_candidate_dict(cand_norm):
                elapsed = time.perf_counter() - total_start
                _record_timing("refine_invalid_candidate", elapsed)
                _log("REFINE_CALL", f"invalid_candidate provider={provider} dimension={dimension} elapsed_sec={elapsed:.2f}")
                return candidate
            finalized = _finalize_candidate(
                cand_norm,
                provider=candidate.get("provider", provider),
                model=candidate.get("model", model),
                variant_id=candidate.get("variant_id", "default"),
                sample_id=candidate.get("sample_id", 1),
                dimension=dimension,
            )
            elapsed = time.perf_counter() - total_start
            _record_timing("refine_total", elapsed)
            _log("REFINE_CALL", f"done provider={provider} dimension={dimension} elapsed_sec={elapsed:.2f}")
            return finalized
        elapsed = time.perf_counter() - total_start
        _record_timing("refine_parse_failed", elapsed)
        _log("REFINE_CALL", f"parse_failed provider={provider} dimension={dimension} elapsed_sec={elapsed:.2f}")
        return candidate
    except Exception as e:
        elapsed = time.perf_counter() - total_start
        _record_timing("refine_failed", elapsed)
        _log("REFINE_CALL", f"failed provider={provider} dimension={dimension} elapsed_sec={elapsed:.2f} error={e}")
        return candidate


def refine_candidates_parallel(
    candidates: List[Dict[str, Any]],
    client,
    model: str,
    dimension: str,
    proposal_context: str,
    provider: str,
    max_tokens: int,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    workers = _bounded_worker_count(STAGE4_REFINE_WORKERS, 1, max(1, len(candidates)))
    if workers <= 1 or len(candidates) <= 1:
        return [
            refine_candidate(
                c,
                client=client,
                model=model,
                dimension=dimension,
                proposal_context=proposal_context,
                provider=provider,
                max_tokens=max_tokens,
                lang=lang,
            )
            for c in candidates
        ]

    _log("CONCURRENCY", f"refine_parallel dimension={dimension} candidates={len(candidates)} workers={workers}")
    refined: List[Dict[str, Any]] = [candidate for candidate in candidates]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                refine_candidate,
                c,
                client=client,
                model=model,
                dimension=dimension,
                proposal_context=proposal_context,
                provider=provider,
                max_tokens=max_tokens,
                lang=lang,
            ): idx
            for idx, c in enumerate(candidates)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            refined[idx] = future.result()
    return refined


# ========== Dimension-level answering ==========
def print_dim_banner(provider_name: str, dim: str, total: int, mode: str):
    _log("DIM", f"{provider_name}: start dimension={dim} questions={total} mode={mode}")


def print_q_progress(provider_name: str, dim: str, idx: int, total: int, qtext: str):
    preview = qtext.strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."
    _log("Q", f"{provider_name}: dimension={dim} question={idx}/{total} preview={preview}")


def chunked(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield i, lst[i : i + n]


def answer_dimension(
    provider: str,
    client,
    model_name: str,
    dim: str,
    q_list: List[str],
    proposal_context: str,
    reg_hints: List[str],
    refine: bool,
    group_size: int,
    max_tokens: int,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    Return list[
      {
        "dimension": dim,
        "q_index": idx,
        "question": q,
        "candidates": [candidate_obj, ...]
      }, ...
    ]
    """
    out_items: List[Dict[str, Any]] = []
    dim_start = time.perf_counter()
    batch_count = 0
    batch_fallback_count = 0
    single_question_count = 0
    refine_candidate_count = 0
    provider_name = provider_display_name(provider)

    if not q_list:
        return out_items

    caps = provider_caps(provider)
    supports_batch = bool(caps.get("batch_ok", True))

    mode = "batch+variants" if supports_batch else "single-question+variants"
    print_dim_banner(provider_name, dim, len(q_list), mode)
    _log(
        "DIM",
        f"{provider_name}: context_chars={len(proposal_context or '')} "
        f"reg_hints={len(reg_hints or [])} refine={bool(refine)} group_size={group_size} "
        f"max_tokens={max_tokens}"
    )

    # Providers without batch support: answer one question at a time across variants.
    if not supports_batch:
        for idx, q in enumerate(q_list, 1):
            q_start = time.perf_counter()
            print_q_progress(provider_name, dim, idx, len(q_list), q)
            raw_cands: List[Dict[str, Any]] = []
            for v in VARIANTS:
                single_question_count += 1
                cand = ask_model_single(
                    provider=provider,
                    client=client,
                    model=model_name,
                    dimension=dim,
                    question=q,
                    proposal_context=proposal_context,
                    reg_hints=reg_hints,
                    variant_id=v,
                    max_tokens=min(900, max_tokens),
                    lang=lang,
                )
                if isinstance(cand, dict) and not cand.get("error") and cand.get("answer", "").strip():
                    raw_cands.append(cand)
            cands = dedup_nearby(raw_cands)
            _log(
                "Q",
                f"{provider_name}: dimension={dim} q_index={idx} raw_candidates={len(raw_cands)} "
                f"deduped_candidates={len(cands)}"
            )
            if refine and cands:
                refine_candidate_count += len(cands)
                refine_start = time.perf_counter()
                cands = refine_candidates_parallel(
                    cands,
                    client=client,
                    model=model_name,
                    dimension=dim,
                    proposal_context=proposal_context,
                    provider=provider,
                    max_tokens=min(700, max_tokens),
                    lang=lang,
                )
                _log(
                    "REFINE",
                    f"{provider_name}: dimension={dim} q_index={idx} refined_candidates={len(cands)} "
                    f"elapsed_sec={time.perf_counter() - refine_start:.2f}"
                )
            for c in cands:
                c["dimension"] = dim
                c["q_index"] = idx

            out_items.append(
                {
                    "dimension": dim,
                    "q_index": idx,
                    "question": q,
                    "candidates": cands,
                }
            )
            _log(
                "Q",
                f"{provider_name}: dimension={dim} q_index={idx} total_question_sec={time.perf_counter() - q_start:.2f}"
            )
        _log(
            "DIM",
            f"{provider_name}: done dimension={dim} items={len(out_items)} "
            f"single_calls={single_question_count} refine_calls={refine_candidate_count} "
            f"elapsed_sec={time.perf_counter() - dim_start:.2f}"
        )
        return out_items

    # Providers with batch support: process batches across variants.
    group_size = max(1, min(int(group_size), 4))
    variant_workers = _bounded_worker_count(STAGE4_VARIANT_WORKERS, 1, max(1, len(VARIANTS)))
    for start_idx, sub_qs in chunked(q_list, group_size):
        batch_count += 1
        batch_tag = f"{start_idx+1}-{start_idx+len(sub_qs)}"
        batch_start = time.perf_counter()
        per_variant_results: Dict[str, List[Dict[str, Any]]] = {}
        batch_failed = False

        def run_batch_variant(variant_id: str) -> Tuple[str, List[Dict[str, Any]], float]:
            t0 = time.time()
            arr = ask_model_batch(
                provider=provider,
                client=client,
                model=model_name,
                dimension=dim,
                q_list=sub_qs,
                proposal_context=proposal_context,
                reg_hints=reg_hints,
                variant_id=variant_id,
                max_tokens=max_tokens,
                lang=lang,
            )
            return variant_id, arr, time.time() - t0

        if variant_workers > 1 and len(VARIANTS) > 1:
            _log(
                "CONCURRENCY",
                f"batch_variants_parallel dimension={dim} batch={batch_tag} "
                f"variants={len(VARIANTS)} workers={variant_workers}"
            )
            with ThreadPoolExecutor(max_workers=variant_workers) as pool:
                future_map = {pool.submit(run_batch_variant, v): v for v in VARIANTS}
                for future in as_completed(future_map):
                    v, arr, elapsed = future.result()
                    ok = bool(arr) and len(arr) == len(sub_qs)
                    _log(
                        "BATCH",
                        f"{provider_name}: dimension={dim} batch={batch_tag} variant={v} "
                        f"returned={len(arr)}/{len(sub_qs)} elapsed_sec={elapsed:.1f} "
                        f"status={'ok' if ok else 'failed_fallback_to_single'}"
                    )
                    if not ok:
                        batch_failed = True
                    else:
                        per_variant_results[v] = arr
        else:
            for v in VARIANTS:
                v, arr, elapsed = run_batch_variant(v)
                ok = bool(arr) and len(arr) == len(sub_qs)
                _log(
                    "BATCH",
                    f"{provider_name}: dimension={dim} batch={batch_tag} variant={v} "
                    f"returned={len(arr)}/{len(sub_qs)} elapsed_sec={elapsed:.1f} "
                    f"status={'ok' if ok else 'failed_fallback_to_single'}"
                )
                if not ok:
                    batch_failed = True
                    break
                per_variant_results[v] = arr

        if not batch_failed:
            # Merge per-variant candidates for each question.
            for j, q in enumerate(sub_qs, 1):
                global_idx = start_idx + j
                raw_cands = [per_variant_results[v][j - 1] for v in VARIANTS]
                cands = [c for c in raw_cands if isinstance(c, dict) and c.get("answer", "").strip()]
                cands = dedup_nearby(cands)
                _log(
                    "Q",
                    f"{provider_name}: dimension={dim} q_index={global_idx} "
                    f"raw_candidates={len(raw_cands)} deduped_candidates={len(cands)}"
                )
                if refine and cands:
                    refine_candidate_count += len(cands)
                    refine_start = time.perf_counter()
                    cands = refine_candidates_parallel(
                        cands,
                        client=client,
                        model=model_name,
                        dimension=dim,
                        proposal_context=proposal_context,
                        provider=provider,
                        max_tokens=min(700, max_tokens),
                        lang=lang,
                    )
                    _log(
                        "REFINE",
                        f"{provider_name}: dimension={dim} q_index={global_idx} refined_candidates={len(cands)} "
                        f"elapsed_sec={time.perf_counter() - refine_start:.2f}"
                    )
                for c in cands:
                    c["dimension"] = dim
                    c["q_index"] = global_idx
                out_items.append(
                    {
                        "dimension": dim,
                        "q_index": global_idx,
                        "question": q,
                        "candidates": cands,
                    }
                )
            _log(
                "BATCH",
                f"{provider_name}: dimension={dim} batch={batch_tag} batch_total_sec={time.perf_counter() - batch_start:.2f}"
            )
            continue

        # Batch failed: fallback to single-question mode for this mini-batch.
        batch_fallback_count += 1
        for j, q in enumerate(sub_qs, 1):
            q_start = time.perf_counter()
            global_idx = start_idx + j
            print_q_progress(provider_name, dim, global_idx, len(q_list), q)
            raw_cands: List[Dict[str, Any]] = []
            for v in VARIANTS:
                single_question_count += 1
                cand = ask_model_single(
                    provider=provider,
                    client=client,
                    model=model_name,
                    dimension=dim,
                    question=q,
                    proposal_context=proposal_context,
                    reg_hints=reg_hints,
                    variant_id=v,
                    max_tokens=min(900, max_tokens),
                    lang=lang,
                )
                if isinstance(cand, dict) and not cand.get("error") and cand.get("answer", "").strip():
                    raw_cands.append(cand)
            cands = dedup_nearby(raw_cands)
            if refine and cands:
                refine_candidate_count += len(cands)
                refine_start = time.perf_counter()
                cands = refine_candidates_parallel(
                    cands,
                    client=client,
                    model=model_name,
                    dimension=dim,
                    proposal_context=proposal_context,
                    provider=provider,
                    max_tokens=min(700, max_tokens),
                    lang=lang,
                )
                _log(
                    "REFINE",
                    f"{provider_name}: dimension={dim} q_index={global_idx} refined_candidates={len(cands)} "
                    f"elapsed_sec={time.perf_counter() - refine_start:.2f}"
                )
            for c in cands:
                c["dimension"] = dim
                c["q_index"] = global_idx
            out_items.append(
                {
                    "dimension": dim,
                    "q_index": global_idx,
                    "question": q,
                    "candidates": cands,
                }
            )
            _log(
                "Q",
                f"{provider_name}: dimension={dim} q_index={global_idx} fallback_question_sec={time.perf_counter() - q_start:.2f}"
            )

    _log(
        "DIM",
        f"{provider_name}: done dimension={dim} items={len(out_items)} batches={batch_count} "
        f"batch_fallbacks={batch_fallback_count} single_calls={single_question_count} "
        f"refine_calls={refine_candidate_count} elapsed_sec={time.perf_counter() - dim_start:.2f}"
    )
    return out_items


# ========== Multi-model merge ==========
def merge_model_items(provider_items: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    def key_of(x: Dict[str, Any]):
        return (x["dimension"], x["q_index"], x["question"])

    pool: Dict[Any, Dict[str, Any]] = {}
    for _provider, items_for_provider in provider_items.items():
        for it in items_for_provider:
            k = key_of(it)
            if k in pool:
                pool[k]["candidates"].extend(it.get("candidates", []))
            else:
                pool[k] = {
                    "dimension": it["dimension"],
                    "q_index": it["q_index"],
                    "question": it["question"],
                    "candidates": list(it.get("candidates", [])),
                }

    items = list(pool.values())
    items.sort(
        key=lambda x: (
            DIM_ORDER.index(x["dimension"]) if x["dimension"] in DIM_ORDER else 99,
            x["q_index"],
        )
    )
    return items


def merge_two_models(chatgpt_items: List[Dict[str, Any]], deepseek_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return merge_model_items({"openai": chatgpt_items, "deepseek": deepseek_items})


# ========== CLI ==========
def parse_args():
    ap = argparse.ArgumentParser(
        description="Proposal-aware multi-LLM answering, no web search, dimension-facts aware."
    )
    ap.add_argument(
        "--proposal_id",
        type=str,
        default="",
        help="Proposal ID. If omitted, the latest directory under data/extracted is used.",
    )
    ap.add_argument(
        "--qs_file",
        type=str,
        default=str(CONFIG_QS_DEFAULT),
        help="Question-set JSON path, usually generated by generate_questions.py.",
    )
    ap.add_argument(
        "--dim-file",
        type=str,
        default="",
        help="Dimension JSON path. Defaults to data/extracted/{pid}/dimensions_v2.json.",
    )
    ap.add_argument(
        "--refine",
        type=int,
        default=1,
        help="Whether to run lightweight self-review on generated answers (0/1).",
    )
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=2200,
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=3,
        help="Questions per batch for providers that support batch mode. Recommended: 2-4.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--variants",
        type=str,
        default=",".join(VARIANTS),
        help="Comma-separated answer variants to run. Available: default,risk,implementation. Use default for faster debugging.",
    )
    ap.add_argument(
        "--providers",
        type=str,
        default="auto",
        help="Comma-separated providers, or auto/all for keys found in .env.",
    )
    return ap.parse_args()


def main():
    stage_start = time.perf_counter()
    args = parse_args()
    global VARIANTS
    requested_variants = [v.strip() for v in str(args.variants or "").split(",") if v.strip()]
    allowed_variants = {"default", "risk", "implementation"}
    invalid_variants = [v for v in requested_variants if v not in allowed_variants]
    if invalid_variants:
        raise ValueError(f"Invalid variants: {invalid_variants}. Allowed variants: {sorted(allowed_variants)}")
    VARIANTS = requested_variants or ["default"]
    if args.seed is not None:
        try:
            random.seed(int(args.seed))
        except Exception:
            pass

    pid = args.proposal_id.strip() or detect_latest_pid()
    if pid == "unknown":
        _log("WARN", "No proposal directory detected under data/extracted; using placeholder pid=unknown.")
    _proposal_lang = detect_proposal_language(pid)
    _log("LANG", f"proposal_language={_proposal_lang}")
    _log("STAGE_4", "Starting Stage 4: proposal-aware LLM answering")
    _log(
        "STAGE_4",
        "Purpose: answer each generated evaluation question using dimensions_v2.json context, "
        "then write provider-specific and merged answer candidates."
    )
    _log(
        "STAGE_4",
        f"proposal_id={pid} refine={bool(args.refine)} max_tokens={int(args.max_tokens)} "
        f"group_size={int(args.group_size)} variants={VARIANTS} seed={args.seed}"
    )
    audit: Dict[str, Any] = {
        "stage": "stage_4_llm_answering",
        "purpose": "Generate proposal-grounded answer candidates for each generated evaluation question.",
        "proposal_id": pid,
        "started_at": now_str(),
        "args": {
            "refine": bool(args.refine),
            "max_tokens": int(args.max_tokens),
            "group_size": int(args.group_size),
            "seed": args.seed,
            "variants": VARIANTS,
            "stage4_dim_workers": STAGE4_DIM_WORKERS,
            "stage4_variant_workers": STAGE4_VARIANT_WORKERS,
            "stage4_refine_workers": STAGE4_REFINE_WORKERS,
        },
        "inputs": {},
        "providers": {},
        "dimensions": {},
        "timing": {},
    }

    qs_path = Path(args.qs_file)
    if not qs_path.exists():
        raise FileNotFoundError(f"Question set not found: {qs_path}")
    qs_cfg = read_json(qs_path)
    _log("INPUT", f"question_set_path={qs_path} size_bytes={qs_path.stat().st_size}")
    audit["inputs"]["question_set_path"] = str(qs_path.resolve())
    audit["inputs"]["question_set_size_bytes"] = qs_path.stat().st_size

    missing = [d for d in DIM_ORDER if not get_q_list(qs_cfg.get(d, []))]
    if missing:
        raise RuntimeError(
            f"Question set is missing dimensions or has empty question lists: {missing}. "
            f"Check {qs_path} or generate_questions.py output."
        )
    for dim in DIM_ORDER:
        q_list = get_q_list(qs_cfg.get(dim, []))
        _log("INPUT", f"{dim}: questions={len(q_list)} first_question={_preview_text(q_list[0] if q_list else '')}")
        audit["dimensions"][dim] = {
            "question_count": len(q_list),
            "first_question_preview": _preview_text(q_list[0] if q_list else ""),
        }

    # Dimension context: read extracted/{pid}/dimensions_v2.json unless explicitly overridden.
    if args.dim_file.strip():
        dim_file = Path(args.dim_file.strip())
    else:
        dim_file = EXTRACTED_DIR / pid / "dimensions_v2.json"

    if not dim_file.exists():
        raise FileNotFoundError(
            f"Dimension file not found: {dim_file}. Run build_dimensions_from_facts.py first to generate dimensions_v2.json."
        )

    _log("INPUT", f"dimension_context_path={dim_file} size_bytes={dim_file.stat().st_size}")
    audit["inputs"]["dimension_context_path"] = str(dim_file.resolve())
    audit["inputs"]["dimension_context_size_bytes"] = dim_file.stat().st_size
    dim_context_map = load_dimension_context(pid, dim_file)
    for dim in DIM_ORDER:
        ctx = dim_context_map.get(dim, "")
        _log("INPUT", f"{dim}: context_chars={len(ctx)} context_preview={_preview_text(ctx)}")
        audit["dimensions"][dim]["context_chars"] = len(ctx)
        audit["dimensions"][dim]["context_preview"] = _preview_text(ctx)

    reg_hints_map: Dict[str, List[str]] = {}
    for dim in DIM_ORDER:
        reg_hints_map[dim] = _load_reg_hints(qs_cfg, dim, limit=8)
        _log("INPUT", f"{dim}: optional_hints={len(reg_hints_map[dim])}")
        audit["dimensions"][dim]["optional_hints_count"] = len(reg_hints_map[dim])

    requested_providers = parse_provider_list(args.providers)
    if ACTIVE_PROVIDER in {"openai", "gemini", "deepseek"} and ACTIVE_PROVIDER not in requested_providers:
        requested_providers.insert(0, ACTIVE_PROVIDER)
    requested_providers = list(dict.fromkeys(requested_providers))
    _log("PROVIDERS", f"configured={requested_providers}")

    provider_clients: Dict[str, Dict[str, Any]] = {}
    for provider in requested_providers:
        client, model = init_provider(provider)
        if client and model:
            ok, reason = validate_chat_provider(provider, client, model)
            audit["providers"][provider] = {
                "enabled": bool(ok),
                "model": model,
                "health_check": {"ok": bool(ok), "reason": reason},
                "dimensions": {},
            }
            if ok:
                provider_clients[provider] = {"client": client, "model": model}
            else:
                _log("WARN", f"Skipping {provider} for this run because provider health check failed: {reason}")
        else:
            audit["providers"][provider] = {
                "enabled": False,
                "reason": f"{provider.upper()} API key not configured or provider unavailable",
            }
            _log("WARN", f"Skipping {provider}: {audit['providers'][provider]['reason']}")

    if not provider_clients:
        raise RuntimeError("No usable model detected. Configure OPENAI_API_KEY, GEMINI_API_KEY, and/or DEEPSEEK_API_KEY in .env.")

    dims = DIM_ORDER[:]
    total_questions = sum(len(get_q_list(qs_cfg.get(dim, []))) for dim in dims)
    variant_count = len(VARIANTS)

    estimated_generation_calls_by_provider: Dict[str, int] = {}
    for provider in provider_clients:
        if provider_caps(provider).get("batch_ok", True):
            calls = sum(
                ((len(get_q_list(qs_cfg.get(dim, []))) + max(1, int(args.group_size)) - 1) // max(1, int(args.group_size)))
                * variant_count
                for dim in dims
            )
        else:
            calls = total_questions * variant_count
        estimated_generation_calls_by_provider[provider] = calls
    estimated_refine_calls = total_questions * variant_count * len(provider_clients) if bool(args.refine) else 0
    estimated_total = sum(estimated_generation_calls_by_provider.values()) + estimated_refine_calls
    _log(
        "CALL_PLAN",
        f"questions={total_questions} variants={variant_count} providers={list(provider_clients.keys())} "
        f"estimated_generation_calls_by_provider={estimated_generation_calls_by_provider} "
        f"estimated_refine_calls_if_all_candidates_valid={estimated_refine_calls} "
        f"estimated_total_llm_calls={estimated_total}"
    )
    audit["call_plan"] = {
        "questions": total_questions,
        "variants": VARIANTS,
        "providers": list(provider_clients.keys()),
        "estimated_generation_calls_by_provider": estimated_generation_calls_by_provider,
        "estimated_refine_calls_if_all_candidates_valid": estimated_refine_calls,
        "estimated_total_llm_calls": estimated_total,
    }

    provider_items_all: Dict[str, List[Dict[str, Any]]] = {provider: [] for provider in provider_clients}

    def run_provider(provider: str, client, model: str) -> Tuple[str, List[Dict[str, Any]]]:
        _log("PROVIDER", f"{provider} answering started")

        def run_dimension(dim: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
            dim_provider_start = time.perf_counter()
            q_list = get_q_list(qs_cfg.get(dim, []))
            ctx = dim_context_map.get(dim, "")
            reg_hints = reg_hints_map.get(dim, [])
            items = answer_dimension(
                provider=provider,
                client=client,
                model_name=model,
                dim=dim,
                q_list=q_list,
                proposal_context=ctx,
                reg_hints=reg_hints,
                refine=bool(args.refine),
                group_size=int(args.group_size),
                max_tokens=int(args.max_tokens),
                lang=_proposal_lang,
            )
            dim_audit = {
                "items": len(items),
                "candidates": sum(len(it.get("candidates", []) or []) for it in items),
                "elapsed_sec": round(time.perf_counter() - dim_provider_start, 2),
            }
            return dim, items, dim_audit

        dim_workers = _bounded_worker_count(STAGE4_DIM_WORKERS, 1, len(dims))
        _log("CONCURRENCY", f"{provider}_dimensions workers={dim_workers} dimensions={len(dims)}")
        dim_results: Dict[str, List[Dict[str, Any]]] = {}
        completed_dims = 0
        if dim_workers > 1:
            with ThreadPoolExecutor(max_workers=dim_workers) as pool:
                future_map = {pool.submit(run_dimension, dim): dim for dim in dims}
                for future in as_completed(future_map):
                    dim, items, dim_audit = future.result()
                    dim_results[dim] = items
                    audit["providers"][provider]["dimensions"][dim] = dim_audit
                    completed_dims += 1
                    _log(
                        "PROVIDER",
                        f"{provider} dimension={dim} items={len(items)} completed_dims={completed_dims}/{len(dims)} "
                        f"elapsed_sec={dim_audit['elapsed_sec']}"
                    )
        else:
            for dim in dims:
                dim, items, dim_audit = run_dimension(dim)
                dim_results[dim] = items
                audit["providers"][provider]["dimensions"][dim] = dim_audit
                completed_dims += 1
                _log(
                    "PROVIDER",
                    f"{provider} dimension={dim} items={len(items)} completed_dims={completed_dims}/{len(dims)} "
                    f"elapsed_sec={dim_audit['elapsed_sec']}"
                )

        provider_items: List[Dict[str, Any]] = []
        for dim in dims:
            provider_items.extend(dim_results.get(dim, []))

        out_path = OUT_REFINED / pid / provider_raw_filename(provider)
        write_json(
            out_path,
            {
                "meta": {
                    "model": model,
                    "provider": provider,
                    "generated_at": now_str(),
                    "pid": pid,
                },
                "items": provider_items,
            },
        )
        _log("OK", f"{provider} raw results written path={out_path}")
        _log("PROVIDER", f"{provider} answering completed items={len(provider_items)}")
        return provider, provider_items

    provider_workers = _bounded_worker_count(int(os.getenv("STAGE4_PROVIDER_WORKERS", "3")), 1, len(provider_clients))
    _log("CONCURRENCY", f"providers workers={provider_workers} providers={list(provider_clients.keys())}")
    if provider_workers > 1:
        with ThreadPoolExecutor(max_workers=provider_workers) as pool:
            future_map = {
                pool.submit(run_provider, provider, cfg["client"], cfg["model"]): provider
                for provider, cfg in provider_clients.items()
            }
            for future in as_completed(future_map):
                provider, items = future.result()
                provider_items_all[provider] = items
    else:
        for provider, cfg in provider_clients.items():
            provider, items = run_provider(provider, cfg["client"], cfg["model"])
            provider_items_all[provider] = items

    merged_items = merge_model_items(provider_items_all)
    merged = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "schema": "refined_items.v2.proposal_aware_with_general_insights",
            "args": {
                "refine": bool(args.refine),
                "max_tokens": int(args.max_tokens),
                "group_size": int(args.group_size),
            },
            "models": {
                provider: {"model": cfg["model"], "provider": provider}
                for provider, cfg in provider_clients.items()
                if provider_items_all.get(provider)
            },
        },
        "items": merged_items,
    }
    out_path = OUT_REFINED / pid / "all_refined_items.json"
    write_json(out_path, merged)
    total_candidates = sum(len(it.get("candidates", []) or []) for it in merged_items)
    elapsed_sec = time.perf_counter() - stage_start
    audit.update(
        {
            "finished_at": now_str(),
            "elapsed_sec": round(elapsed_sec, 2),
            "timing": _timing_summary(),
            "outputs": {
                "merged_output_path": str(out_path.resolve()),
                "provider_output_paths": {
                    provider: str((OUT_REFINED / pid / provider_raw_filename(provider)).resolve())
                    for provider, items in provider_items_all.items()
                    if items
                },
            },
            "summary": {
                "merged_items": len(merged_items),
                "total_candidates": total_candidates,
                "providers": list(provider_clients.keys()),
                "items_by_provider": {
                    provider: len(items)
                    for provider, items in provider_items_all.items()
                },
            },
        }
    )
    for dim in DIM_ORDER:
        dim_items = [it for it in merged_items if it.get("dimension") == dim]
        audit["dimensions"][dim]["answered_questions"] = len(dim_items)
        audit["dimensions"][dim]["candidates"] = sum(len(it.get("candidates", []) or []) for it in dim_items)
    audit_path = OUT_REFINED / pid / "stage4_answering_audit.json"
    write_json(audit_path, audit)
    _log("OK", f"merged refined items written path={out_path}")
    _log("OK", f"stage 4 answering audit written path={audit_path}")
    for name, bucket in _timing_summary().items():
        _log(
            "TIMING",
            f"{name}: count={bucket['count']} total_sec={bucket['total_sec']} "
            f"avg_sec={bucket['avg_sec']} max_sec={bucket['max_sec']}"
        )
    _log(
        "SUMMARY",
        f"merged_items={len(merged_items)} total_candidates={total_candidates} "
        f"items_by_provider={{{', '.join(f'{p}: {len(items)}' for p, items in provider_items_all.items())}}} "
        f"elapsed_sec={elapsed_sec:.2f}"
    )
    for dim in DIM_ORDER:
        dim_items = [it for it in merged_items if it.get("dimension") == dim]
        dim_candidates = sum(len(it.get("candidates", []) or []) for it in dim_items)
        _log("SUMMARY", f"{dim}: answered_questions={len(dim_items)} candidates={dim_candidates}")
    _log("STAGE_4", "Completed Stage 4")


if __name__ == "__main__":  # pragma: no cover
    main()
