# -*- coding: utf-8 -*-
"""
Stage 3 - Dimension-specific evaluation question generator (generate_questions.py v3.1)

Input:
  - src/data/extracted/<proposal_id>/dimensions_v2.json

Outputs:
  1) Simplified per-proposal version for llm_answering with question string lists grouped by dimension:
     - src/data/questions/<proposal_id>/generated_questions.json

  2) Active simplified version used by llm_answering:
     - src/data/config/question_sets/generated_questions.json

  3) Detailed per-proposal version with qid, aspect, answer_type, links_to, and full metadata:
     - src/data/questions/<proposal_id>/generated_questions_detail.json
     Example:
     {
       "proposal_id": "XXX",
       "generated_at": "...",
       "model": "...",
       "provider": "...",
       "team": {
         "dimension": "team",
         "questions": ["Question 1", "Question 2", "..."],
         "search_hints": [],
         "source_proposal_id": "XXX"
       },
       ...
     }

Key behavior:
  - Questions must explicitly anchor to key_points, risks, or mitigations through links_to indexes.
  - Each dimension should include at least two rating questions and several analysis questions.
  - Questions must be based on the dimension payload rather than generic assumptions.
  - Missing-information handling and long-term/platform-oriented questions reduce hallucination risk later.
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.llm_provider import UnifiedChatClient, default_model

load_dotenv()

# ========== Path configuration ==========

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROGRESS_FILE = DATA_DIR / "step_progress.json"


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass
QUESTIONS_DIR = DATA_DIR / "questions"
CONFIG_QS_DIR = DATA_DIR / "config" / "question_sets"

QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_QS_DIR.mkdir(parents=True, exist_ok=True)

# ========== LLM configuration ==========

PROVIDER = os.getenv("PROVIDER", "openai").lower()
OPENAI_MODEL = default_model(PROVIDER)
PREVIEW_CHARS = int(os.getenv("QUESTIONS_PREVIEW_CHARS", "280"))
LLM_JSON_RETRIES = int(os.getenv("QUESTIONS_LLM_JSON_RETRIES", "2"))

DIMENSION_NAMES = ["team", "objectives", "strategy", "innovation", "feasibility"]

# Aspect IDs used for long-term/platform sanity checks.
PLATFORM_ASPECT_IDS = {
    "platform_and_extensibility",
    "scaling_and_globalization",
}

BUSINESS_ARCHETYPE_GUIDE = """
Business-context adaptation guide:
- Before writing questions, silently infer the most likely business archetype from the payload. Do not force a label if the evidence is weak.
- Possible archetypes include: SaaS/software, AI/data product, marketplace, consumer product/brand, B2B service, professional services,
  education/training, healthcare or life-science initiative, industrial/manufacturing, hardware/device, energy/climate,
  logistics/supply-chain, fintech/financial service, real estate/infrastructure, agriculture/food, media/content,
  nonprofit/public-sector program, research commercialization, internal corporate transformation, or mixed model.
- For every archetype, adapt the diligence lens:
  - Revenue model: subscription, usage, transaction, license, services, grant, reimbursement, procurement, asset sale, or hybrid.
  - Customer/user: buyer, end user, payer, channel, beneficiary, regulator, operator, partner, or internal sponsor.
  - Proof standard: customer discovery, pilots, LOIs, retention, benchmark tests, certifications, adoption data, operational metrics,
    financial model, delivery capacity, legal approvals, safety/security review, or impact evidence, depending on the archetype.
  - Scale constraints: people, capital, inventory, supply chain, data, regulation, channels, geography, facilities, implementation labor,
    partnerships, or governance.
- Use the payload's own vocabulary for the domain. If evidence is missing, ask about the missing evidence instead of importing a sector template.
"""


# ========== Diagnostic logging ==========

def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def _preview_text(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "<empty>"
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_question_text(text: str) -> str:
    return re.sub(r"\W+", " ", _normalize_text(text).lower()).strip()


def _dedupe_question_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove exact or near-exact duplicates while preserving the first occurrence.
    This keeps the generated set cleaner when the LLM repeats itself.
    """
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        question = _normalize_text(item.get("question", ""))
        question_en = _normalize_text(item.get("question_en", ""))
        if not question:
            continue
        key = (
            _normalize_question_text(question),
            _normalize_question_text(question_en),
            _normalize_text(item.get("aspect", "")).lower(),
            _normalize_text(item.get("answer_type", "")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        cleaned_item = dict(item)
        cleaned_item["question"] = question
        cleaned_item["question_en"] = question_en or question
        deduped.append(cleaned_item)
    return deduped


def _count_values(values: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _question_distribution(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    answer_types = []
    priorities = []
    aspects = []
    linked_to_key_points = 0
    linked_to_risks = 0
    linked_to_mitigations = 0
    for q in questions:
        if not isinstance(q, dict):
            continue
        answer_types.append(str(q.get("answer_type", "unknown")))
        priorities.append(str(q.get("priority", "unknown")))
        aspects.append(str(q.get("aspect", "unknown")))
        links = q.get("links_to") or {}
        if links.get("key_points"):
            linked_to_key_points += 1
        if links.get("risks"):
            linked_to_risks += 1
        if links.get("mitigations"):
            linked_to_mitigations += 1
    return {
        "answer_types": _count_values(answer_types),
        "priorities": _count_values(priorities),
        "aspects": _count_values(aspects),
        "linked_to_key_points": linked_to_key_points,
        "linked_to_risks": linked_to_risks,
        "linked_to_mitigations": linked_to_mitigations,
    }


def _dimension_input_stats(dim_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary", "") or ""
    key_points = payload.get("key_points", []) or []
    risks = payload.get("risks", []) or []
    mitigations = payload.get("mitigations", []) or []
    return {
        "dimension": dim_name,
        "summary_chars": len(summary),
        "key_points_count": len(key_points),
        "risks_count": len(risks),
        "mitigations_count": len(mitigations),
        "summary_preview": _preview_text(summary),
        "first_key_point_preview": _preview_text(key_points[0]) if key_points else "<none>",
        "first_risk_preview": _preview_text(risks[0]) if risks else "<none>",
        "first_mitigation_preview": _preview_text(mitigations[0]) if mitigations else "<none>",
    }

def _looks_like_team_bio_question(question_text: str, question_en: str) -> bool:
    """Use keywords to detect whether a question asks about team background or experience."""
    question_text = (question_text or "").lower()
    question_en = (question_en or "").lower()

    kw_en = [
        "team", "core team", "core member", "leader", "leadership",
        "principal investigator", "pi",
        "background", "track record", "experience", "experiences",
        "domain", "validation", "compliance", "commercialization", "operations",
    ]

    return any(k in question_text for k in kw_en) or any(k in question_en for k in kw_en)


def _looks_like_market_question(question_text: str, question_en: str) -> bool:
    """Use keywords to detect whether a question asks about market, competition, or pricing."""
    question_text = (question_text or "").lower()
    question_en = (question_en or "").lower()

    kw_en = [
        "market", "market size", "market analysis", "market opportunity",
        "competitive", "competition", "competitor", "competitors",
        "customer", "customers", "user", "users", "buyer", "payer", "beneficiary",
        "target population", "stakeholder", "channel", "procurement",
        "cagr", "growth", "sales", "revenue", "pricing", "funding", "payment",
    ]

    return any(k in question_text for k in kw_en) or any(k in question_en for k in kw_en)

# ========== Dimension-specific configuration ==========

DIMENSION_CONFIG: Dict[str, Dict[str, Any]] = {
    "team": {
        "min_q": 6,
        "max_q": 9,
        "focus": (
            "Focus on team composition, founder/leader track record, domain-specific execution experience, "
            "partner/collaboration networks, long-term execution stability, operating capacity, and time commitment."
        ),
        "aspects": [
            {
                "id": "leadership_experience",
                "desc": "Leadership experience and prior execution record of founders, project leaders, operators, sponsors, or core decision-makers.",
            },
            {
                "id": "domain_expertise",
                "desc": "Depth of team expertise in the target customer problem, product/service domain, operating model, technical area, or implementation setting.",
            },
            {
                "id": "validation_and_compliance_experience",
                "desc": "Team experience with domain-specific validation, quality systems, compliance, approvals, enterprise/customer qualification, or regulated translation when applicable.",
            },
            {
                "id": "collaboration_network",
                "desc": "Domestic and international collaborators, strategic customers, operators, suppliers, channel partners, institutions, and their stability and complementarity.",
            },
            {
                "id": "governance_and_decision_making",
                "desc": "Project governance, decision-making process, conflict-of-interest handling, and quality control system.",
            },
            {
                "id": "team_capacity_and_bandwidth",
                "desc": "Current workload, parallel projects, and whether the team has enough time and resources for this project.",
            },
        ],
    },
    "objectives": {
        "min_q": 6,
        "max_q": 9,
        "focus": (
            "Focus on clarity of overall goals, staged milestones, alignment with user/customer/stakeholder or market needs, "
            "quantifiable success metrics, business or impact logic, and realism."
        ),
        "aspects": [
            {
                "id": "overall_goal_clarity",
                "desc": "Whether the overall goal is clear and focused, without too many unrelated sub-goals.",
            },
            {
                "id": "unmet_need_alignment",
                "desc": "Degree and urgency of alignment with the proposal's stated unmet need, customer problem, stakeholder need, domain-specific need, or market opportunity.",
            },
            {
                "id": "milestones_and_timeline",
                "desc": "Whether staged milestones, such as 0-12 months and 12-36 months, are reasonable and executable.",
            },
            {
                "id": "outcome_and_success_metrics",
                "desc": "Whether goals include clear and quantifiable success metrics, such as product metrics, service levels, customer outcomes, impact metrics, quality targets, financial KPIs, or operating KPIs.",
            },
            {
                "id": "business_or_impact_logic",
                "desc": "Whether the objectives explain how the initiative creates value, captures revenue or funding, delivers impact, or changes stakeholder behavior.",
            },
            {
                "id": "scope_and_prioritization",
                "desc": "Whether the project scope is too broad, has too many parallel pipelines, and uses clear prioritization.",
            },
            {
                "id": "realism_and_ambition_balance",
                "desc": "Balance between ambition and feasibility, including whether goals are too idealistic or too conservative.",
            },
        ],
    },
    "strategy": {
        "min_q": 7,
        "max_q": 10,
        "focus": (
            "Focus on solution route design, validation/development path, quality/compliance strategy when applicable, go-to-market path, "
            "partner strategy, operating model, and use of data/resources."
        ),
        "aspects": [
            {
                "id": "solution_delivery_strategy",
                "desc": "Rationale and alternatives for the solution, delivery, operating, product, service, technical, or program route, including major trade-offs and validation path.",
            },
            {
                "id": "development_and_validation_path",
                "desc": "Development and validation staging, such as prototype validation, pilot deployment, customer qualification, real-world validation, operating readiness, regulated-use readiness when applicable, or scale-up gates.",
            },
            {
                "id": "quality_compliance_strategy",
                "desc": "Quality, compliance, approval, procurement, security, data, registration, or regulatory strategy for the target product/service and use setting when applicable.",
            },
            {
                "id": "commercialization_and_market_entry",
                "desc": "Commercialization, adoption, or rollout model, including pricing/payment/funding logic, sales or procurement motion, channel strategy, and market-entry path across segments, regions, or settings.",
            },
            {
                "id": "business_model_and_unit_economics",
                "desc": "Revenue, cost, margin, funding, pricing, procurement, reimbursement, utilization, retention, or impact economics that support the intended model.",
            },
            {
                "id": "partnership_and_business_model",
                "desc": "Partnership model with channel partners, strategic customers, suppliers, platforms, research institutions, operators, or service partners, including licensing, co-development, procurement, or delivery.",
            },
            {
                "id": "data_and_real_world_evidence_strategy",
                "desc": "Strategy for using experimental, operating, production, customer, market, real-world, or cohort data plus privacy, governance, and compliance arrangements when relevant.",
            },
            {
                "id": "scaling_and_globalization",
                "desc": "Scaling strategy from early validation to large-scale rollout, including multicenter or international expansion.",
            },
        ],
    },
    "innovation": {
        "min_q": 6,
        "max_q": 9,
        "focus": (
            "Focus on technical or product novelty relative to current approaches, differentiation, IP position, and supporting evidence."
        ),
        "aspects": [
            {
                "id": "novelty_vs_state_of_art",
                "desc": "True novelty compared with the state of the art, including products, services, platforms, processes, materials, business models, scientific methods, or AI systems as relevant.",
            },
            {
                "id": "differentiation_and_competitive_edge",
                "desc": "Clear advantages versus existing similar or alternative approaches, such as customer value, speed, quality, performance, safety, convenience, scalability, trust, brand, access, or cost.",
            },
            {
                "id": "ip_and_protection",
                "desc": "Patent, software, and data-asset protection strategy and whether it supports medium- to long-term competition.",
            },
            {
                "id": "evidence_strength_for_innovation",
                "desc": "Strength of experimental, analytical, operational, field, customer, market, regulated-use, or real-world evidence for the innovation, including design quality and reproducibility.",
            },
            {
                "id": "platform_and_extensibility",
                "desc": "Whether the innovation can extend to other customer segments, use cases, geographies, channels, product lines, or operating contexts, or is only a one-off improvement.",
            },
            {
                "id": "risk_of_obsolescence_or_commoditization",
                "desc": "Risk that the offering, model, technology, channel, or operating advantage may be copied, commoditized, regulated away, or become obsolete within 3-5 years.",
            },
        ],
    },
    "feasibility": {
        "min_q": 7,
        "max_q": 10,
        "focus": (
            "Focus on resources and infrastructure, funding and budget, implementation path, key risks and mitigations, "
            "and practical feasibility, including quality, compliance, procurement, security, regulatory, payment, or adoption environments when relevant."
        ),
        "aspects": [
            {
                "id": "resources_and_infrastructure",
                "desc": "Whether people, capital, operating resources, tools, facilities, delivery sites, data platforms, supply chain, channels, and other resources are sufficient and stable long term.",
            },
            {
                "id": "funding_and_budget_planning",
                "desc": "Diversity of funding sources, reasonableness of budget allocation, and follow-on financing or sustainability plan.",
            },
            {
                "id": "operational_execution_plan",
                "desc": "Whether the implementation plan is specific and clear, including key tasks, owners, dependencies, operating cadence, decision rights, and timeline.",
            },
            {
                "id": "risk_management",
                "desc": "Identification and quantification of technical, validation, operating, production, domain-specific, market, security, compliance, and other risks, plus mitigations.",
            },
            {
                "id": "approval_payment_or_customer_acceptance_feasibility",
                "desc": "Realistic feasibility of customer acceptance, procurement, payment, funding, reimbursement, legal approval, security review, quality acceptance, or stakeholder buy-in in target settings when applicable.",
            },
            {
                "id": "implementation_barriers",
                "desc": "Barriers to implementation in real production, customer, service, regulated, operational, or deployment settings, including adoption and system integration.",
            },
            {
                "id": "timeline_and_resource_alignment",
                "desc": "Whether timeline and resources are aligned, including potential bottleneck periods or staffing gaps.",
            },
        ],
    },
}


# ========== Utility functions ==========

def find_latest_extracted_proposal_id() -> str:
    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(f"Extracted directory not found: {EXTRACTED_DIR}")

    candidates = []
    for d in EXTRACTED_DIR.iterdir():
        if d.is_dir():
            candidates.append((d.stat().st_mtime, d.name))

    if not candidates:
        raise FileNotFoundError(f"No proposal subdirectories found under extracted directory: {EXTRACTED_DIR}")

    proposal_id = max(candidates, key=lambda x: x[0])[1]
    _log("INFO", f"[auto] selected latest extracted proposal_id={proposal_id}")
    return proposal_id


def load_dimensions(proposal_id: str) -> Dict[str, Any]:
    path = EXTRACTED_DIR / proposal_id / "dimensions_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"dimensions_v2.json does not exist. Run build_dimensions_from_facts.py first: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    _log("INFO", f"loaded dimensions_v2.json path={path} size_bytes={path.stat().st_size}")
    present_dims = [dim for dim in DIMENSION_NAMES if isinstance(data.get(dim), dict)]
    _log("DIAG", f"dimensions_present={present_dims}")
    missing_dims = [dim for dim in DIMENSION_NAMES if dim not in present_dims]
    if missing_dims:
        _log("WARN", f"dimensions_missing={missing_dims}")
    return data


def safe_truncate(text: str, max_len: int = 4000) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ... [truncated for question-generation context]"


def build_dimension_payload(dim_name: str, dim_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the LLM payload for one dimension from dimensions_v2.json, with metadata counts.
    """
    summary = dim_data.get("summary", "") or ""
    key_points = dim_data.get("key_points", []) or []
    risks = dim_data.get("risks", []) or []
    mitigations = dim_data.get("mitigations", []) or []

    key_points_trunc = [safe_truncate(k, 400) for k in key_points[:12]]
    risks_trunc = [safe_truncate(r, 400) for r in risks[:8]]
    mitigations_trunc = [safe_truncate(m, 400) for m in mitigations[:8]]

    # Include risk_coverage in the payload metadata when available.
    risk_coverage = dim_data.get("risk_coverage", {})

    payload = {
        "dimension": dim_name,
        "summary": safe_truncate(summary, 1200),
        "key_points": key_points_trunc,
        "risks": risks_trunc,
        "mitigations": mitigations_trunc,
        "meta": {
            "key_points_count": len(key_points),
            "risks_count": len(risks),
            "mitigations_count": len(mitigations),
            "risk_coverage": risk_coverage,
        },
    }
    return payload


def _log_dimension_payload(dim_name: str, dim_payload: Dict[str, Any]) -> None:
    stats = _dimension_input_stats(dim_name, dim_payload)
    _log(
        "INPUT",
        f"{dim_name}: summary_chars={stats['summary_chars']} "
        f"key_points={stats['key_points_count']} risks={stats['risks_count']} "
        f"mitigations={stats['mitigations_count']}"
    )
    _log("INPUT", f"{dim_name}: summary_preview={stats['summary_preview']}")
    _log("INPUT", f"{dim_name}: first_key_point={stats['first_key_point_preview']}")
    _log("INPUT", f"{dim_name}: first_risk={stats['first_risk_preview']}")
    _log("INPUT", f"{dim_name}: first_mitigation={stats['first_mitigation_preview']}")


def get_openai_client() -> UnifiedChatClient:
    return UnifiedChatClient(provider=PROVIDER, model=OPENAI_MODEL)


# ========== Prompt template with anti-hallucination and platform perspective ==========

QUESTION_PROMPT_TEMPLATE = """
You are acting as a sector-agnostic venture, business, product, technical, and operating diligence expert and questionnaire-design consultant.
Your task is to design an evaluation question set for one dimension of an AI-assisted review system.

System context:
- The system extracts five dimension summaries from a proposal: team / objectives / strategy / innovation / feasibility.
- You are responsible only for this dimension: {dimension_name}.
- You will receive this dimension's payload: summary + key_points + risks + mitigations.
- Questions must be grounded in that payload, not generic review commentary.
- Design questions for serious business, startup, investment, and scale-up diligence. Prefer questions that expose
  whether the project has proof of problem urgency, product/service performance, customer demand, competitive
  defensibility, go-to-market motion, execution milestones, unit economics or budget realism, operating scalability,
  governance, compliance/security needs, and risk controls.

{business_archetype_guide}

Current dimension:
- Dimension key: {dimension_name}
- Dimension focus: {dimension_focus}

Aspect configuration:
- You will receive a JSON array of aspects, each shaped like:
  {{
    "id": "leadership_experience",
    "desc": "Leadership experience and prior execution record of founders, project leaders, operators, sponsors, or core decision-makers."
  }}
- Use aspects as sub-directions for focused questions.
- Read the payload first, decide which aspects are most relevant, and design questions around those aspects.
- You do not need to cover every aspect, but cover at least 3-5 key aspects when possible.
- For platform or long-term aspects such as platform_and_extensibility and scaling_and_globalization,
  design at least one long-term or platform-oriented question. If the payload lacks relevant information,
  ask the respondent to evaluate the risk created by that information gap.
- If the initiative is not technology-centered, translate aspects into the closest useful business meaning:
  "solution route" may mean service design, operating model, channel strategy, curriculum design, facility rollout,
  financing structure, supply chain setup, partner delivery model, or public-program implementation.

Content anchoring requirements:
1. Explicitly use key_points, risks, and mitigations from the payload:
   - At least half of the questions should link to one or more key_points.
   - If risks exist, create at least two questions specifically about those risks.
   - If mitigations exist, create at least one question assessing their sufficiency.
2. Entity-name constraints:
   - Do not introduce company, institution, university, customer, partner, platform, product, fund, country, or city names
     that do not appear in the payload.
   - If you need to refer to unnamed partners or institutions, use generic phrases such as
     "an international enterprise", "a partner customer", "a research institution", or "a platform company".
   - If an entity name appears in the payload, you may cite it exactly as written, but do not add new entity names
     or invent foreign names for people.
3. If a question requires target customer type, buyer/user separation, market size, pricing, proof of demand,
   operating capacity, unit economics, approval path, impact metrics, efficacy improvement, or similar details that the
   payload does not provide, phrase the question so the answer can first state the information gap and then analyze
   its impact.
4. Do not assume the project belongs to any specific sector, customer type, business model, operating model,
   regulated pathway, or delivery setting unless the payload explicitly supports that. Use
   domain-neutral wording such as validation, customer qualification, adoption, operating scalability, compliance,
   security, procurement, delivery, or quality system when that better fits the project.

Dimension-specific requirements:
- If the dimension is team:
  - Create at least two high-priority resume/track-record questions with priority 1 or 2.
  - These should ask about core members, PIs, project leaders, prior project experience, domain experience,
    compliance/validation experience when relevant, and translation from plan/prototype to launch, delivery, operation, or scale-up.
  - They should ask respondents to judge whether the team can advance the project to its intended application,
    validation, deployment, operation, scale-up, or commercialization path based on the team background actually present in the payload.
- If the dimension is strategy:
  - Create at least one high-priority question about the model by which the initiative reaches users/customers,
    secures buyers/payers/funders, and sustains revenue, funding, adoption, or impact.
  - If key_points mention market, CAGR, competition, customers, users, sales, procurement, pricing, funding,
    adoption, or policy support, create at least one high-priority market/adoption-driven question about positioning,
    competitive pressure, stakeholder adoption, route to market, or rollout strategy.
  - These questions should link to relevant key_points when possible.
- If the dimension is objectives:
  - If key_points mention target users, buyers, beneficiaries, target customers, target market, funders, public need,
    operational pain point, or market opportunity, create at least one question about alignment between objectives and
    the stated need, value proposition, impact logic, or market opportunity.
  - Allow the answer to identify missing information first when the payload is incomplete.

Missing-information handling:
- Questions should allow for insufficient proposal information.
- For questions that depend on concrete metrics, countries, companies, milestones, or detailed assumptions,
  include wording such as:
  "If the proposal does not provide sufficient details on X, please first state this information gap and then
  discuss its impact on the assessment."
- Do not imply that missing details were definitely provided, because that can push later answer generation into
  inventing facts.
- For rating questions, ask for a 1-5 score plus the exact evidence that would move the score up. This makes the
  downstream report more actionable for founders, investors, and operating teams.
- Make every rating question decision-useful: specify what a low score would block and what evidence would raise confidence.

links_to requirements:
- For every question, identify which payload items it primarily targets:
  - links_to.key_points: zero-based indexes into the key_points array, for example [0, 2].
  - links_to.risks: zero-based indexes into the risks array.
  - links_to.mitigations: zero-based indexes into the mitigations array.
- If a question mainly targets missing information or a blind spot, all three lists may be empty.
- Indexes must be integers and must start from 0.

Question-design principles:
1. Specificity:
   - Questions must fit the current dimension and relevant aspects, not generic project-review topics.
   - At least half of the questions must include a concrete noun or phrase from the payload, unless the payload is too sparse.
2. Answerability:
   - Questions should be answerable from general domain knowledge plus the dimension payload.
   - Avoid relying on hidden details.
   - Do not ask for exact numbers or complete company lists that are not present in the proposal unless the question
     explicitly includes missing-information handling.
3. Structured use:
   - answer_type must be one of ["analysis", "rating", "yes_no", "open"].
   - "analysis": asks for analytical text.
   - "rating": supports a 1-5 or similar scoring scale.
   - "yes_no": asks for a judgment and should invite a reason.
   - "open": open-ended without a forced structure.
   - Questions within one dimension should cover different aspects.
4. Language and schema:
   - Output professional English questions only.
   - Populate both question and question_en fields. They may be identical or closely equivalent.
   - Use generic references such as "the initiative", "the project", "the team", "the offering", "the operating model",
     "the service", "the product", "the program", or "the solution" when that is clearer
     than repeating long payload text.

Quantity and priority requirements:
- You will receive a target question range, such as [6, 9]. Stay within that range when possible:
  - Use the upper end for information-rich dimensions.
  - Use the lower end for sparse dimensions.
- priority meanings:
  - 1: high priority, core question.
  - 2: medium priority, recommended question.
  - 3: optional question.
- Each dimension should have at least three priority=1 questions when possible.

Answer-type mix:
- Each dimension should include at least two rating questions for later quantitative scoring.
- Each dimension should include at least three analysis questions for deeper written analysis.
- Other questions may be yes_no or open.

Search hints:
- Also generate a "search_hints" list: 6-10 short phrases or terms (1-5 words each) that would appear in a
  substantive, well-evidenced answer to the questions in this dimension.
- These terms should DISCRIMINATE quality: they are the specific evidence markers, credentials, validation
  artifacts, or technical anchors that a STRONG proposal would discuss — not generic topic words.
- Ground hints in the dimension payload: use key domain terms, validation standards, approval pathways,
  specific evidence types, or outcome metrics that a high-quality answer should reference.
- Avoid generic nouns like "analysis", "plan", "team", "system", "proposal", "model" — they appear in every answer.
- Good examples for feasibility: ["phase II trial", "IRB approval", "budget breakdown",
  "clinical site identified", "regulatory pathway", "prospective validation", "endpoint definition"]
- Good examples for team: ["principal investigator", "publications", "clinical trial experience",
  "industry partnership", "regulatory submission experience"]
- Return hints as a flat JSON array of strings.

Output JSON structure:
- Return a JSON object with exactly two top-level keys: "questions" and "search_hints".
- "questions" must be an array of question objects with this shape:

  {{
    "aspect": "aspect id, such as leadership_experience or solution_delivery_strategy",
    "question": "English question text",
    "question_en": "English question text",
    "answer_type": "analysis" | "rating" | "yes_no" | "open",
    "priority": 1 | 2 | 3,
    "links_to": {{
      "key_points": [0, 2],
      "risks": [],
      "mitigations": []
    }}
  }}

- "search_hints" must be a flat array of 6-10 strings.
- Do not output any text outside JSON. Do not add explanations or comments.
"""

# ========== LLM question generation ==========
def call_llm_for_dimension_questions(
    client: UnifiedChatClient,
    dimension_name: str,
    dim_payload: Dict[str, Any],
    min_q: int,
    max_q: int,
    dim_config: Dict[str, Any],
    trace: Dict[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Call the LLM to generate a customized question list for one dimension, including links_to.
    Returns (questions, search_hints).
    """

    key_points = dim_payload.get("key_points", []) or []
    kp_cnt = len(key_points)

    # Adjust the target range based on information volume without exceeding dimension config limits.
    if kp_cnt >= 8:
        target_min, target_max = max(min_q, dim_config["min_q"]), dim_config["max_q"]
    elif 4 <= kp_cnt <= 7:
        target_min, target_max = max(min_q, dim_config["min_q"] - 1), min(max_q, dim_config["max_q"])
    elif kp_cnt > 0:
        target_min, target_max = max(min_q, 4), min(max_q, dim_config["max_q"] - 1)
    else:
        target_min, target_max = min_q, min(max_q, 6)

    if trace is not None:
        trace["target_min"] = target_min
        trace["target_max"] = target_max
        trace["input_stats"] = _dimension_input_stats(dimension_name, dim_payload)

    aspects = dim_config.get("aspects", [])
    aspects_str = json.dumps(aspects, ensure_ascii=False, indent=2)

    payload_str = json.dumps(dim_payload, ensure_ascii=False, indent=2)
    focus = dim_config.get("focus", "")

    # Dimension content overview for the model.
    overview = dim_payload.get("meta", {})
    overview_str = json.dumps(overview, ensure_ascii=False, indent=2)

    prompt = (
        QUESTION_PROMPT_TEMPLATE
        .replace("{dimension_name}", dimension_name)
        .replace("{dimension_focus}", focus)
        .replace("{business_archetype_guide}", BUSINESS_ARCHETYPE_GUIDE.strip())
    )

    user_content = (
        prompt
        + "\n\n=== Aspect configuration for this dimension ===\n"
        + aspects_str
        + "\n\n=== Current dimension content overview ===\n"
        + overview_str
        + "\n\n=== Current dimension summary payload ===\n"
        + payload_str
        + "\n\n=== Quantity guidance ===\n"
        + f"- Recommended question range: [{target_min}, {target_max}]. Stay within this range when possible.\n"
        + "- Cover at least 3-5 key aspects, including at least one platform or long-term aspect when such an aspect exists.\n"
        + "- Generate at least 2 rating questions and 3 analysis questions when the source information supports them.\n"
        + "- Strictly follow the JSON structure above."
    )
    _log(
        "DIM",
        f"{dimension_name}: target_questions=[{target_min}, {target_max}] "
        f"key_points={len(key_points)} risks={len(dim_payload.get('risks', []) or [])} "
        f"mitigations={len(dim_payload.get('mitigations', []) or [])} aspects={len(aspects)}"
    )
    _log("PREVIEW", f"{dimension_name}: summary_preview={_preview_text(dim_payload.get('summary', ''))}")
    _log("LLM", f"{dimension_name}: prompt_chars={len(user_content)} model={OPENAI_MODEL} provider={PROVIDER}")

    messages = [
        {
            "role": "system",
            "content": "You are a sector-agnostic venture, business, product, technical, and operating diligence expert designing structured questions for an AI system.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    call_start = time.perf_counter()
    raw = ""
    usage_data = None
    data: Dict[str, Any] = {}
    json_parse_failed = False
    last_error: Exception | None = None
    for attempt in range(1, max(1, LLM_JSON_RETRIES) + 1):
        try:
            attempt_messages = messages
            if attempt > 1:
                retry_prompt = (
                    user_content
                    + "\n\nImportant: the previous response was not valid JSON. "
                    "Return only one JSON object with exactly one top-level key named questions, "
                    "and keep all question objects within the schema."
                )
                attempt_messages = [
                    messages[0],
                    {"role": "user", "content": retry_prompt},
                ]

            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=attempt_messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2200,
            )
        except Exception as e:
            elapsed = time.perf_counter() - call_start
            _log("ERROR", f"{dimension_name}: LLM question generation failed elapsed_sec={elapsed:.2f}: {e}")
            if trace is not None:
                trace["llm_call"] = {
                    "ok": False,
                    "elapsed_sec": round(elapsed, 2),
                    "error": str(e),
                }
            raise

        raw = resp.choices[0].message.content
        elapsed = time.perf_counter() - call_start
        usage = getattr(resp, "usage", None)
        if usage is not None:
            usage_data = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        _log(
            "LLM",
            f"{dimension_name}: call_done attempt={attempt} elapsed_sec={elapsed:.2f} raw_chars={len(raw or '')} "
            f"usage={json.dumps(usage_data, ensure_ascii=False)}"
        )
        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            json_parse_failed = True
            last_error = e
            _log("WARN", f"{dimension_name}: JSON parse failed for generated questions on attempt {attempt}: {e}")
            _log("WARN", f"{dimension_name}: raw_response_preview={_preview_text(raw or '', 1000)}")
            if attempt >= max(1, LLM_JSON_RETRIES):
                data = {}
                break

    if trace is not None:
        trace["llm_call"] = {
            "ok": not json_parse_failed or bool(data),
            "elapsed_sec": round(time.perf_counter() - call_start, 2),
            "raw_chars": len(raw or ""),
            "usage": usage_data,
        }
        if json_parse_failed:
            trace["json_parse_failed"] = True
        if last_error is not None:
            trace["json_parse_error"] = str(last_error)

    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    raw_hints = data.get("search_hints", [])
    search_hints = [
        str(h).strip() for h in raw_hints
        if isinstance(h, str) and str(h).strip()
    ] if isinstance(raw_hints, list) else []
    if not search_hints:
        _log("WARN", f"{dimension_name}: LLM returned no search_hints; alignment scoring will use fallback.")
    if trace is not None:
        trace["raw_question_count"] = len(questions)
        trace["search_hints"] = search_hints
    _log("DIM", f"{dimension_name}: raw_question_count={len(questions)} search_hints_count={len(search_hints)}")

    cleaned: List[Dict[str, Any]] = []
    rating_count = 0
    analysis_count = 0
    linked_to_kp_count = 0
    platform_aspect_used = 0
    skipped_non_dict = 0
    skipped_missing_text = 0
    invalid_answer_type_count = 0
    invalid_priority_count = 0
    fallback_questions_added = []
    truncation_applied = False

    for q in questions:
        if not isinstance(q, dict):
            skipped_non_dict += 1
            continue

        aspect = str(q.get("aspect", "")).strip()
        question_text = str(
            q.get("question")
            or q.get("question_en")
            or ""
        ).strip()
        q_en = str(q.get("question_en", "")).strip()
        answer_type = str(q.get("answer_type", "analysis")).strip().lower()
        priority = q.get("priority", 2)
        links_to = q.get("links_to") or {}

        if not question_text:
            skipped_missing_text += 1
            continue
        if not q_en:
            q_en = question_text

        if answer_type not in ["analysis", "rating", "yes_no", "open"]:
            invalid_answer_type_count += 1
            answer_type = "analysis"

        try:
            priority_int = int(priority)
        except Exception:
            priority_int = 2
            invalid_priority_count += 1
        if priority_int < 1 or priority_int > 3:
            priority_int = 2
            invalid_priority_count += 1

        if not aspect:
            aspect = "general"

        # Clean links_to structure.
        if not isinstance(links_to, dict):
            links_to = {}
        kp_idx = links_to.get("key_points", [])
        rk_idx = links_to.get("risks", [])
        mt_idx = links_to.get("mitigations", [])

        def _clean_index_list(v, max_len: int):
            if not isinstance(v, list):
                return []
            cleaned_idx = []
            for x in v:
                try:
                    ix = int(x)
                    if 0 <= ix < max_len:
                        cleaned_idx.append(ix)
                except Exception:
                    continue
            return cleaned_idx

        kp_idx_clean = _clean_index_list(kp_idx, len(key_points))
        rk_idx_clean = _clean_index_list(
            rk_idx,
            len(dim_payload.get("risks", []) or [])
        )
        mt_idx_clean = _clean_index_list(
            mt_idx,
            len(dim_payload.get("mitigations", []) or [])
        )

        if kp_idx_clean:
            linked_to_kp_count += 1

        if answer_type == "rating":
            rating_count += 1
        if answer_type == "analysis":
            analysis_count += 1

        if aspect in PLATFORM_ASPECT_IDS:
            platform_aspect_used += 1

        cleaned.append(
            {
                "aspect": aspect,
                "question": question_text,
                "question_en": q_en,
                "answer_type": answer_type,
                "priority": priority_int,
                "links_to": {
                    "key_points": kp_idx_clean,
                    "risks": rk_idx_clean,
                    "mitigations": mt_idx_clean,
                },
            }
        )

    # ===== Dimension-specific fallback questions =====
    if cleaned and key_points:
        # --- Team dimension: add a resume/track-record question if missing. ---
        if dimension_name == "team":
            has_team_bio_q = any(
                _looks_like_team_bio_question(
                    q.get("question", ""), q.get("question_en", "")
                )
                for q in cleaned
            )
            if not has_team_bio_q:
                _log("INFO", f"{dimension_name}: no clear resume/track-record question detected; adding one fallback question.")
                fallback_questions_added.append("team_resume_driven")
                extra_q_team = {
                    "aspect": "leadership_experience",
                    "question": (
                        "Based on the proposal's description of the core team members and project leaders, "
                        "including education, domain-specific technical, product, operating, or commercial experience, and prior major-project records, "
                        "how would you assess the team's ability to advance this project toward its intended validation, "
                        "deployment, operation, scale-up, or commercialization path? If key track-record details are not specific or verifiable, first "
                        "state that information gap and then discuss its impact on the assessment."
                    ),
                    "question_en": (
                        "Based on the proposal's description of the core team members and project leaders "
                        "(education, domain-specific technical, product, operating, or commercial experience, and track record in previous major projects), "
                        "how would you assess the team's overall ability to drive this project toward its intended validation, "
                        "deployment, operation, scale-up, or commercialization path? If the proposal does not provide sufficiently specific "
                        "or verifiable track record information, please first state this information gap and then "
                        "discuss its impact on your assessment."
                    ),
                    "answer_type": "analysis",
                    "priority": 1,
                    "links_to": {
                        "key_points": list(range(len(key_points))),
                        "risks": [],
                        "mitigations": [],
                    },
                }
                cleaned.insert(0, extra_q_team)

        # --- Strategy/objectives dimensions: add a market-driven question if missing. ---
        if dimension_name in ("strategy", "objectives"):
            has_market_q = any(
                _looks_like_market_question(
                    q.get("question", ""), q.get("question_en", "")
                )
                for q in cleaned
            )
            if not has_market_q:
                _log("INFO", f"{dimension_name}: no clear market-driven question detected; adding one fallback question.")
                fallback_questions_added.append("market_driven")
                if dimension_name == "strategy":
                    aspect_id = "commercialization_and_market_entry"
                else:
                    aspect_id = "unmet_need_alignment"

                extra_q_market = {
                    "aspect": aspect_id,
                    "question": (
                        "Based on the proposal's description of target users/customers/beneficiaries, market or need size, growth, main alternatives, "
                        "buyers/payers/funders, channels, or target segments, assess the initiative's positioning, competitive or substitute pressure, "
                        "and proposed adoption, rollout, market-entry, or commercialization strategy in the intended niche. If the "
                        "proposal lacks concrete demand, stakeholder, market-size, or competitive-landscape data, first state that "
                        "information gap and then analyze its impact on the assessment."
                    ),
                    "question_en": (
                        "Drawing on the proposal's description of target users/customers/beneficiaries, market or need size, growth, main alternatives, "
                        "buyers/payers/funders, channels, or target segments, how would you assess the initiative's positioning, competitive "
                        "or substitute pressure and the soundness of its adoption, rollout, market-entry, or commercialization strategy in the intended "
                        "niche? If the proposal does not provide concrete data on demand, stakeholders, market size, or the competitive "
                        "landscape, please first state this information gap and then discuss its impact on your "
                        "assessment."
                    ),
                    "answer_type": "analysis",
                    "priority": 1,
                    "links_to": {
                        "key_points": list(range(len(key_points))),
                        "risks": [],
                        "mitigations": [],
                    },
                }
                cleaned.insert(0, extra_q_market)

        cleaned = _dedupe_question_items(cleaned)

        def _recount_question_types(items: List[Dict[str, Any]]) -> tuple[int, int, int]:
            ratings = sum(1 for item in items if item.get("answer_type") == "rating")
            analyses = sum(1 for item in items if item.get("answer_type") == "analysis")
            linked = sum(1 for item in items if (item.get("links_to") or {}).get("key_points"))
            return ratings, analyses, linked

        rating_count, analysis_count, linked_to_kp_count = _recount_question_types(cleaned)

        while rating_count < 2 and len(cleaned) < target_max:
            kp_index = min(rating_count, len(key_points) - 1)
            fallback_questions_added.append("rating_coverage")
            cleaned.append(
                {
                    "aspect": "evidence_strength",
                    "question": (
                        "On a 1-5 scale, how strong and specific is the proposal evidence supporting "
                        f"this {dimension_name} point, and what exact missing evidence would most improve the score?"
                    ),
                    "question_en": (
                        "On a 1-5 scale, how strong and specific is the proposal evidence supporting "
                        f"this {dimension_name} point, and what exact missing evidence would most improve the score?"
                    ),
                    "answer_type": "rating",
                    "priority": 2,
                    "links_to": {
                        "key_points": [kp_index],
                        "risks": [],
                        "mitigations": [],
                    },
                }
            )
            rating_count, analysis_count, linked_to_kp_count = _recount_question_types(cleaned)

        min_linked = max(3, len(cleaned) // 2)
        if linked_to_kp_count < min_linked and len(cleaned) < target_max:
            linked_indexes = {
                ix
                for item in cleaned
                for ix in ((item.get("links_to") or {}).get("key_points") or [])
            }
            uncovered = [ix for ix in range(len(key_points)) if ix not in linked_indexes]
            selected = uncovered[:3] or list(range(min(3, len(key_points))))
            fallback_questions_added.append("key_point_coverage")
            cleaned.append(
                {
                    "aspect": "key_point_traceability",
                    "question": (
                        "For the cited key project points, what concrete proposal evidence supports each point, "
                        "what remains unverified, and how should those gaps affect the evaluation?"
                    ),
                    "question_en": (
                        "For the cited key project points, what concrete proposal evidence supports each point, "
                        "what remains unverified, and how should those gaps affect the evaluation?"
                    ),
                    "answer_type": "analysis",
                    "priority": 2,
                    "links_to": {
                        "key_points": selected,
                        "risks": [],
                        "mitigations": [],
                    },
                }
            )
            rating_count, analysis_count, linked_to_kp_count = _recount_question_types(cleaned)

    # Enforce the upper bound using priority order.
    if cleaned:
        if len(cleaned) > target_max:
            # Keep priority 1 first, then 2, then 3.
            cleaned.sort(key=lambda q: q.get("priority", 2))
            original_len = len(cleaned)
            cleaned = cleaned[:target_max]
            truncation_applied = True
            _log(
                "INFO",
                f"{dimension_name}: generated {original_len} questions; truncated to {len(cleaned)} by priority "
                f"(target_max={target_max})."
            )
        elif len(cleaned) < target_min:
            _log(
                "WARN",
                f"{dimension_name}: only {len(cleaned)} questions after cleaning; below target_min={target_min}."
            )

    # Sanity checks only; these do not force retries.
    if cleaned:
        if rating_count < 2:
            _log(
                "WARN",
                f"{dimension_name}: only {rating_count} rating question(s); quantitative scoring may be weak."
            )
        if analysis_count < 3 and len(cleaned) >= 5:
            _log(
                "WARN",
                f"{dimension_name}: only {analysis_count} analysis question(s); qualitative analysis may be shallow."
            )
        if linked_to_kp_count < len(cleaned) // 2:
            _log(
                "WARN",
                f"{dimension_name}: only {linked_to_kp_count}/{len(cleaned)} questions link to key_points; inspect manually."
            )
        if dimension_name == "innovation" and platform_aspect_used == 0:
            _log(
                "WARN",
                f"{dimension_name}: no platform/extensibility aspect detected; consider adding long-term platform questions."
            )

    distribution = _question_distribution(cleaned)
    _log("DIM", f"{dimension_name}: cleaned_question_count={len(cleaned)}")
    _log(
        "CLEAN",
        f"{dimension_name}: skipped_non_dict={skipped_non_dict} "
        f"skipped_missing_text={skipped_missing_text} "
        f"invalid_answer_type_fixed={invalid_answer_type_count} "
        f"invalid_priority_fixed={invalid_priority_count} "
        f"fallbacks={fallback_questions_added or []} "
        f"truncation_applied={truncation_applied}"
    )
    _log("DIM", f"{dimension_name}: answer_types={json.dumps(distribution['answer_types'], ensure_ascii=False)}")
    _log("DIM", f"{dimension_name}: priorities={json.dumps(distribution['priorities'], ensure_ascii=False)}")
    _log("DIM", f"{dimension_name}: aspects={json.dumps(distribution['aspects'], ensure_ascii=False)}")
    _log(
        "DIM",
        f"{dimension_name}: linked_to key_points={distribution['linked_to_key_points']} "
        f"risks={distribution['linked_to_risks']} mitigations={distribution['linked_to_mitigations']}"
    )

    if trace is not None:
        trace.update(
            {
                "cleaned_question_count": len(cleaned),
                "skipped_non_dict": skipped_non_dict,
                "skipped_missing_text": skipped_missing_text,
                "invalid_answer_type_count": invalid_answer_type_count,
                "invalid_priority_count": invalid_priority_count,
                "fallback_questions_added": fallback_questions_added,
                "truncation_applied": truncation_applied,
                "distribution": distribution,
                "question_previews": [
                    {
                        "aspect": q.get("aspect", ""),
                        "answer_type": q.get("answer_type", ""),
                        "priority": q.get("priority", ""),
                        "question_preview": _preview_text(q.get("question", ""), 220),
                        "question_en_preview": _preview_text(q.get("question_en", ""), 220),
                    }
                    for q in cleaned[:5]
                ],
            }
        )

    return cleaned, search_hints

# ========== Main flow ==========

def run_generate_questions(
    proposal_id: str,
    min_q_per_dim: int = 5,
    max_q_per_dim: int = 10,
):
    stage_start = time.perf_counter()
    _log("STAGE_3", "Starting Stage 3: dimension-specific evaluation-question generation")
    _log(
        "STAGE_3",
        "Purpose: convert dimensions_v2.json summaries into grounded reviewer questions "
        "for downstream llm_answering."
    )
    _log(
        "STAGE_3",
        f"proposal_id={proposal_id} provider={PROVIDER} model={OPENAI_MODEL} "
        f"min_q_per_dim={min_q_per_dim} max_q_per_dim={max_q_per_dim}"
    )
    input_path = EXTRACTED_DIR / proposal_id / "dimensions_v2.json"
    per_proposal_output_dir = QUESTIONS_DIR / proposal_id
    _log("STAGE_3", f"input_dimensions_path={input_path}")
    _log("STAGE_3", f"per_proposal_output_dir={per_proposal_output_dir}")
    client = get_openai_client()
    dimensions = load_dimensions(proposal_id)

    all_dim_questions: Dict[str, Any] = {}
    audit: Dict[str, Any] = {
        "stage": "stage_3_generate_questions",
        "purpose": "Generate grounded, dimension-specific evaluation questions from dimensions_v2.json.",
        "proposal_id": proposal_id,
        "provider": PROVIDER,
        "model": OPENAI_MODEL,
        "min_q_per_dim": min_q_per_dim,
        "max_q_per_dim": max_q_per_dim,
        "input_path": str((EXTRACTED_DIR / proposal_id / "dimensions_v2.json").resolve()),
        "dimensions": {},
    }

    _write_progress(0, len(DIMENSION_NAMES), proposal_id)
    for dim_idx, dim in enumerate(DIMENSION_NAMES):
        dim_start = time.perf_counter()
        dim_data = dimensions.get(dim, {})
        dim_config = DIMENSION_CONFIG.get(dim)

        if not dim_config:
            _log("WARN", f"{dim}: missing DIMENSION_CONFIG; skipping.")
            audit["dimensions"][dim] = {
                "skipped": True,
                "reason": "missing_dimension_config",
                "elapsed_sec": round(time.perf_counter() - dim_start, 2),
            }
            _write_progress(dim_idx + 1, len(DIMENSION_NAMES), proposal_id)
            continue

        _log("DIM", f"start dimension={dim} index={dim_idx+1}/{len(DIMENSION_NAMES)}")

        dim_payload = build_dimension_payload(dim, dim_data)
        _log_dimension_payload(dim, dim_payload)
        dim_trace: Dict[str, Any] = {
            "skipped": False,
            "dimension": dim,
            "configured_min_q": dim_config.get("min_q"),
            "configured_max_q": dim_config.get("max_q"),
            "configured_aspects": [a.get("id") for a in dim_config.get("aspects", []) if isinstance(a, dict)],
        }
        questions, dim_search_hints = call_llm_for_dimension_questions(
            client=client,
            dimension_name=dim,
            dim_payload=dim_payload,
            min_q=min_q_per_dim,
            max_q=max_q_per_dim,
            dim_config=dim_config,
            trace=dim_trace,
        )

        dim_qs_with_id = []
        for idx, q in enumerate(questions, start=1):
            qid = f"{dim}_q{idx:02d}"
            item = dict(q)
            item["qid"] = qid
            item["dimension"] = dim
            dim_qs_with_id.append(item)

        elapsed_dim = time.perf_counter() - dim_start
        dim_trace["elapsed_sec"] = round(elapsed_dim, 2)
        dim_trace["final_question_count"] = len(dim_qs_with_id)
        audit["dimensions"][dim] = dim_trace
        _log(
            "DIM",
            f"done dimension={dim} final_questions={len(dim_qs_with_id)} "
            f"search_hints={len(dim_search_hints)} "
            f"configured_range=[{dim_config['min_q']}, {dim_config['max_q']}] elapsed_sec={elapsed_dim:.2f}"
        )

        all_dim_questions[dim] = {
            "dimension": dim,
            "questions": dim_qs_with_id,
            "search_hints": dim_search_hints,
        }
        _write_progress(dim_idx + 1, len(DIMENSION_NAMES), proposal_id)

    out_dir = QUESTIONS_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_at_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build simplified format (plain string question lists) consumed by llm_answering.py.
    qs_simple: Dict[str, Any] = {}
    for dim in DIMENSION_NAMES:
        dim_block = all_dim_questions.get(dim, {})
        q_items = dim_block.get("questions", []) if isinstance(dim_block, dict) else []
        q_texts = [
            str(q.get("question", "") or q.get("question_en", "")).strip()
            for q in q_items
            if isinstance(q, dict) and str(q.get("question", "") or q.get("question_en", "")).strip()
        ]
        dim_hints = dim_block.get("search_hints", []) if isinstance(dim_block, dict) else []
        qs_simple[dim] = {
            "dimension": dim,
            "questions": q_texts,
            "search_hints": dim_hints,
            "source_proposal_id": proposal_id,
        }

    simple_obj = {
        "proposal_id": proposal_id,
        "generated_at": generated_at_utc,
        "model": OPENAI_MODEL,
        "provider": PROVIDER,
        **qs_simple,
    }

    # ===== 1) Write simplified version for llm_answering to the per-pid directory =====
    simple_out_path = out_dir / "generated_questions.json"
    simple_out_path.write_text(
        json.dumps(simple_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log("OK", f"per-proposal question set written path={simple_out_path}")

    # ===== 2) Write active simplified version for llm_answering default lookup =====
    active_out_path = CONFIG_QS_DIR / "generated_questions.json"
    active_out_path.write_text(
        json.dumps(simple_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log("OK", f"active question set for llm_answering written path={active_out_path}")

    # ===== 3) Write detailed version with full question objects to the per-pid directory =====
    detail_output_obj = {
        "proposal_id": proposal_id,
        "generated_at": generated_at_utc,
        "model": OPENAI_MODEL,
        "provider": PROVIDER,
        "dimensions": all_dim_questions,
    }
    detail_out_path = out_dir / "generated_questions_detail.json"
    detail_out_path.write_text(
        json.dumps(detail_output_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log("OK", f"detailed question set written path={detail_out_path}")

    total_questions = sum(
        len((block or {}).get("questions", []))
        for block in all_dim_questions.values()
        if isinstance(block, dict)
    )
    elapsed_stage = time.perf_counter() - stage_start
    audit.update(
        {
            "generated_at": generated_at_utc,
            "elapsed_sec": round(elapsed_stage, 2),
            "total_questions": total_questions,
            "simple_output_path": str(simple_out_path.resolve()),
            "active_output_path": str(active_out_path.resolve()),
            "detail_output_path": str(detail_out_path.resolve()),
        }
    )
    audit_path = out_dir / "stage_questions_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _log("OK", f"question-generation audit written path={audit_path}")

    _log("SUMMARY", f"total_questions={total_questions} elapsed_sec={elapsed_stage:.2f}")
    for dim in DIMENSION_NAMES:
        block = all_dim_questions.get(dim, {})
        q_count = len(block.get("questions", [])) if isinstance(block, dict) else 0
        _log("SUMMARY", f"{dim}={q_count} questions")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: generate differentiated question sets from dimensions_v2.json with links_to and anti-hallucination rules"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="Proposal ID corresponding to src/data/extracted/<proposal_id>",
    )
    parser.add_argument(
        "--min_q_per_dim",
        type=int,
        default=5,
        help="Minimum number of questions per dimension (recommended lower bound, default: 5)",
    )
    parser.add_argument(
        "--max_q_per_dim",
        type=int,
        default=10,
        help="Maximum number of questions per dimension (recommended upper bound, default: 10)",
    )
    parser.add_argument(
        "--llm_provider",
        required=False,
        help="LLM provider: openai, deepseek, or gemini; defaults to PROVIDER from .env",
    )

    args = parser.parse_args()

    global PROVIDER, OPENAI_MODEL
    if args.llm_provider:
        PROVIDER = args.llm_provider.lower()
        OPENAI_MODEL = default_model(PROVIDER)

    if args.proposal_id:
        pid = args.proposal_id
    else:
        pid = find_latest_extracted_proposal_id()

    run_generate_questions(
        proposal_id=pid,
        min_q_per_dim=args.min_q_per_dim,
        max_q_per_dim=args.max_q_per_dim,
    )


if __name__ == "__main__":
    main()
