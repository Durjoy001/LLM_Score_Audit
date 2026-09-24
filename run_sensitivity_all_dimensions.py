#!/usr/bin/env python3
"""
Content Sensitivity Experiment — All Proposals, All Dimensions.

Default mode (hardcoded):
  Runs the sensitivity experiment (Baseline / PLUS / MINUS) for:
  - strategy   : R3 Revenue, R4 Regulatory (PLUS); R6 Partners, R8 Timing (MINUS)
  - advantages : R2 IP Status, R3 Performance Proof (PLUS); R1 Mechanism Novelty, R5 Defensibility (MINUS)
  - objectives : R2 Market Size, R3 Buyer Pathway (PLUS); R1 Problem Clarity, R7 Problem-Sol Fit (MINUS)

Adaptive mode (--adaptive):
  For each proposal × dimension, an LLM generates:
  - PLUS injection: domain-specific evidence covering ALL 8 rubrics
  - MINUS stripping: finds and replaces the key evidence for ALL 8 rubrics
  Hypothesis checks jump from 4 to 16 per proposal.

Usage:
    python3 run_sensitivity_all_dimensions.py
    python3 run_sensitivity_all_dimensions.py --dimensions advantages objectives
    python3 run_sensitivity_all_dimensions.py --pids p1 p8 --dimensions strategy
    python3 run_sensitivity_all_dimensions.py --skip_score   # print from cached scores
    python3 run_sensitivity_all_dimensions.py --adaptive
    python3 run_sensitivity_all_dimensions.py --adaptive --skip_generate
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

ROOT        = Path(__file__).resolve().parent
REPORT_DIR  = ROOT / "src" / "data" / "reports"
RESULTS_DIR = ROOT / "src" / "data" / "sensitivity_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ADAPTIVE_CONTENT_DIR = RESULTS_DIR / "adaptive_content"
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.backend.utils.llm_provider import chat_json

ALL_PIDS   = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]
PID_SUFFIX = "__openai"

# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

DIMENSIONS = {

    # ── Strategy ──────────────────────────────────────────────────────────────
    "strategy": {
        "tool":       "src/tools/generate_strategy_rubric_scores.py",
        "scores_dir": "src/data/strategy_rubric_scores",
        "scores_key": "strategy_rubrics",
        "result_file": "strategy_rubric_scores.json",
        "rubrics": [
            ("strategy_R1_gtm",        "R1 Go-to-Market",    "doc-verifiable"),
            ("strategy_R2_milestones", "R2 Milestones",       "doc-verifiable"),
            ("strategy_R3_revenue",    "R3 Revenue",          "doc-verifiable"),
            ("strategy_R4_regulatory", "R4 Regulatory",       "doc-verifiable"),
            ("strategy_R5_moat",       "R5 Moat",             "judgment-dep"),
            ("strategy_R6_partners",   "R6 Partners",         "judgment-dep"),
            ("strategy_R7_team_fit",   "R7 Team Fit",         "judgment-dep"),
            ("strategy_R8_timing",     "R8 Timing",           "judgment-dep"),
        ],
        # Hardcoded mode targets
        "plus_targets":  ["strategy_R3_revenue", "strategy_R4_regulatory"],
        "minus_targets": ["strategy_R6_partners", "strategy_R8_timing"],
        "plus_injection": (
            "\n**Revenue Model:** The company will commercialize via a B2B SaaS licensing model "
            "at $240,000 per hospital system per year (enterprise tier), $48,000 per year "
            "(mid-market tier). Unit economics: estimated cost to serve = $18,000/year per customer; "
            "gross margin = 62.5%. Payment structure: annual upfront license with quarterly "
            "usage-based overage at $0.40 per processed patient record. Year-1 revenue target: "
            "$2.4M from 10 enterprise contracts currently in the sales pipeline.\n\n"
            "**Regulatory Strategy:** The product is classified as a Software as a Medical Device "
            "(SaMD) under FDA 21 CFR Part 820. The team filed a Pre-Submission (Q-Sub) with FDA "
            "(reference Q260312-01) and received written feedback confirming the "
            "510(k) pathway via predicate device K213456. De Novo classification has been ruled "
            "out. CE mark submission under EU MDR Article 51 is in progress, with a "
            "notified body (TÜV SÜD) engaged under a formal contract.\n"
        ),
        "plus_marker":  "##### Technology and product innovation",
        "minus_strip_bullets": ["Merck", "Fujifilm", "Pfizer", "Roche", "AstraZeneca",
                                "Novartis", "GSK", "Sanofi", "phased funding strategy",
                                "Phased funding strategy"],
        "minus_inline_subs": [
            (r"Merck|Fujifilm|Pfizer|Roche|AstraZeneca|Novartis|GSK|Sanofi", "strategic partner"),
            (r"[Pp]hased funding strategy[^.]*\.", ""),
        ],
    },

    # ── Advantages ────────────────────────────────────────────────────────────
    "advantages": {
        "tool":       "src/tools/generate_advantages_rubric_scores.py",
        "scores_dir": "src/data/advantages_rubric_scores",
        "scores_key": "advantages_rubrics",
        "result_file": "advantages_rubric_scores.json",
        "rubrics": [
            ("advantages_R1_mechanism_novelty",    "R1 Mechanism Novelty",    "doc-verifiable"),
            ("advantages_R2_ip_status",            "R2 IP Status",            "doc-verifiable"),
            ("advantages_R3_performance_proof",    "R3 Performance Proof",    "doc-verifiable"),
            ("advantages_R4_competitor_benchmark", "R4 Competitor Benchmark", "doc-verifiable"),
            ("advantages_R5_defensibility",        "R5 Defensibility",        "judgment-dep"),
            ("advantages_R6_platform_potential",   "R6 Platform Potential",   "judgment-dep"),
            ("advantages_R7_validation_signals",   "R7 Validation Signals",   "judgment-dep"),
            ("advantages_R8_adoption_readiness",   "R8 Adoption Readiness",   "judgment-dep"),
        ],
        "plus_targets":  ["advantages_R2_ip_status", "advantages_R3_performance_proof"],
        "minus_targets": ["advantages_R1_mechanism_novelty", "advantages_R5_defensibility"],
        "plus_injection": (
            "\n**IP Portfolio:** The company holds US patent 11,234,567 (granted October 2024) "
            "covering the core LNP stabilization technology, with two continuation applications "
            "pending (US 18/456,789; US 18/456,790) covering AI-guided delivery optimization and "
            "excipient composition. A European counterpart (EP4123456) was granted March 2026, "
            "with PCT filings active in Japan and China.\n\n"
            "**Performance vs Benchmark:** Head-to-head in vitro comparison against the leading "
            "commercial LNP formulation (MC3-based) demonstrated: transfection efficiency 94.3% "
            "vs 71.2% (p<0.001, n=6 replicates); cytotoxicity IC50 of 847 µg/mL vs 312 µg/mL "
            "(2.7× safer); shelf stability at 4°C of 18 months vs 6 months (3× longer). Data "
            "generated under GLP conditions, protocol on file.\n"
        ),
        "plus_marker":  "##### Resources and feasibility",
        "minus_strip_bullets": [
            "lung-targeted stable lipid", "generative design", "recurrent neural",
            " RNN", "AI drug screening", "AI model", "AI-guided",
            "advanced gene engineering", "precision delivery",
            "proprietary LNP", "AI integration",
            "novel protein", "first-in-class", "breakthrough", "proprietary platform",
            "unique mechanism", "innovative platform", "novel approach", "novel formulation",
        ],
        "minus_inline_subs": [
            (r"lung-targeted stable lipid nanoparticles?(?:\s*\(LNPs?\))?", "lipid nanoparticles"),
            (r"generative design and recurrent neural networks?(?:\s*\(RNN\))?", "computational screening"),
            (r"[Aa]dvanced gene engineering techniques?", "standard formulation methods"),
            (r"\b[Aa][Ii]-guided( delivery)?", "standard"),
            (r"[Ii]ntegration of AI models?", "use of computational tools"),
            (r"AI models? (?:is|are|were|has been) a distinguishing feature[^.]*\.", ""),
            (r"AI models? enable[^.]*\.", ""),
            (r"using AI models?[^.]*\.", "."),
            (r"\b(?:novel|innovative|unique|breakthrough)\s+(?:protein|platform|formulation|technology|approach|method|design|mechanism|pipeline)\b",
             "standard approach"),
            (r"[^.]*\bpatent(?:s|ed|ing|[-\s]pending|[-\s]filed|[-\s]granted)?\b[^.]*\.",
             " No patent protection is described."),
            (r"[^.]*\b(?:[Pp]roprietary|IP portfolio|exclusive rights?|exclusive licen[sc]e)\b[^.]*\.",
             " No proprietary IP or exclusivity barrier is described."),
        ],
    },

    # ── Objectives ────────────────────────────────────────────────────────────
    "objectives": {
        "tool":       "src/tools/generate_objectives_rubric_scores.py",
        "scores_dir": "src/data/objectives_rubric_scores",
        "scores_key": "objectives_rubrics",
        "result_file": "objectives_rubric_scores.json",
        "rubrics": [
            ("objectives_R1_problem_clarity",     "R1 Problem Clarity",    "doc-verifiable"),
            ("objectives_R2_market_size",         "R2 Market Size",        "doc-verifiable"),
            ("objectives_R3_buyer_pathway",       "R3 Buyer Pathway",      "doc-verifiable"),
            ("objectives_R4_unmet_need_evidence", "R4 Unmet Need Evidence","doc-verifiable"),
            ("objectives_R5_why_now",             "R5 Why Now",            "judgment-dep"),
            ("objectives_R6_competitive_context", "R6 Competitive Context","judgment-dep"),
            ("objectives_R7_problem_solution_fit","R7 Problem-Sol Fit",    "judgment-dep"),
            ("objectives_R8_addressability",      "R8 Addressability",     "judgment-dep"),
        ],
        "plus_targets":  ["objectives_R2_market_size", "objectives_R3_buyer_pathway"],
        "minus_targets": ["objectives_R1_problem_clarity", "objectives_R7_problem_solution_fit"],
        "plus_injection": (
            "\n**Market Size:** The global antiviral therapeutics market was valued at $57.3B in "
            "2024 (Source: Grand View Research, 2024), growing at 8.2% CAGR. The serviceable "
            "addressable market for RNA-based broad-spectrum antivirals in high-risk populations "
            "is estimated at $4.2B by 2028 (Source: EvaluatePharma, 2025). The initial target "
            "segment — immunocompromised patients with recurrent respiratory viral infections — "
            "represents approximately 18 million patients in the US and EU combined, with an "
            "average treatment cost of $12,000–$18,000 per episode.\n\n"
            "**Buyer Pathway:** The company has signed a Letter of Intent (LOI, dated April 2026) "
            "with MedHealth Network (42 hospital sites, 180,000 patients/year) for a pilot "
            "procurement of the antiviral therapy at $8,500 per treatment course. The LOI includes "
            "a 12-month exclusivity window and a defined path to a Master Supply Agreement upon "
            "Phase II trial completion. Two additional health system LOIs are in final review.\n"
        ),
        "plus_marker":  "##### Implementation path and strategy",
        "minus_strip_bullets": [
            "high-risk exposure", "high-risk population", "immunocompromised",
            "COVID-19", "SARS-CoV-2", "specifically designed to", "specifically targets",
            "directly addresses", "addresses the unmet need", "targets a critical need",
        ],
        "minus_inline_subs": [
            (r"high-risk (?:exposure )?populations?", "patients"),
            (r"COVID-?19(?:[^,\.]*)", "viral disease"),
            (r"SARS-CoV-2(?:[^,\.]*)", "the virus"),
            (r"\bimmunocompromised\b", "at-risk"),
            (r"18 million patients?[^.]*\.", "many patients."),
            (r"lung-targeted[^,\.]*(?:therapy|RNA|treatment)[^,\.]*\.", "therapeutic approach."),
            (r"broad-spectrum antiviral (?:status|therapy|RNA/LNP therapy)", "antiviral treatment"),
            (r"specifically (?:targets?|designed to|addresses?)[^.]*\.",
             "The connection between problem and solution is not made explicit."),
            (r"[Tt]he project targets? a critical need[^.]*\.",
             "The project addresses a general medical area."),
            (r"aligns? with (?:the )?(?:critical |urgent |pressing |unmet )?need[^.]*\.",
             "No explicit alignment between problem and solution is stated."),
            (r"directly addresses? the (?:problem|need|gap)[^.]*\.",
             "No direct problem-solution link is described."),
            (r"proactive approach to addressing[^.]*\.", "approach."),
            (r"objectives? align with[^.]*\.",
             "No explicit statement of objective alignment is made."),
        ],
    },

    # ── Team ──────────────────────────────────────────────────────────────────
    # Added 2026-09-16, adaptive mode only: no hardcoded PLUS/MINUS content exists,
    # so the hardcoded fallback is disabled (see create_adaptive_variants).
    "team": {
        "tool":       "src/tools/generate_team_rubric_scores.py",
        "scores_dir": "src/data/team_rubric_scores",
        "scores_key": "team_rubrics",
        "result_file": "team_rubric_scores.json",
        "rubrics": [
            ("team_R1_composition",     "R1 Composition",     "doc-verifiable"),
            ("team_R2_credentials",     "R2 Credentials",     "doc-verifiable"),
            ("team_R3_track_record",    "R3 Track Record",    "doc-verifiable"),
            ("team_R4_governance",      "R4 Governance",      "doc-verifiable"),
            ("team_R5_capacity",        "R5 Capacity",        "judgment-dep"),
            ("team_R6_complementarity", "R6 Complementarity", "judgment-dep"),
            ("team_R7_cohesion",        "R7 Cohesion",        "judgment-dep"),
            ("team_R8_key_person_risk", "R8 Key-Person Risk", "judgment-dep"),
        ],
        "plus_targets":  [],
        "minus_targets": [],
        "plus_injection": "",
        "plus_marker":  "##### Project objectives",
        "minus_strip_bullets": [],
        "minus_inline_subs": [],
    },

    # ── Feasibility ───────────────────────────────────────────────────────────
    "feasibility": {
        "tool":       "src/tools/generate_feasibility_rubric_scores.py",
        "scores_dir": "src/data/feasibility_rubric_scores",
        "scores_key": "feasibility_rubrics",
        "result_file": "feasibility_rubric_scores.json",
        "rubrics": [
            ("feasibility_R1_budget_detail",         "R1 Budget Detail",         "doc-verifiable"),
            ("feasibility_R2_funding_secured",       "R2 Funding Secured",       "doc-verifiable"),
            ("feasibility_R3_infrastructure",        "R3 Infrastructure",        "doc-verifiable"),
            ("feasibility_R4_technical_readiness",   "R4 Technical Readiness",   "doc-verifiable"),
            ("feasibility_R5_risk_mitigation",       "R5 Risk Mitigation",       "judgment-dep"),
            ("feasibility_R6_resource_timeline_fit", "R6 Resource-Timeline Fit", "judgment-dep"),
            ("feasibility_R7_financial_realism",     "R7 Financial Realism",     "judgment-dep"),
            ("feasibility_R8_sustainability",        "R8 Sustainability",        "judgment-dep"),
        ],
        "plus_targets":  [],
        "minus_targets": [],
        "plus_injection": "",
        # Deliberately NOT the next heading after the feasibility section (the pattern
        # used for the other dimensions): that heading sits at ~11-13k chars, past the
        # scorers' 12,000-char _trim_text window, so the injection would be invisible.
        "plus_marker":  "##### Team and governance",
        "minus_strip_bullets": [],
        "minus_inline_subs": [],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  RUBRIC ANCHORS (for adaptive mode — score-5 criteria for all 8 rubrics)
# ══════════════════════════════════════════════════════════════════════════════

RUBRIC_ANCHORS: dict[str, dict[str, dict]] = {
    "strategy": {
        "strategy_R1_gtm": {
            "name": "Go-to-Market channel",
            "score_1": "no channel named",
            "score_3": "channel type identified but no named partner or contract",
            "score_5": "named contracted distribution channel or paying channel partner with revenue evidence",
        },
        "strategy_R2_milestones": {
            "name": "Milestone roadmap",
            "score_1": "no roadmap",
            "score_3": "roadmap exists but milestones are vague or undated",
            "score_5": "phased roadmap with specific dated milestones and measurable completion criteria",
        },
        "strategy_R3_revenue": {
            "name": "Revenue model",
            "score_1": "no revenue model described",
            "score_3": "revenue model named but without pricing or unit economics",
            "score_5": "explicit pricing, payment structure, and unit economics with gross margin stated",
        },
        "strategy_R4_regulatory": {
            "name": "Regulatory pathway",
            "score_1": "no regulatory requirement mentioned",
            "score_3": "regulatory requirement acknowledged but pathway not specified",
            "score_5": "filed regulatory submission or confirmed specific compliance pathway with documented engagement",
        },
        "strategy_R5_moat": {
            "name": "Competitive moat",
            "score_1": "no competitive moat described",
            "score_3": "moat claimed without evidence of exclusivity",
            "score_5": "concrete durable advantage: issued IP, exclusive channel contract, or network effect with data",
        },
        "strategy_R6_partners": {
            "name": "Partner engagement",
            "score_1": "no partner named",
            "score_3": "partner names mentioned without engagement evidence",
            "score_5": "active commercial engagement: contract signed, revenue received, or co-development agreement in place",
        },
        "strategy_R7_team_fit": {
            "name": "Team-strategy fit",
            "score_1": "team background is unrelated to the stated go-to-market strategy",
            "score_3": "team has adjacent experience",
            "score_5": "team has direct prior success executing the specific strategy described",
        },
        "strategy_R8_timing": {
            "name": "Market timing",
            "score_1": "no market timing rationale given",
            "score_3": "timing mentioned but based only on proposal claims",
            "score_5": "compelling external catalyst (regulatory change, technology inflection, validated demand spike) cited with evidence",
        },
    },
    "advantages": {
        "advantages_R1_mechanism_novelty": {
            "name": "Mechanism novelty",
            "score_1": "no novel mechanism claimed or mechanism is standard/known",
            "score_3": "novelty claimed but without mechanistic explanation",
            "score_5": "new mechanistic class described with specific technical differentiation",
        },
        "advantages_R2_ip_status": {
            "name": "IP status",
            "score_1": "no IP mentioned",
            "score_3": "IP described as planned or in preparation",
            "score_5": "specific patents filed or granted with application numbers or publication cited",
        },
        "advantages_R3_performance_proof": {
            "name": "Performance proof",
            "score_1": "performance claims made with no data",
            "score_3": "performance data referenced but without specific figures or comparators",
            "score_5": "quantitative performance proof vs benchmark with methodology described",
        },
        "advantages_R4_competitor_benchmark": {
            "name": "Competitor benchmark",
            "score_1": "no comparison to existing solutions",
            "score_3": "competitors named but comparison is qualitative only",
            "score_5": "head-to-head data against named competitor products with specific metrics",
        },
        "advantages_R5_defensibility": {
            "name": "Defensibility",
            "score_1": "no defensible moat; advantage is easily replicable",
            "score_3": "some barrier exists but it is partial or time-limited",
            "score_5": "durable moat combining IP, exclusive channel, network effect, or regulatory barrier",
        },
        "advantages_R6_platform_potential": {
            "name": "Platform potential",
            "score_1": "single-product with no platform or expansion potential described",
            "score_3": "platform potential mentioned but not substantiated",
            "score_5": "clear platform: one technology enabling multiple validated applications",
        },
        "advantages_R7_validation_signals": {
            "name": "Validation signals",
            "score_1": "no external validation of advantage claims",
            "score_3": "validation mentioned (awards, pilots) but not independently verifiable",
            "score_5": "regulatory milestone, peer-reviewed publication, or paying customer validates advantage",
        },
        "advantages_R8_adoption_readiness": {
            "name": "Adoption readiness",
            "score_1": "no evidence of market readiness or adoption pathway",
            "score_3": "adoption pathway described but speculative",
            "score_5": "active pilots, LOIs, or contracts demonstrate real adoption momentum",
        },
    },
    "objectives": {
        "objectives_R1_problem_clarity": {
            "name": "Problem clarity",
            "score_1": "problem is vague or not stated",
            "score_3": "problem described but without specific context or data",
            "score_5": "problem precisely defined with specific patient/customer context and quantified evidence",
        },
        "objectives_R2_market_size": {
            "name": "Market size",
            "score_1": "no market size stated",
            "score_3": "market size mentioned but unverified or broad estimate only",
            "score_5": "specific, sourced market size figure with addressable segment breakdown",
        },
        "objectives_R3_buyer_pathway": {
            "name": "Buyer pathway",
            "score_1": "no buyer or payer identified",
            "score_3": "buyer type named but no pathway or engagement described",
            "score_5": "named buyer/payer with documented pathway, LOI, or pilot evidence",
        },
        "objectives_R4_unmet_need_evidence": {
            "name": "Unmet need evidence",
            "score_1": "unmet need asserted with no supporting evidence",
            "score_3": "clinical/market gap described with partial references",
            "score_5": "validated unmet need with published data, regulatory evidence, or patient outcome data",
        },
        "objectives_R5_why_now": {
            "name": "Why-now timing",
            "score_1": "no timing rationale given",
            "score_3": "why-now mentioned but based only on proposal claims",
            "score_5": "strong external catalyst cited (regulation change, technology inflection, validated demand spike)",
        },
        "objectives_R6_competitive_context": {
            "name": "Competitive context",
            "score_1": "no mention of existing solutions or competitors",
            "score_3": "competitors acknowledged but differentiation is vague",
            "score_5": "clear positioning vs named alternatives with specific differentiating evidence",
        },
        "objectives_R7_problem_solution_fit": {
            "name": "Problem-solution fit",
            "score_1": "solution does not clearly address the stated problem",
            "score_3": "fit is plausible but not explicitly argued",
            "score_5": "tight and explicit problem-solution fit with mechanism explained",
        },
        "objectives_R8_addressability": {
            "name": "Addressability",
            "score_1": "target segment is undefined or too broad to serve",
            "score_3": "segment defined but access strategy is missing",
            "score_5": "specific addressable segment with defined entry point and realistic reach estimate",
        },
    },
    "team": {
        "team_R1_composition": {
            "name": "Team composition",
            "score_1": "only a single lead is named or roles are unspecified",
            "score_3": "core roles named but key technical, domain, or commercial functions are unfilled",
            "score_5": "all critical roles filled by named individuals with stated responsibilities",
        },
        "team_R2_credentials": {
            "name": "Domain credentials",
            "score_1": "no relevant qualifications stated",
            "score_3": "credentials stated only generically (degrees, titles)",
            "score_5": "specific credentials in the project's domain: named institutions, publications, patents, or prior positions",
        },
        "team_R3_track_record": {
            "name": "Prior execution track record",
            "score_1": "no prior projects or ventures cited",
            "score_3": "prior roles or projects mentioned without outcomes",
            "score_5": "named prior ventures or projects with documented outcomes (launch, approval, exit, funded grant)",
        },
        "team_R4_governance": {
            "name": "Governance and advisors",
            "score_1": "no governance structure, board, or advisors described",
            "score_3": "advisors or board mentioned without names or roles",
            "score_5": "named board or advisors, defined decision-making structure, and formal agreements (collaboration, equity)",
        },
        "team_R5_capacity": {
            "name": "Execution capacity",
            "score_1": "team size or time commitment clearly insufficient for the project scope",
            "score_3": "capacity plausible but commitment (full-time vs part-time) is unclear",
            "score_5": "dedicated team sized to the project scope, with a hiring plan for gaps",
        },
        "team_R6_complementarity": {
            "name": "Skill complementarity",
            "score_1": "skills concentrated in one area with a critical gap",
            "score_3": "skills mostly complementary with one notable gap",
            "score_5": "balanced technical, domain, and commercial skills with no critical gap",
        },
        "team_R7_cohesion": {
            "name": "Team cohesion",
            "score_1": "no evidence the members have worked together",
            "score_3": "members linked by a shared institution but no joint delivery",
            "score_5": "core members have jointly delivered prior projects with documented outcomes",
        },
        "team_R8_key_person_risk": {
            "name": "Key-person risk",
            "score_1": "project depends on one individual with no backup",
            "score_3": "some redundancy, but key functions rest on one person",
            "score_5": "responsibilities distributed, with backup and retention mechanisms",
        },
    },
    "feasibility": {
        "feasibility_R1_budget_detail": {
            "name": "Budget detail",
            "score_1": "no budget described",
            "score_3": "total budget stated without a breakdown",
            "score_5": "itemised budget by category, with amounts and time period",
        },
        "feasibility_R2_funding_secured": {
            "name": "Funding secured",
            "score_1": "no funding source described",
            "score_3": "funding sources named but not committed",
            "score_5": "committed funding documented (grant awarded, investment closed) with amounts",
        },
        "feasibility_R3_infrastructure": {
            "name": "Infrastructure and resource access",
            "score_1": "no facilities, equipment, or data access described",
            "score_3": "resources mentioned without confirmed access",
            "score_5": "named facilities, equipment, data, or suppliers with confirmed access (ownership or agreement)",
        },
        "feasibility_R4_technical_readiness": {
            "name": "Technical readiness",
            "score_1": "concept only, no prototype or data",
            "score_3": "prototype or preliminary data described without figures",
            "score_5": "working prototype or validated pilot with stated maturity level and results",
        },
        "feasibility_R5_risk_mitigation": {
            "name": "Risk mitigation",
            "score_1": "no risks identified",
            "score_3": "risks listed without mitigation",
            "score_5": "key risks identified with specific mitigation and contingency plans",
        },
        "feasibility_R6_resource_timeline_fit": {
            "name": "Resource-timeline fit",
            "score_1": "timeline clearly unrealistic for the available resources",
            "score_3": "timeline plausible but not argued",
            "score_5": "timeline explicitly justified by resources, staffing, and dependencies",
        },
        "feasibility_R7_financial_realism": {
            "name": "Financial realism",
            "score_1": "budget clearly insufficient for the stated scope",
            "score_3": "budget plausible but no comparables given",
            "score_5": "budget benchmarked against comparables, with runway to the next milestone",
        },
        "feasibility_R8_sustainability": {
            "name": "Operational sustainability",
            "score_1": "no plan beyond the funded period",
            "score_3": "sustainability asserted without a mechanism",
            "score_5": "concrete mechanism to sustain operations (revenue, follow-on funding, institutional commitment) with evidence",
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  HARDCODED VARIANT CREATION (default mode)
# ══════════════════════════════════════════════════════════════════════════════

def _inject_plus(text: str, cfg: dict) -> str:
    marker = cfg["plus_marker"]
    injection = cfg["plus_injection"]
    if marker in text:
        return text.replace(marker, injection.rstrip() + "\n\n" + marker, 1)
    qa = "## 2. Dimension Q&A"
    if qa in text:
        return text.replace(qa, injection.rstrip() + "\n\n" + qa, 1)
    return text + injection


def _strip_minus(text: str, cfg: dict) -> str:
    strip_keywords = cfg["minus_strip_bullets"]
    inline_subs    = cfg["minus_inline_subs"]

    lines = text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        is_list = bool(re.match(r"^-\s|^\d+\.\s", stripped))
        lower   = line.lower()

        if is_list and any(kw.lower() in lower for kw in strip_keywords):
            continue

        for pattern, repl in inline_subs:
            line = re.sub(pattern, repl, line)

        filtered.append(line)
    return "\n".join(filtered)


def create_variants(short_pid: str, dim: str, cfg: dict) -> tuple[Path, Path]:
    full_pid   = short_pid + PID_SUFFIX
    plus_tag   = f"{short_pid}_{dim}plus{PID_SUFFIX}"
    minus_tag  = f"{short_pid}_{dim}minus{PID_SUFFIX}"
    src        = REPORT_DIR / f"{full_pid}_final_report.md"
    plus_path  = REPORT_DIR / f"{plus_tag}_final_report.md"
    minus_path = REPORT_DIR / f"{minus_tag}_final_report.md"

    text = src.read_text(encoding="utf-8")
    plus_path.write_text(_inject_plus(text, cfg), encoding="utf-8")
    minus_path.write_text(_strip_minus(text, cfg), encoding="utf-8")
    return plus_path, minus_path


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE CONTENT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _trim(text: str, max_chars: int = 12000) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def generate_adaptive_plus(pid: str, dim: str, report_text: str) -> dict | None:
    """LLM generates domain-specific injection text covering all 8 rubrics."""
    anchors = RUBRIC_ANCHORS[dim]
    rubric_descriptions = "\n".join(
        f"- {key} ({info['name']}): score_5 means {info['score_5']}"
        for key, info in anchors.items()
    )
    system_prompt = (
        "You are designing a content-sensitivity experiment for investment proposal scoring. "
        "Task: write injection text that, when added to this proposal report, would cause "
        "an LLM scorer to assign score 4 or 5 on ALL 8 rubrics listed.\n\n"
        "Requirements:\n"
        "1. Read the proposal carefully — understand its domain, technology, and market.\n"
        "2. Write 4-6 paragraphs of realistic, specific, domain-appropriate evidence.\n"
        "3. Cover ALL 8 rubrics — every rubric must have at least one sentence addressing it.\n"
        "4. Use specific numbers, named organisations, realistic dates and references.\n"
        "5. Match the report writing style and domain vocabulary.\n"
        "6. Do NOT repeat or contradict existing content — add NEW evidence only.\n"
        "7. Text will be inserted verbatim, so it must read naturally in context.\n\n"
        f"The 8 rubrics and their score-5 criteria:\n{rubric_descriptions}\n\n"
        "Return ONLY valid JSON with exactly these two keys:\n"
        '{"injection_text": "<the paragraphs to inject>", '
        '"rubric_coverage": {"<rubric_key>": "<which sentence covers this rubric>", ...}}'
    )
    try:
        result = chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(
                    {"proposal_id": pid, "report_excerpt": _trim(report_text)},
                    ensure_ascii=False,
                )},
            ],
            provider=PROVIDER,
            temperature=0.0,
            max_tokens=2500,
        )
        if "injection_text" not in result:
            print(f"  [WARN] generate_adaptive_plus: missing injection_text for {pid}/{dim}")
            return None
        return result
    except Exception as e:
        print(f"  [WARN] generate_adaptive_plus failed for {pid}/{dim}: {e}")
        return None


def generate_adaptive_minus(pid: str, dim: str, report_text: str) -> dict | None:
    """LLM identifies exact passages to strip/replace for all 8 rubrics."""
    anchors = RUBRIC_ANCHORS[dim]
    rubric_descriptions = "\n".join(
        f"- {key} ({info['name']}): score_5={info['score_5']}; score_1={info['score_1']}"
        for key, info in anchors.items()
    )
    system_prompt = (
        "You are designing a content-sensitivity experiment for investment proposal scoring. "
        "Task: identify the specific text a scorer would use to justify a HIGH score on each "
        "rubric, and provide replacements that make the evidence explicitly absent.\n\n"
        "Requirements:\n"
        "1. For each of the 8 rubrics, find 1-3 sentences or phrases that are the "
        "strongest evidence for a high score on that rubric.\n"
        "2. The 'find' value MUST be an EXACT verbatim copy from the report — no paraphrase.\n"
        "3. The 'replace' value must EXPLICITLY state absence "
        "(e.g. 'No specific regulatory pathway is described.' not empty string).\n"
        "4. Cover ALL 8 rubrics — if a rubric has weak evidence, find the closest claim.\n"
        "5. Replacements must keep the surrounding text grammatically intact.\n\n"
        f"The 8 rubrics:\n{rubric_descriptions}\n\n"
        "Return ONLY valid JSON:\n"
        '{"replacements": [{"rubric": "<rubric_key>", "find": "<exact verbatim text>", '
        '"replace": "<explicit absence statement>"}, ...]}'
    )
    try:
        result = chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(
                    {"proposal_id": pid, "report_excerpt": _trim(report_text)},
                    ensure_ascii=False,
                )},
            ],
            provider=PROVIDER,
            temperature=0.0,
            max_tokens=2500,
        )
        if "replacements" not in result:
            print(f"  [WARN] generate_adaptive_minus: missing replacements for {pid}/{dim}")
            return None
        return result
    except Exception as e:
        print(f"  [WARN] generate_adaptive_minus failed for {pid}/{dim}: {e}")
        return None


def save_adaptive_content(pid: str, dim: str, plus_result: dict, minus_result: dict) -> None:
    ADAPTIVE_CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    path = ADAPTIVE_CONTENT_DIR / f"{pid}_{dim}.json"
    path.write_text(json.dumps({
        "pid": pid,
        "dim": dim,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plus":  plus_result,
        "minus": minus_result,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] Adaptive content saved → {path.name}")


def load_adaptive_content(pid: str, dim: str) -> dict | None:
    path = ADAPTIVE_CONTENT_DIR / f"{pid}_{dim}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def adaptive_inject_plus(text: str, injection_text: str, marker: str) -> str:
    if marker and marker in text:
        return text.replace(marker, injection_text.rstrip() + "\n\n" + marker, 1)
    qa = "## 2. Dimension Q&A"
    if qa in text:
        return text.replace(qa, injection_text.rstrip() + "\n\n" + qa, 1)
    return text + "\n\n" + injection_text


def adaptive_strip_minus(text: str, replacements: list) -> str:
    applied = 0
    for item in replacements:
        find    = (item.get("find") or "").strip()
        replace = item.get("replace") or ""
        if not find:
            continue
        if find in text:
            text = text.replace(find, replace, 1)
            applied += 1
        else:
            # Fallback: try normalising internal whitespace
            pattern = re.sub(r"\s+", r"\\s+", re.escape(" ".join(find.split())))
            m = re.search(pattern, text)
            if m:
                text = text[:m.start()] + replace + text[m.end():]
                applied += 1
            else:
                print(f"  [WARN] strip_minus: could not find: {find[:70]!r}")
    print(f"  [INFO] Applied {applied}/{len(replacements)} minus replacements")
    return text


def create_adaptive_variants(
    short_pid: str, dim: str, cfg: dict, skip_generate: bool
) -> tuple[Path, Path]:
    full_pid   = short_pid + PID_SUFFIX
    plus_tag   = f"{short_pid}_{dim}plus{PID_SUFFIX}"
    minus_tag  = f"{short_pid}_{dim}minus{PID_SUFFIX}"
    src        = REPORT_DIR / f"{full_pid}_final_report.md"
    plus_path  = REPORT_DIR / f"{plus_tag}_final_report.md"
    minus_path = REPORT_DIR / f"{minus_tag}_final_report.md"

    text = src.read_text(encoding="utf-8")

    content = load_adaptive_content(short_pid, dim)

    if content is None or not skip_generate:
        print(f"  [..] Generating adaptive PLUS  for {short_pid}/{dim} …")
        plus_result  = generate_adaptive_plus(short_pid, dim, text)
        print(f"  [..] Generating adaptive MINUS for {short_pid}/{dim} …")
        minus_result = generate_adaptive_minus(short_pid, dim, text)

        if plus_result and minus_result:
            save_adaptive_content(short_pid, dim, plus_result, minus_result)
            content = {"plus": plus_result, "minus": minus_result}
        elif not cfg.get("plus_injection"):
            raise RuntimeError(
                f"Adaptive generation failed for {short_pid}/{dim} and this dimension "
                f"has no hardcoded fallback content."
            )
        else:
            print(f"  [WARN] Adaptive generation failed for {short_pid}/{dim} — using hardcoded fallback.")
            plus_path.write_text(_inject_plus(text, cfg),  encoding="utf-8")
            minus_path.write_text(_strip_minus(text, cfg), encoding="utf-8")
            return plus_path, minus_path

    injection_text = content["plus"].get("injection_text", "")
    replacements   = content["minus"].get("replacements", [])

    plus_path.write_text(
        adaptive_inject_plus(text, injection_text, cfg.get("plus_marker", "")),
        encoding="utf-8",
    )
    minus_path.write_text(
        adaptive_strip_minus(text, replacements),
        encoding="utf-8",
    )
    return plus_path, minus_path


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_variant(pid: str, tool_path: str) -> bool:
    cmd = [sys.executable, str(ROOT / tool_path), "--pid", pid, "--use_llm"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] Scoring failed for {pid}: {result.stderr.strip()}")
        return False
    return True


def load_scores(full_pid: str, cfg: dict) -> dict | None:
    path = ROOT / cfg["scores_dir"] / full_pid / cfg["result_file"]
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["score"] for k, v in data.get(cfg["scores_key"], {}).items()}


# ══════════════════════════════════════════════════════════════════════════════
#  PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def arrow(new: int, old: int) -> str:
    if new > old: return f"+{new-old}↑"
    if new < old: return f"{new-old}↓"
    return "→"


def print_results(dim: str, cfg: dict, results: dict, pids: list):
    rubrics       = cfg["rubrics"]
    plus_targets  = cfg["plus_targets"]
    minus_targets = cfg["minus_targets"]
    SEP  = "─" * 75
    SEP2 = "═" * 75

    print()
    print(SEP2)
    print(f"  DIMENSION: {dim.upper()}")
    print(f"  PLUS  = inject specific evidence (adding content)")
    print(f"  MINUS = strip key language (removing content)")
    print(SEP2)
    print(f"  {'Proposal':<10} {'Rubric':<26} {'Baseline':>10} {'PLUS':>8} {'MINUS':>8}")
    print(SEP)

    for short_pid in pids:
        if short_pid not in results:
            continue
        b = results[short_pid]["baseline"] or {}
        p = results[short_pid]["plus"]     or {}
        m = results[short_pid]["minus"]    or {}
        for key, label, _ in rubrics:
            bv = b.get(key, "-")
            pv = p.get(key, "-")
            mv = m.get(key, "-")
            pa = arrow(pv, bv) if isinstance(pv, int) and isinstance(bv, int) else ""
            ma = arrow(mv, bv) if isinstance(mv, int) and isinstance(bv, int) else ""
            p_str = f"{pv}({pa})" if pa else str(pv)
            m_str = f"{mv}({ma})" if ma else str(mv)
            print(f"  {short_pid:<10} {label:<26} {str(bv):>10} {p_str:>8} {m_str:>8}")
        print(SEP)

    print()
    print(f"  HYPOTHESIS CHECK")
    print(f"  {'PID':<6}  " + "  ".join(f"{'PLUS:'+k.split('_R')[1][:4]:>12}" for k in plus_targets) +
          "  " + "  ".join(f"{'MINUS:'+k.split('_R')[1][:4]:>12}" for k in minus_targets) + "  Passed")
    print(f"  {'─'*6}  " + "  ".join(["─"*12] * (len(plus_targets) + len(minus_targets))) + "  ───────")

    total_pass   = 0
    total_checks = (len(plus_targets) + len(minus_targets)) * len([p for p in pids if p in results])
    for short_pid in pids:
        if short_pid not in results:
            continue
        b = results[short_pid]["baseline"] or {}
        p = results[short_pid]["plus"]     or {}
        m = results[short_pid]["minus"]    or {}
        row = f"  {short_pid:<6}  "
        passed = 0
        checks = 0
        for key in plus_targets:
            ok = isinstance(p.get(key), int) and isinstance(b.get(key), int) and p[key] > b[key]
            row += f"{'✓ PASS':>12}  " if ok else f"{'✗ FAIL':>12}  "
            passed += ok; checks += 1
        for key in minus_targets:
            ok = isinstance(m.get(key), int) and isinstance(b.get(key), int) and m[key] < b[key]
            row += f"{'✓ PASS':>12}  " if ok else f"{'✗ FAIL':>12}  "
            passed += ok; checks += 1
        total_pass += passed
        row += f"{passed}/{checks}"
        print(row)

    print(f"  {'─'*6}")
    pct = 100 * total_pass // total_checks if total_checks else 0
    print(f"  Overall: {total_pass}/{total_checks} checks passed ({pct}%)")
    print(SEP2)
    return total_pass, total_checks


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids",          nargs="+", default=ALL_PIDS)
    ap.add_argument("--dimensions",    nargs="+", default=list(DIMENSIONS.keys()),
                    choices=list(DIMENSIONS.keys()))
    ap.add_argument("--skip_score",    action="store_true",
                    help="Skip LLM scoring — reload cached scores and reprint.")
    ap.add_argument("--adaptive",      action="store_true",
                    help="Use LLM-generated per-proposal content (all 8 rubrics targeted).")
    ap.add_argument("--skip_generate", action="store_true",
                    help="With --adaptive: reuse cached adaptive content, skip generation.")
    args = ap.parse_args()

    all_results = {}

    for dim in args.dimensions:
        cfg = DIMENSIONS[dim]
        print(f"\n{'='*60}")
        print(f"  Running: {dim.upper()} dimension")
        if args.adaptive:
            all_rubric_keys = [r[0] for r in cfg["rubrics"]]
            effective_cfg = {**cfg, "plus_targets": all_rubric_keys, "minus_targets": all_rubric_keys}
            print(f"  Mode: ADAPTIVE (all 8 rubrics targeted for PLUS and MINUS)")
        else:
            effective_cfg = cfg
            print(f"  Mode: HARDCODED")
        print(f"{'='*60}")

        dim_results = {}

        for short_pid in args.pids:
            full_pid  = short_pid + PID_SUFFIX
            plus_pid  = f"{short_pid}_{dim}plus{PID_SUFFIX}"
            minus_pid = f"{short_pid}_{dim}minus{PID_SUFFIX}"

            print(f"[{short_pid}] Creating {dim} PLUS / MINUS variants …")
            if args.adaptive:
                create_adaptive_variants(short_pid, dim, cfg, args.skip_generate)
            else:
                create_variants(short_pid, dim, cfg)

            if not args.skip_score:
                print(f"[{short_pid}] Scoring PLUS …")
                score_variant(plus_pid, cfg["tool"])
                print(f"[{short_pid}] Scoring MINUS …")
                score_variant(minus_pid, cfg["tool"])

            baseline = load_scores(full_pid,  cfg)
            plus_s   = load_scores(plus_pid,  cfg)
            minus_s  = load_scores(minus_pid, cfg)

            if not baseline:
                print(f"  [WARN] No baseline for {short_pid}, skipping.")
                continue

            dim_results[short_pid] = {
                "baseline": baseline,
                "plus":     plus_s,
                "minus":    minus_s,
            }

        all_results[dim] = dim_results

        out = {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "dimension":     dim,
            "mode":          "adaptive" if args.adaptive else "hardcoded",
            "pids":          args.pids,
            "plus_targets":  effective_cfg["plus_targets"],
            "minus_targets": effective_cfg["minus_targets"],
            "results":       dim_results,
        }
        out_path = RESULTS_DIR / f"sensitivity_{dim}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n[OK] {dim} results stored → {out_path}")

        print_results(dim, effective_cfg, dim_results, args.pids)

    # Merge into an existing combined file of the same mode, so running a subset of
    # dimensions does not discard results for the others.
    mode = "adaptive" if args.adaptive else "hardcoded"
    combined_path = RESULTS_DIR / "sensitivity_all_dimensions.json"
    dimensions = list(args.dimensions)
    if combined_path.exists():
        previous = json.loads(combined_path.read_text(encoding="utf-8"))
        if previous.get("mode") == mode:
            all_results = {**previous.get("results", {}), **all_results}
            dimensions = list(dict.fromkeys(previous.get("dimensions", []) + dimensions))
    combined_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode":         mode,
        "dimensions":   dimensions,
        "pids":         args.pids,
        "results":      all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n[OK] Combined results → {combined_path}")


if __name__ == "__main__":
    main()
