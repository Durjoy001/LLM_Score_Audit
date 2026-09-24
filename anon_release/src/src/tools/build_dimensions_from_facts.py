# -*- coding: utf-8 -*-
"""
Stage 2 - Dimension builder (build_dimensions_from_facts.py)
-----------------------------------------------------------
Input:
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  (Stage 1 output)

Outputs:
  - src/data/extracted/<proposal_id>/dimensions_v2.json
  - src/data/extracted/<proposal_id>/dimension_facts.json
  - src/data/parsed/parsed_dimensions.clean.llm.json

Main responsibilities:
  - Bucket facts by dimension into team/objectives/strategy/innovation/feasibility.
  - Call the LLM once per dimension and generate, only from that dimension's facts:
      summary / key_points / risks / mitigations
  - Avoid inventing facts. Every output item should be grounded in the fact list.
  - Cover the different fact types seen in that dimension when possible.
"""

import os
import json
import re
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.llm_provider import UnifiedChatClient, active_provider, default_model

load_dotenv()

# Expected path: <project_root>/src/tools/build_dimensions_from_facts.py
BASE_DIR = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"
PARSED_DIR = BASE_DIR / "src" / "data" / "parsed"   # Kept aligned with llm_answering.
PROGRESS_FILE = BASE_DIR / "src" / "data" / "step_progress.json"

PROVIDER = active_provider()
OPENAI_MODEL = default_model(PROVIDER)
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
STAGE2_PROMPT_FACT_CHAR_LIMIT = int(os.getenv("STAGE2_PROMPT_FACT_CHAR_LIMIT", "10000"))
STAGE2_LLM_JSON_RETRIES = int(os.getenv("STAGE2_LLM_JSON_RETRIES", "2"))


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass

DIMENSION_NAMES = ["team", "objectives", "strategy", "innovation", "feasibility"]

# Reuse one global configured-provider client.
client = UnifiedChatClient(provider=PROVIDER, model=OPENAI_MODEL)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _normalize_fact_text(text: str) -> str:
    return re.sub(r"\W+", " ", _normalize_text(text).lower()).strip()


def _fact_identity_key(fact: Dict[str, Any]) -> tuple:
    text = _normalize_fact_text(str(fact.get("text", "") or ""))
    fact_type = str(fact.get("type", "other") or "other")
    dims = fact.get("dimensions", [])
    if not isinstance(dims, list):
        dims = []
    dims_key = tuple(sorted(d for d in dims if d in DIMENSION_NAMES))
    return text, fact_type, dims_key


def _dedupe_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for fact in facts:
        key = _fact_identity_key(fact)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _dedupe_strings(items: List[Any]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        text = _normalize_text(str(item))
        if not text:
            continue
        key = _normalize_fact_text(text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _fact_prompt_priority(fact: Dict[str, Any]) -> int:
    t = fact.get("type", "other") or "other"
    txt = fact.get("text", "") or ""
    if t == "risk" or _looks_like_risk(txt):
        return 0
    if t == "mitigation" or _looks_like_mitigation(txt):
        return 1
    return 2


def _select_facts_for_prompt(facts: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    """
    Keep the prompt within context limits while preferring risk and mitigation facts first.
    """
    ordered = _dedupe_facts(facts)
    ordered = sorted(enumerate(ordered), key=lambda item: (_fact_prompt_priority(item[1]), item[0]))

    kept: List[Dict[str, Any]] = []
    total = 0
    for _, fact in ordered:
        text = str(fact.get("text", "") or "")
        text_len = len(text)
        if total + text_len > max_chars and kept:
            break
        kept.append(fact)
        total += text_len
    return kept

# Use {dimension_name} as the only placeholder. Other braces are literal JSON examples.
# Later code uses .replace("{dimension_name}", xxx), not .format().
DIMENSION_PROMPT_TEMPLATE = """
You are a rigorous project-review assistant. Based only on the extracted fact list,
generate a structured summary for one evaluation dimension.

Hard constraints:
1. Use only the information provided in the fact list. Do not invent organizations,
   people, diseases, drugs, technologies, models, pipeline names, market numbers,
   patient counts, CAGR values, or any other figures.
2. Do not use speculative phrases such as "based on common knowledge",
   "industry practice", "generally", "usually", "likely", or "probably" unless
   that wording is itself present in the original facts.
3. You may merge, deduplicate, and compress facts, but every key_point, risk, and
   mitigation must be traceable to at least one fact text.
4. Summarize only the current dimension. You may mention strongly related facts,
   but do not drift into generic evaluation of other dimensions.

Current dimension: {dimension_name}

Representative focus examples for this dimension:
- team:
  - Core team members, organizations, titles, research areas, and project roles.
  - Organization structure, cross-border collaboration, and division of work.
  - Composition of partner teams from other institutions.
- objectives:
  - Overall goals, such as what the project aims to achieve in 3-5 years.
  - Stage milestones such as 0-6, 6-12, 12-18, or 18-36 months.
  - Goals for each pipeline or subproject.
- strategy:
  - Technical route, including AI, algorithmic, or experimental pathways.
  - Partnership strategy, business model, and market-entry plan.
  - Operating model, domestic/international coordination, and regulatory path.
- innovation:
  - Innovation in technology, product, or operating model.
  - Advantages over existing approaches.
  - Key patents, distinctive data/resources, and validation evidence.
- feasibility:
  - Resource base, such as labs, platforms, and partners.
  - Budget, funding plan, and funding sources.
  - Risk matrix and mitigation measures across technical, market, funding,
    regulatory, AI-model, or other risks.
  - Timeline, implementation path, enrollment, adherence, and other feasibility factors.

Input JSON structure (payload):

payload = {
  "all_facts": [
    {
      "text": "One natural-language fact",
      "dimensions": ["team", "strategy"],
      "type": "team_member, pipeline, etc.",
      "meta": { ... }
    },
    ...
  ],
  "risk_facts": [
    // Facts with type = "risk" or text that clearly includes risk/challenge wording.
  ],
  "mitigation_facts": [
    // Facts with type = "mitigation" or text that clearly includes mitigation/solution wording.
  ]
}

Your tasks:

1. Classify, deduplicate, and merge all_facts mentally. Identify the most important
   and representative points for the current dimension.
   - Cover different fact types seen in this dimension when possible.
2. Output exactly one JSON object in this format:

{
  "summary": "string, 2-4 English sentences summarizing this dimension based only on facts",
  "key_points": [
    "string, specific point 1 traceable to at least one fact",
    "string, specific point 2",
    "... if enough facts exist, output 6-10 non-duplicative key points."
  ],
  "risks": [
    "string, risk summarized from risk_facts",
    "If risk_facts is empty, say that the proposal provides limited or no detailed risk information for this dimension.",
    "Do not invent risks that the proposal did not state.",
    "..."
  ],
  "mitigations": [
    "string, mitigation for a risk, based on mitigation_facts or explicit strategies/mechanisms in all_facts",
    "If mitigation_facts is empty and all_facts has no clear mitigation, say that the proposal does not specify mitigation measures for this dimension.",
    "Do not invent solutions.",
    "..."
  ]
}

Key point count requirements:
- If all_facts has 20 or more facts, try to output 6-10 non-duplicative key_points.
- If all_facts has 8-19 facts, output 5-8 key_points when possible.
- If all_facts has 7 or fewer facts, output 3-5 key_points as supported by the facts.
- In every case, cover different types of information instead of repeating the same fact type.

Coverage requirements by dimension:
- team: if team_member, org_structure, and collaboration facts exist, cover each type.
- objectives: cover both overall goals and staged milestones if both are present.
- strategy: cover technical route, market/customer information, partnership/business model,
  funding source, and regulatory path when present.
- innovation: prioritize innovation points such as ip_asset, ai_model, or tech_route,
  plus available evidence.
- feasibility: include resources/capabilities, budget/funding, risks/mitigations, and
  timeline or implementation difficulty when present.

Risk and mitigation constraints:
- Generate "risks" primarily from risk_facts.
- Generate "mitigations" primarily from mitigation_facts, and use all_facts only when
  it explicitly mentions a response strategy.
- Do not assume risks or mitigations that are not present in the text.

Additional requirements:
- Only describe what the proposal text shows.
- Output no text outside the JSON object. Do not explain or add comments.
"""


def load_raw_facts(proposal_id: str) -> List[Dict[str, Any]]:
    path = EXTRACTED_DIR / proposal_id / "raw_facts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"raw_facts.jsonl was not found. Run extract_facts_by_chunk.py first: {path}")

    facts: List[Dict[str, Any]] = []
    malformed_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if isinstance(obj, dict):
                facts.append(obj)
    print(f"[INFO] Loaded {len(facts)} facts from {path}")
    if malformed_lines:
        print(f"[WARN] Skipped {malformed_lines} malformed raw fact line(s) while loading {path}")
    return facts


def group_facts_by_dimension(facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Bucket facts into the five dimensions.
    A single fact can belong to multiple dimensions and appear in multiple buckets.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {dim: [] for dim in DIMENSION_NAMES}

    for fact in facts:
        dims = fact.get("dimensions", [])
        if not isinstance(dims, list):
            continue
        for dim in dims:
            if dim in grouped:
                grouped[dim].append(fact)

    for dim in DIMENSION_NAMES:
        print(f"[INFO] Dimension {dim} related facts: {len(grouped[dim])}")
    return grouped


def sort_facts_for_dimension(dimension_name: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort facts by dimension-specific type priority so important information is seen first.
    This is a general rule and does not depend on proposal-specific content.
    """
    priority_map = {
        "team": [
            # Specific team profiles first, then structure and collaboration mode.
            "team_member", "org_structure", "collaboration", "resource", "other"
        ],
        "objectives": [
            # Milestones and pipeline goals are central, with market-related goals included.
            "milestone", "pipeline", "clinical_design", "product", "market", "other"
        ],
        "strategy": [
            # Put market first so market analysis is always visible to the LLM.
            "market", "tech_route", "product", "collaboration",
            "funding_source", "regulatory", "other"
        ],
        "innovation": [
            "ip_asset", "evidence", "ai_model", "tech_route", "product", "other"
        ],
        "feasibility": [
            "resource", "budget_item", "funding_source",
            "risk", "mitigation", "regulatory", "other"
        ],
    }
    order = priority_map.get(dimension_name, ["other"])

    def type_rank(t: str) -> int:
        return order.index(t) if t in order else len(order)

    return sorted(
        facts,
        key=lambda f: type_rank(f.get("type", "other"))
    )


def truncate_facts_for_prompt(facts: List[Dict[str, Any]], max_chars: int = 10000) -> List[Dict[str, Any]]:
    """
    Keep the prompt within context limits by truncating facts by text length.
    Facts are accumulated in order until max_chars is reached. Metadata is preserved.
    """
    kept: List[Dict[str, Any]] = []
    total = 0
    for fact in facts:
        t = fact.get("text", "") or ""
        t_len = len(t)
        if total + t_len > max_chars and kept:
            break
        kept.append(fact)
        total += t_len
    return kept


# ===== Helpers: fallback risk / mitigation detection from text =====

_RISK_EN = ["risk", "risks", "challenge", "challenges", "bottleneck", "bottlenecks",
            "uncertainty", "limitation", "limitations", "weakness", "weaknesses",
            "barrier", "barriers", "issue", "issues", "difficulty", "difficulties"]

_MITIG_EN = ["mitigation", "mitigate", "mitigating", "address", "addresses", "addressing",
             "solve", "solves", "solving", "overcome", "overcoming",
             "reduce", "reduces", "reducing", "decrease", "decreases", "decreasing",
             "improve", "improves", "improving", "optimize", "optimizing", "optimization"]


def _looks_like_risk(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in t for k in _RISK_EN):
        return True
    return False

def reclassify_risk_mitigation_global(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Correct risk/mitigation labels globally before building dimensions:
    - Promote a non-risk fact to risk if the text clearly describes risk/challenge.
    - Promote a non-mitigation fact to mitigation if the text clearly describes a response.
    - Demote a risk/mitigation fact to other if the text no longer matches the label,
      switching between risk and mitigation when the opposite label is clearer.
    """
    new_facts: List[Dict[str, Any]] = []

    for f in facts:
        t = f.get("type", "other") or "other"
        txt = f.get("text", "") or ""

        # Correct existing risk/mitigation labels first.
        if t == "risk":
            if not _looks_like_risk(txt):
                # Switch to mitigation if the text looks like a response; otherwise demote.
                if _looks_like_mitigation(txt):
                    t = "mitigation"
                else:
                    t = "other"
        elif t == "mitigation":
            if not _looks_like_mitigation(txt):
                # Switch to risk if the text looks like a risk; otherwise demote.
                if _looks_like_risk(txt):
                    t = "risk"
                else:
                    t = "other"
        else:
            # Promote neutral labels when the text clearly indicates risk or mitigation.
            if _looks_like_risk(txt):
                t = "risk"
            elif _looks_like_mitigation(txt):
                t = "mitigation"

        f["type"] = t
        new_facts.append(f)

    return new_facts

def _looks_like_mitigation(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in t for k in _MITIG_EN):
        return True
    return False


def _normalize_for_overlap(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def complete_key_points_from_facts(
    dimension_name: str,
    key_points: List[Any],
    sorted_facts: List[Dict[str, Any]],
    target_min: int,
) -> List[str]:
    """
    Deterministically add grounded key points when the LLM compresses a rich fact
    bucket too aggressively. This keeps Stage 3 question generation anchored to
    more of the evidence without adding another model call.
    """
    completed = [str(x).strip() for x in key_points if str(x).strip()]
    if len(completed) >= target_min:
        return completed

    seen = {_normalize_for_overlap(x)[:140] for x in completed}
    existing_blob = " ".join(seen)

    for fact in sorted_facts:
        raw = str(fact.get("text") or "").strip()
        if len(raw) < 24:
            continue
        norm = _normalize_for_overlap(raw)
        if not norm or norm[:140] in seen:
            continue
        # Avoid adding near-identical facts already summarized by the model.
        fact_terms = {w for w in norm.split() if len(w) >= 5}
        if fact_terms and existing_blob:
            overlap = sum(1 for w in fact_terms if w in existing_blob)
            if overlap / max(len(fact_terms), 1) >= 0.65:
                continue
        text = raw
        if len(text) > 260:
            text = text[:257].rstrip() + "..."
        completed.append(text)
        seen.add(norm[:140])
        existing_blob += " " + norm
        if len(completed) >= target_min:
            break

    if len(completed) > len(key_points):
        print(
            f"[INFO] Dimension {dimension_name}: completed key_points "
            f"from {len(key_points)} to {len(completed)} using grounded facts."
        )
    return completed


def call_llm_for_dimension(dimension_name: str, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Call the LLM once for a single dimension and generate summary/key_points/risks/mitigations.
    Hard constraint: use only the provided facts.
    """

    # 1) Sort by type, then truncate, so higher-priority facts are seen first.
    sorted_facts = sort_facts_for_dimension(dimension_name, facts)
    all_facts_for_prompt = _select_facts_for_prompt(sorted_facts, max_chars=STAGE2_PROMPT_FACT_CHAR_LIMIT)

    # 2) Identify risk / mitigation facts by both type and text keywords.
    risk_facts: List[Dict[str, Any]] = []
    mitigation_facts: List[Dict[str, Any]] = []

    for f in all_facts_for_prompt:
        t = f.get("type", "")
        txt = f.get("text", "") or ""

        # Include explicit risk labels and text that looks like risk/challenge content.
        if t == "risk" or _looks_like_risk(txt):
            risk_facts.append(f)

        # Include explicit mitigation labels and text that looks like response/solution content.
        if t == "mitigation" or _looks_like_mitigation(txt):
            mitigation_facts.append(f)

    risk_facts = _dedupe_facts(risk_facts)
    mitigation_facts = _dedupe_facts(mitigation_facts)

    payload = {
        "all_facts": all_facts_for_prompt,
        "risk_facts": risk_facts,
        "mitigation_facts": mitigation_facts,
    }
    facts_json_str = json.dumps(payload, ensure_ascii=False, indent=2)

    # Use replace, not format, so literal JSON braces are not treated as placeholders.
    prompt = DIMENSION_PROMPT_TEMPLATE.replace("{dimension_name}", dimension_name)

    messages = [
        {
            "role": "system",
            "content": "You are a rigorous project-review assistant. Summarize only from the provided facts and do not invent details.",
        },
        {
            "role": "user",
            "content": prompt + "\n\n=== payload starts ===\n" + facts_json_str,
        },
    ]

    raw = ""
    data: Dict[str, Any] = {}
    for attempt in range(1, max(1, STAGE2_LLM_JSON_RETRIES) + 1):
        if attempt == 1:
            attempt_messages = messages
        else:
            retry_prompt = (
                prompt
                + "\n\nImportant: the previous response was not valid JSON. "
                "Return only one JSON object with the exact keys summary, key_points, risks, and mitigations."
                + "\n\n=== payload starts ===\n"
                + facts_json_str
            )
            attempt_messages = [
                messages[0],
                {
                    "role": "user",
                    "content": retry_prompt,
                },
            ]

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=attempt_messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2600,
        )
        raw = resp.choices[0].message.content

        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            print(f"[WARN] Dimension {dimension_name} JSON parse failed on attempt {attempt}.")
            if attempt >= max(1, STAGE2_LLM_JSON_RETRIES):
                print(f"[WARN] Dimension {dimension_name} raw content preview follows:")
                print(raw)
                raise e

    # Ensure the four expected fields exist.
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = ""
    key_points = data.get("key_points", [])
    if not isinstance(key_points, list):
        key_points = []
    risks = data.get("risks", [])
    if not isinstance(risks, list):
        risks = []
    mitigations = data.get("mitigations", [])
    if not isinstance(mitigations, list):
        mitigations = []

    # Lightweight guardrail: warn when many facts produced too few key points.
    fact_count = len(all_facts_for_prompt)
    if fact_count >= 20 and len(key_points) < 6:
        print(
            f"[WARN] Dimension {dimension_name}: all_facts={fact_count}, "
            f"but key_points has only {len(key_points)} items. "
            f"adding grounded point-completion from source facts."
        )
        key_points = complete_key_points_from_facts(
            dimension_name,
            key_points,
            all_facts_for_prompt,
            target_min=6,
        )

    return {
        "summary": summary.strip(),
        "key_points": _dedupe_strings(key_points),
        "risks": _dedupe_strings(risks),
        "mitigations": _dedupe_strings(mitigations),
    }


def run_build(proposal_id: str):
    pid = proposal_id
    facts = load_raw_facts(proposal_id)
    # Correct risk/mitigation labels once globally.
    facts = reclassify_risk_mitigation_global(facts)
    grouped = group_facts_by_dimension(facts)

    # Write dimension_facts.json for manual inspection and debugging.
    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_facts_path = out_dir / "dimension_facts.json"
    dim_facts_path.write_text(
        json.dumps(grouped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Wrote facts grouped by dimension: {dim_facts_path}")

    dimensions_result: Dict[str, Dict[str, Any]] = {}

    _write_progress(0, len(DIMENSION_NAMES), pid)
    for dim_idx, dim in enumerate(DIMENSION_NAMES):
        dim_facts = grouped.get(dim, [])
        print(f"\n[INFO] Building dimension {dim} ...")

        if not dim_facts:
            # No facts were found, so write an explicit information-gap placeholder.
            dimensions_result[dim] = {
                "summary": f"The proposal contains limited explicit information about the {dim} dimension, so a detailed summary cannot be produced.",
                "key_points": [],
                "risks": [f"The proposal provides limited detail for the {dim} dimension, which may affect evaluation quality."],
                "mitigations": ["The proposal does not specify how to address or mitigate this information gap."],
            }
            print(f"[INFO] Dimension {dim} has no facts; wrote placeholder result.")
            _write_progress(dim_idx + 1, len(DIMENSION_NAMES), pid)
            continue

        data = call_llm_for_dimension(dim, dim_facts)

        # ==== Risk coverage marker ====
        risk_count = len(data.get("risks", []) or [])
        source_risk_count = sum(
            1
            for f in dim_facts
            if f.get("type", "") == "risk" or _looks_like_risk(f.get("text", "") or "")
        )
        source_mitigation_count = sum(
            1
            for f in dim_facts
            if f.get("type", "") == "mitigation" or _looks_like_mitigation(f.get("text", "") or "")
        )
        if risk_count == 0:
            level = "low"
            reason = "The proposal contains almost no explicit risk information for this dimension, so risk analysis is limited."
        elif risk_count <= 2:
            level = "medium"
            reason = "The proposal contains only limited risk information for this dimension, so risk granularity is limited."
        else:
            level = "high"
            reason = "The proposal contains relatively rich risk information for this dimension, allowing more detailed risk analysis."

        data["risk_coverage"] = {
            "level": level,
            "reason": reason,
            "risk_count": risk_count,
            "source_risk_count": source_risk_count,
            "source_mitigation_count": source_mitigation_count,
        }

        dimensions_result[dim] = data
        _write_progress(dim_idx + 1, len(DIMENSION_NAMES), pid)

        print(
            f"[INFO] Dimension {dim} complete: summary_len={len(data['summary'])}, "
            f"key_points={len(data['key_points'])}, risks={len(data['risks'])}, "
            f"mitigations={len(data['mitigations'])}, "
            f"risk_coverage={level}"
        )

    # 1) Write the per-proposal dimension file.
    out_path = out_dir / "dimensions_v2.json"
    out_path.write_text(
        json.dumps(dimensions_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] Generated the five-dimension file: {out_path}")

    # 2) Also write the global parsed file used by llm_answering.
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_path = PARSED_DIR / "parsed_dimensions.clean.llm.json"

    parsed_obj = {
        dim: {
            "summary": data.get("summary", ""),
            "key_points": data.get("key_points", []),
            "risks": data.get("risks", []),
            "mitigations": data.get("mitigations", []),
        }
        for dim, data in dimensions_result.items()
    }

    parsed_path.write_text(
        json.dumps(parsed_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Wrote parsed dimension file for llm_answering: {parsed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: build the five-dimension dimensions_v2.json from raw_facts.jsonl"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="Proposal ID, matching src/data/extracted/<proposal_id>",
    )
    args = parser.parse_args()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        # Default to the latest subdirectory under extracted.
        if not EXTRACTED_DIR.exists():
            raise FileNotFoundError(f"Extracted directory not found: {EXTRACTED_DIR}")
        candidates = [
            (d.stat().st_mtime, d.name)
            for d in EXTRACTED_DIR.iterdir()
            if d.is_dir()
        ]
        if not candidates:
            raise FileNotFoundError(f"No proposal subdirectories found under extracted: {EXTRACTED_DIR}")
        pid = max(candidates, key=lambda x: x[0])[1]
        print(f"[INFO] [auto] Selected latest proposal ID: {pid}")

    run_build(pid)


if __name__ == "__main__":
    main()
