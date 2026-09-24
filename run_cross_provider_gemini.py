#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-provider replication check (paper fix B3) — Gemini arm, Dataset1.

What this does
--------------
Re-runs ONLY Stage 6 "Call B" (deterministic investment-grade scoring) under
Gemini for the 12 Dataset1 proposals (p1-p8, pA-pD), reading the *same
untouched* Jun-16 `src/data/extracted/<pid>__openai/dimensions_v2.json`
evidence used by the canonical OpenAI run. Stages 0-5 are not re-run and
nothing under canonical/, results/ or resultsNew/ is modified.

Call B is the only LLM-varying component of the canonical scoring chain. The
provenance was traced explicitly before this run: the `overall_ranking` field
in canonical/canonical_ai_scores_dataset1.json is the dimension-weight-weighted
mean of the Stage 6 blended dimensions (max residual 0.00064 across 12
proposals), NOT Stage 8's rubric re-scoring of the compiled report. Stage 8
output exists for all 12 pids but correlates with the canonical set at only
rho=0.14 overall and never fed the evaluation (the per_item `provider` column
reads "report"). So swapping the provider for Call B alone changes exactly the
quantity the paper reports.

`qa_echo`, `dim_weights` and `confidence` are the FROZEN Stage-5 values from
canonical/canonical_ai_scores_dataset1.json — identical for both providers by
construction, so every OpenAI-vs-Gemini difference is attributable to Call B.

The production function `src.tools.ai_expert_opinion.investment_score_from_dims`
is called directly, so INVEST_SCORE_SYSTEM, the user payload construction,
temperature=0.0 and max_tokens=600 are byte-for-byte those of the OpenAI runs.
`call_openai_chat` is wrapped by a transparent spy that asserts the invariants
and captures the raw response; it forwards the call unmodified.

Two runs are executed (same two-run protocol as the baseline and the
repeatability work) so provider differences can be separated from run-to-run
scorer noise.

SEED CAVEAT (important, and recorded in the output provenance)
--------------------------------------------------------------
Gemini has no seed parameter. `llm_provider.chat_completion` accepts `seed` but
the Gemini branch calls `_call_gemini(...)` without it, and the v1beta
generateContent surface exposes no seed field. The spy's `seed == 42` assert
therefore still passes — it checks the value entering `call_openai_chat`, one
layer above where the seed is discarded. The Gemini arm is effectively
UNSEEDED while the OpenAI arm is nominally seeded. This asymmetry is recorded
as `seed_requested` / `seed_reached_provider` and must be stated in the paper.

THINKING
--------
Gemini 3.x models think by default and thinking tokens draw on maxOutputTokens;
at Call B's max_tokens=600 that can return empty `parts`. GEMINI_THINKING_LEVEL
is set to "minimal" (opt-in env gate added to `_call_gemini`) to keep the arm
comparable to a non-reasoning gpt-4o-mini and to protect the 600-token budget.

Output
------
canonical/cross_provider_gemini_raw.json   (raw two-run scores; analysis is a
                                            separate, LLM-free step)

Usage:
    set -a && source .env && set +a
    python run_cross_provider_gemini.py --dry_run     # verify wiring, no calls
    python run_cross_provider_gemini.py               # 2 runs x 12 = 24 calls
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import src.tools.ai_expert_opinion as aeo  # noqa: E402
from src.backend.utils.llm_provider import provider_api_key  # noqa: E402

PROPOSALS = {p: f"{p}__openai" for p in
             ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]}

DIMS = aeo.DIM_ORDER  # team, objectives, strategy, innovation, feasibility
INVEST_WEIGHT = 0.90  # must match ai_expert_opinion.main()

CANON_PATH = BASE_DIR / "canonical" / "canonical_ai_scores_dataset1.json"
EXTRACTED = BASE_DIR / "src" / "data" / "extracted"
OUT_PATH = BASE_DIR / "canonical" / "cross_provider_gemini_raw.json"

# Canonical verdict rule (evaluation_summary.json OR-logic) — identical to the
# rule applied to the OpenAI runs, so verdict deltas reflect score deltas only.
SCORE_THRESHOLD = 0.725
CONF_THRESHOLD = 0.811

DEFAULT_MODEL = "gemini-3.5-flash-lite"
THINKING_LEVEL = "minimal"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_verdict(overall: float, confidence: float, blended: Dict[str, float]) -> str:
    """Canonical GO/HOLD/NO-GO rule, mirroring ai_expert_opinion.verdict_rule."""
    if overall >= SCORE_THRESHOLD or confidence >= CONF_THRESHOLD:
        return "GO"
    if overall < 0.40 or blended["innovation"] < 0.30 or blended["feasibility"] < 0.30:
        return "NO-GO"
    return "HOLD"


def score_one(cpid: str, provider: str, model: str, canon: Dict[str, Any],
              captured: Dict[str, Any]) -> Dict[str, Any] | None:
    """One Call B invocation + the deterministic canonical derivation."""
    pid = PROPOSALS[cpid]
    dims_path = EXTRACTED / pid / "dimensions_v2.json"
    c = canon[cpid]

    captured.clear()
    invest = aeo.investment_score_from_dims(pid, provider, model, dims_path)
    if len(invest) != len(DIMS):
        print(f"  [FAIL] {cpid}: got {len(invest)}/{len(DIMS)} dimension scores")
        return None

    qa = c["qa_echo"]
    weights = c["dim_weights"]
    confidence = float(c["confidence"])

    blended = {d: round(INVEST_WEIGHT * invest[d] + (1 - INVEST_WEIGHT) * qa[d], 4)
               for d in DIMS}
    num = sum(blended[d] * float(weights.get(d, 1.0)) for d in DIMS)
    den = sum(float(weights.get(d, 1.0)) for d in DIMS)
    overall = round(num / den, 4)
    verdict = derive_verdict(overall, confidence, blended)

    raw = captured.get("raw_response", {}) or {}
    print(f"  invest={ {d: invest[d] for d in DIMS} }")
    print(f"  overall={overall:.4f} confidence={confidence:.4f} verdict={verdict}")

    return {
        "pid": cpid,
        "pipeline_pid": pid,
        "file_name": c.get("file_name", ""),
        "overall_score": overall,
        "dimension_scores_blended": blended,
        "invest_scores_raw": {d: invest[d] for d in DIMS},
        "qa_scores_frozen": {d: qa[d] for d in DIMS},
        "confidence": confidence,
        "verdict": verdict,
        "categories": raw.get("categories", {}) or {},
        "rationale": raw.get("rationale", {}) or {},
        "evidence_input": {
            "path": str(dims_path.relative_to(BASE_DIR)),
            "sha256": sha256(dims_path),
            "mtime_utc": datetime.fromtimestamp(
                dims_path.stat().st_mtime, timezone.utc).isoformat(),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pids", default="", help="Comma-separated subset of canonical pids.")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=2, help="Independent samples (default 2).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate inputs and wiring without calling the LLM.")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    provider = args.provider.strip().lower()
    model = args.model.strip()
    out_path = Path(args.out)

    wanted = [p.strip() for p in args.pids.split(",") if p.strip()] or list(PROPOSALS)
    unknown = [p for p in wanted if p not in PROPOSALS]
    if unknown:
        print(f"[FAIL] unknown pids: {unknown}")
        return 2

    if not CANON_PATH.exists():
        print(f"[FAIL] canonical score file not found: {CANON_PATH}")
        return 2
    canon = json.loads(CANON_PATH.read_text(encoding="utf-8"))["canonical"]

    # Pre-flight: every evidence file must exist BEFORE any LLM spend.
    missing = [p for p in wanted if not (EXTRACTED / PROPOSALS[p] / "dimensions_v2.json").exists()]
    if missing:
        print(f"[FAIL] dimensions_v2.json missing for: {missing}")
        return 2

    if not args.dry_run and not provider_api_key(provider):
        print("[FAIL] GEMINI_API_KEY is not set in the environment.")
        print("       set -a && source .env && set +a")
        return 2

    # Protect the 600-token Call B budget from default-on 3.x thinking.
    os.environ["GEMINI_THINKING_LEVEL"] = THINKING_LEVEL

    print(f"[CONFIG] provider={provider} model={model} temperature=0.0 "
          f"thinking_level={THINKING_LEVEL} invest_weight={INVEST_WEIGHT} "
          f"n_proposals={len(wanted)} runs={args.runs}")
    print("[CONFIG] seed=42 requested at call_openai_chat, but Gemini has no seed "
          "parameter -- this arm is UNSEEDED.")

    # --- transparent spy over the production LLM call -----------------------
    real_call = aeo.call_openai_chat
    captured: Dict[str, Any] = {}

    def spy(**kwargs):
        # Invariants that make this arm comparable to the OpenAI runs.
        assert kwargs["system_prompt"] == aeo.INVEST_SCORE_SYSTEM, "system prompt drifted"
        assert kwargs["temperature"] == 0.0, f"temperature={kwargs['temperature']}, expected 0.0"
        assert kwargs["seed"] == 42, f"seed={kwargs['seed']}, expected 42"
        resp = real_call(**kwargs)
        captured["user_payload"] = kwargs["user_payload"]
        captured["max_tokens"] = kwargs["max_tokens"]
        captured["raw_response"] = resp
        return resp

    aeo.call_openai_chat = spy

    runs: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, list] = {}
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    try:
        if args.dry_run:
            for cpid in wanted:
                dims_path = EXTRACTED / PROPOSALS[cpid] / "dimensions_v2.json"
                print(f"  dry_run {cpid}: evidence={dims_path.name} sha256={sha256(dims_path)[:16]}...")
            print("\n[OK] dry run complete; wiring and evidence files verified. No LLM calls made.")
            return 0

        for run_idx in range(1, args.runs + 1):
            label = f"gemini_run{run_idx}"
            runs[label] = {}
            failures[label] = []
            print(f"\n########## {label} ##########")
            for i, cpid in enumerate(wanted, 1):
                print(f"\n[{label} {i}/{len(wanted)}] === {cpid} ({PROPOSALS[cpid]}) ===")
                rec = score_one(cpid, provider, model, canon, captured)
                if rec is None:
                    failures[label].append(cpid)
                else:
                    runs[label][cpid] = rec
    finally:
        aeo.call_openai_chat = real_call

    if not any(runs.values()):
        print("\n[FAIL] no proposals scored; nothing written.")
        return 1

    payload = {
        "provenance": {
            "study": "cross_provider_replication",
            "paper_fix": "B3",
            "description": ("Stage 6 Call B re-run under Gemini on the 12 Dataset1 "
                            "proposals, reading the untouched Jun-16 dimensions_v2.json "
                            "evidence used by the canonical OpenAI run."),
            "stage": "stage_6_call_b_investment_scoring",
            "date_utc": started.isoformat(),
            "date_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider,
            "model": model,
            "model_tier": "Flash-Lite (GA) -- tier match for gpt-4o-mini; not a Pro/flagship model",
            "temperature": 0.0,
            "seed_requested": 42,
            "seed_reached_provider": False,
            "seed_note": ("Gemini exposes no seed parameter; llm_provider._call_gemini "
                          "does not receive or forward `seed`. This arm is UNSEEDED "
                          "while the OpenAI arm is nominally seeded (best-effort)."),
            "thinking_level": THINKING_LEVEL,
            "thinking_note": ("Gemini 3.x thinks by default and thinking tokens draw on "
                              "maxOutputTokens; pinned to 'minimal' to protect the 600-token "
                              "Call B budget and stay comparable to non-reasoning gpt-4o-mini. "
                              "2.5-era thinkingBudget=0 is rejected with HTTP 400 by 3.x."),
            "max_tokens": captured.get("max_tokens", 600),
            "invest_weight": INVEST_WEIGHT,
            "runs": args.runs,
            "verdict_rule": (f"GO iff overall >= {SCORE_THRESHOLD} OR confidence >= "
                             f"{CONF_THRESHOLD}; NO-GO iff overall < 0.40 or blended "
                             f"innovation < 0.30 or blended feasibility < 0.30; else HOLD"),
            "evidence_source": ("existing src/data/extracted/<pid>__openai/dimensions_v2.json "
                                "(Jun-16); no re-extraction performed"),
            "frozen_inputs": ("qa_echo, dim_weights and confidence taken from "
                              "canonical/canonical_ai_scores_dataset1.json (Stage-5 output, "
                              "identical across providers by construction)"),
            "system_prompt_sha256": hashlib.sha256(
                aeo.INVEST_SCORE_SYSTEM.encode("utf-8")).hexdigest(),
            "code_source": "src.tools.ai_expert_opinion.investment_score_from_dims",
            "credential_source": "GEMINI_API_KEY environment variable",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "failures": failures,
            "not_modified": ["canonical/ (existing files)", "results/", "resultsNew/",
                             "src/data/ (all stages)"],
        },
        **runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] wrote {out_path}")
    for label, recs in runs.items():
        print(f"     {label}: {len(recs)} proposals, failures={failures[label]}")
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
