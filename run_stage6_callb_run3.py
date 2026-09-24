#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6 "Call B" (deterministic investment-grade scoring) -- REPEATABILITY RUN 3.

This adds a third independent sample of the Stage 6 Call B LLM call to the
existing 2-run repeatability check reported in the paper (Section 8: 11/12
proposals differed between runs 1 and 2).

What this does
--------------
For each of the 12 Dataset1 proposals (p1-p8, pA-pD) it invokes the *actual*
production function `src.tools.ai_expert_opinion.investment_score_from_dims`
with the *existing* `src/data/extracted/<pid>/dimensions_v2.json` evidence file.
No re-extraction is performed; the Stage 1-5 pipeline is not touched.

Because the production function is called directly, the system prompt
(INVEST_SCORE_SYSTEM), the user payload construction, temperature=0.0,
seed=42 and max_tokens=600 are byte-for-byte those of runs 1 and 2.
`call_openai_chat` is wrapped by a transparent spy so the raw response
(scores + categories + rationale) can be persisted; the spy forwards the
call unmodified and asserts the invariant parameters.

Derived quantities (identical formulas to the production pipeline):
  blended_dim = INVEST_WEIGHT * invest_dim + (1 - INVEST_WEIGHT) * qa_dim
  overall     = dimension-weight-weighted mean of blended_dim
  verdict     = canonical OR-logic rule (GO iff overall >= 0.725 OR conf >= 0.811)

`qa_echo`, `dim_weights` and `confidence` are the FROZEN Stage-5 values from
canonical/canonical_ai_scores_dataset1.json. Stage 5 was never regenerated, so
these are identical across runs 1, 2 and 3 by construction -- which means every
run-to-run difference observed is attributable to Call B alone.

Output
------
canonical/stage6_repeatability_run3.json   (new file; nothing else is modified)

Credentials are read from the environment only (OPENAI_API_KEY); no key-file
path is stored in this repo. Costs 12 short LLM calls.

Usage:
    set -a && source /path/to/your/.env && set +a   # or just export the key
    python run_stage6_callb_run3.py                 # run all 12
    python run_stage6_callb_run3.py --pids p1,p2    # subset
    python run_stage6_callb_run3.py --dry_run       # verify wiring, no LLM call
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Credentials come from the environment only -- no path to a key file is stored
# in this repo. Provide OPENAI_API_KEY (and optionally OPENAI_MODEL) before
# running, e.g. by sourcing whatever .env you keep outside the repo:
#     set -a && source /path/to/your/.env && set +a
#     python run_stage6_callb_run3.py
# A repo-local ./.env is honoured if you create one (it is gitignored), because
# llm_provider and ai_expert_opinion both call load_dotenv() at import time.
import src.tools.ai_expert_opinion as aeo  # noqa: E402
from src.backend.utils.llm_provider import default_model, provider_api_key  # noqa: E402

# The 12 Dataset1 proposals. Key = canonical pid, value = pipeline pid (dir name).
PROPOSALS = {p: f"{p}__openai" for p in
             ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]}

DIMS = aeo.DIM_ORDER  # team, objectives, strategy, innovation, feasibility
INVEST_WEIGHT = 0.90  # must match ai_expert_opinion.main()

CANON_PATH = BASE_DIR / "canonical" / "canonical_ai_scores_dataset1.json"
EXTRACTED = BASE_DIR / "src" / "data" / "extracted"
OUT_PATH = BASE_DIR / "canonical" / "stage6_repeatability_run3.json"

# Canonical verdict rule (evaluation_summary.json OR-logic), applied identically
# to all three runs by the analysis script so verdict deltas reflect score
# deltas only -- not the AND->OR rule change that happened in git between runs.
SCORE_THRESHOLD = 0.725
CONF_THRESHOLD = 0.811


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_verdict(overall: float, confidence: float, blended: Dict[str, float]) -> str:
    """Canonical GO/HOLD/NO-GO rule, mirroring ai_expert_opinion.verdict_rule."""
    if overall >= SCORE_THRESHOLD or confidence >= CONF_THRESHOLD:
        return "GO"
    if overall < 0.40 or blended["innovation"] < 0.30 or blended["feasibility"] < 0.30:
        return "NO-GO"
    return "HOLD"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pids", default="", help="Comma-separated subset of canonical pids.")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="", help="Defaults to OPENAI_MODEL / gpt-4o-mini.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate inputs and wiring without calling the LLM.")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    provider = args.provider.strip().lower()
    model = args.model.strip() or default_model(provider)
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
        env_var = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
                   "gemini": "GEMINI_API_KEY"}.get(provider, "OPENAI_API_KEY")
        print(f"[FAIL] {env_var} is not set in the environment.")
        print(f"       Export it, or source a .env you keep outside the repo:")
        print(f"           set -a && source /path/to/your/.env && set +a")
        return 2

    print(f"[CONFIG] provider={provider} model={model} temperature=0.0 seed=42 "
          f"invest_weight={INVEST_WEIGHT} n_proposals={len(wanted)}")

    # --- transparent spy over the production LLM call -----------------------
    real_call = aeo.call_openai_chat
    captured: Dict[str, Any] = {}

    def spy(**kwargs):
        # Invariants that make run 3 comparable to runs 1 and 2.
        assert kwargs["system_prompt"] == aeo.INVEST_SCORE_SYSTEM, "system prompt drifted"
        assert kwargs["temperature"] == 0.0, f"temperature={kwargs['temperature']}, expected 0.0"
        assert kwargs["seed"] == 42, f"seed={kwargs['seed']}, expected 42"
        resp = real_call(**kwargs)
        captured["user_payload"] = kwargs["user_payload"]
        captured["max_tokens"] = kwargs["max_tokens"]
        captured["raw_response"] = resp
        return resp

    aeo.call_openai_chat = spy

    results: Dict[str, Any] = {}
    failures = []
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    try:
        for i, cpid in enumerate(wanted, 1):
            pid = PROPOSALS[cpid]
            dims_path = EXTRACTED / pid / "dimensions_v2.json"
            c = canon[cpid]
            print(f"\n[{i}/{len(wanted)}] === {cpid} ({pid}) ===")

            if args.dry_run:
                print(f"  dry_run: evidence={dims_path} sha256={sha256(dims_path)[:16]}…")
                continue

            captured.clear()
            invest = aeo.investment_score_from_dims(pid, provider, model, dims_path)
            if len(invest) != len(DIMS):
                print(f"  [FAIL] {cpid}: got {len(invest)}/{len(DIMS)} dimension scores")
                failures.append(cpid)
                continue

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
            results[cpid] = {
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
            print(f"  invest={ {d: invest[d] for d in DIMS} }")
            print(f"  overall={overall:.4f} confidence={confidence:.4f} verdict={verdict}")
    finally:
        aeo.call_openai_chat = real_call

    if args.dry_run:
        print("\n[OK] dry run complete; wiring and evidence files verified. No LLM calls made.")
        return 0

    if not results:
        print("\n[FAIL] no proposals scored; nothing written.")
        return 1

    payload = {
        "provenance": {
            "run_label": "run3",
            "description": ("Third independent sample of Stage 6 Call B (deterministic "
                            "investment-grade scoring), added to the 2-run repeatability "
                            "check in Section 8."),
            "stage": "stage_6_call_b_investment_scoring",
            "date_utc": started.isoformat(),
            "date_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider,
            "model": model,
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": captured.get("max_tokens", 600),
            "invest_weight": INVEST_WEIGHT,
            "verdict_rule": (f"GO iff overall >= {SCORE_THRESHOLD} OR confidence >= "
                             f"{CONF_THRESHOLD}; NO-GO iff overall < 0.40 or blended "
                             f"innovation < 0.30 or blended feasibility < 0.30; else HOLD"),
            "evidence_source": ("existing src/data/extracted/<pid>/dimensions_v2.json "
                                "(Jun-16); no re-extraction performed"),
            "frozen_inputs": ("qa_echo, dim_weights and confidence taken from "
                              "canonical/canonical_ai_scores_dataset1.json (Stage-5 output, "
                              "never regenerated -> identical across runs 1/2/3)"),
            "system_prompt_sha256": hashlib.sha256(
                aeo.INVEST_SCORE_SYSTEM.encode("utf-8")).hexdigest(),
            "code_source": "src/tools/ai_expert_opinion.investment_score_from_dims",
            "credential_source": "OPENAI_API_KEY environment variable",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "n_proposals": len(results),
            "failures": failures,
            "not_modified": ["canonical/canonical_ai_scores_dataset1.json",
                             "canonical/canonical_kickstarter_scores.json",
                             "results/", "resultsNew/", "src/data/ (all stages)"],
        },
        "run3": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] wrote {out_path} ({len(results)} proposals)")
    if failures:
        print(f"[WARN] failed proposals: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
