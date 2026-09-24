#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B4 component 1 — ablate the "AI-powered" innovation cap.

Design
------
  condition A  cap present        = ALREADY CACHED, no calls:
                                      OpenAI -> canonical_ai_scores_dataset1.json (+ run3)
                                      Gemini -> cross_provider_gemini_raw.json (2 runs)
  condition B  cap header removed = pre-built variant (sha 27977547…). Retains a
                                    worked example that restates the cap for p1's
                                    archetype -> partial ablation, kept for
                                    completeness because it was pre-registered.
  condition C  cap fully removed  = genuine ablation (sha 926d894b…), built by
                                    build_b4_prompt_c.py.

Run for BOTH providers so the cap's effect can be separated from the scorer:
2 providers x 2 variants x 2 runs x 12 proposals = 96 Call B invocations.

Everything except the system prompt is held fixed and identical to condition A:
the same untouched Jun-16 dimensions_v2.json evidence, the same user-payload
construction, temperature=0.0, max_tokens=600, seed=42 (OpenAI only — Gemini has
no seed parameter), the frozen Stage-5 qa_echo/dim_weights/confidence, and the
canonical OR-logic verdict rule.

Output: canonical/b4_cap_ablation_raw.json

Usage:
    set -a && source .env && set +a
    python run_b4_cap_ablation.py --dry_run
    python run_b4_cap_ablation.py
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
DIMS = aeo.DIM_ORDER
INVEST_WEIGHT = 0.90
SCORE_THRESHOLD, CONF_THRESHOLD = 0.725, 0.811

CANON_PATH = BASE_DIR / "canonical" / "canonical_ai_scores_dataset1.json"
VARIANTS_PATH = BASE_DIR / "canonical" / "b4_prompt_variants.json"
EXTRACTED = BASE_DIR / "src" / "data" / "extracted"
OUT_PATH = BASE_DIR / "canonical" / "b4_cap_ablation_raw.json"

ARMS = [("openai", "gpt-4o-mini"), ("gemini", "gemini-3.5-flash-lite")]
VARIANTS = ["B_cap_removed", "C_cap_fully_removed"]
GEMINI_THINKING = "minimal"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_verdict(overall: float, confidence: float, blended: Dict[str, float]) -> str:
    if overall >= SCORE_THRESHOLD or confidence >= CONF_THRESHOLD:
        return "GO"
    if overall < 0.40 or blended["innovation"] < 0.30 or blended["feasibility"] < 0.30:
        return "NO-GO"
    return "HOLD"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pids", default="")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    wanted = [p.strip() for p in args.pids.split(",") if p.strip()] or list(PROPOSALS)
    unknown = [p for p in wanted if p not in PROPOSALS]
    if unknown:
        print(f"[FAIL] unknown pids: {unknown}")
        return 2
    if not VARIANTS_PATH.exists():
        print(f"[FAIL] prompt variants not found: {VARIANTS_PATH} (run build_b4_prompt_c.py)")
        return 2

    variants = json.loads(VARIANTS_PATH.read_text())["variants"]
    canon = json.loads(CANON_PATH.read_text(encoding="utf-8"))["canonical"]

    missing = [p for p in wanted if not (EXTRACTED / PROPOSALS[p] / "dimensions_v2.json").exists()]
    if missing:
        print(f"[FAIL] dimensions_v2.json missing for: {missing}")
        return 2

    # sanity: the live production prompt must still equal variant A
    if aeo.INVEST_SCORE_SYSTEM != variants["A_production"]["text"]:
        print("[FAIL] live INVEST_SCORE_SYSTEM != stored variant A — prompt drifted.")
        return 2

    if not args.dry_run:
        for provider, _ in ARMS:
            if not provider_api_key(provider):
                env = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
                print(f"[FAIL] {env} is not set. set -a && source .env && set +a")
                return 2

    os.environ["GEMINI_THINKING_LEVEL"] = GEMINI_THINKING
    total = len(ARMS) * len(VARIANTS) * args.runs * len(wanted)
    print(f"[CONFIG] arms={[a for a, _ in ARMS]} variants={VARIANTS} runs={args.runs} "
          f"n_proposals={len(wanted)} -> {total} calls")
    for v in VARIANTS:
        print(f"[CONFIG] {v}: sha {variants[v]['sha256'][:16]}… {variants[v]['chars']} chars")

    if args.dry_run:
        print("\n[OK] dry run: wiring, evidence and prompt variants verified. No LLM calls.")
        return 0

    real_call = aeo.call_openai_chat
    real_prompt = aeo.INVEST_SCORE_SYSTEM
    captured: Dict[str, Any] = {}
    results: Dict[str, Any] = {}
    failures: Dict[str, list] = {}
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)

    def spy(**kwargs):
        assert kwargs["temperature"] == 0.0, f"temperature={kwargs['temperature']}"
        assert kwargs["seed"] == 42, f"seed={kwargs['seed']}"
        assert kwargs["system_prompt"] == aeo.INVEST_SCORE_SYSTEM, "prompt mismatch at call time"
        resp = real_call(**kwargs)
        captured["max_tokens"] = kwargs["max_tokens"]
        captured["raw_response"] = resp
        return resp

    aeo.call_openai_chat = spy
    try:
        for provider, model in ARMS:
            for variant in VARIANTS:
                aeo.INVEST_SCORE_SYSTEM = variants[variant]["text"]
                for run_idx in range(1, args.runs + 1):
                    key = f"{provider}__{variant}__run{run_idx}"
                    results[key] = {}
                    failures[key] = []
                    print(f"\n########## {key} ##########")
                    for i, cpid in enumerate(wanted, 1):
                        pid = PROPOSALS[cpid]
                        dims_path = EXTRACTED / pid / "dimensions_v2.json"
                        c = canon[cpid]
                        captured.clear()
                        invest = aeo.investment_score_from_dims(pid, provider, model, dims_path)
                        if len(invest) != len(DIMS):
                            print(f"  [FAIL] {cpid}: {len(invest)}/{len(DIMS)} dims")
                            failures[key].append(cpid)
                            continue
                        qa, w, conf = c["qa_echo"], c["dim_weights"], float(c["confidence"])
                        blended = {d: round(INVEST_WEIGHT * invest[d] + (1 - INVEST_WEIGHT) * qa[d], 4)
                                   for d in DIMS}
                        num = sum(blended[d] * float(w.get(d, 1.0)) for d in DIMS)
                        den = sum(float(w.get(d, 1.0)) for d in DIMS)
                        overall = round(num / den, 4)
                        raw = captured.get("raw_response", {}) or {}
                        results[key][cpid] = {
                            "pid": cpid, "overall_score": overall,
                            "dimension_scores_blended": blended,
                            "invest_scores_raw": {d: invest[d] for d in DIMS},
                            "confidence": conf,
                            "verdict": derive_verdict(overall, conf, blended),
                            "categories": raw.get("categories", {}) or {},
                            "rationale": raw.get("rationale", {}) or {},
                        }
                        print(f"  [{i:2}/{len(wanted)}] {cpid}: innov={invest['innovation']:.2f} "
                              f"({(raw.get('categories') or {}).get('innovation','?')}) "
                              f"overall={overall:.4f}")
    finally:
        aeo.call_openai_chat = real_call
        aeo.INVEST_SCORE_SYSTEM = real_prompt

    payload = {
        "provenance": {
            "study": "b4_cap_ablation",
            "paper_fix": "B4",
            "date_utc": started.isoformat(),
            "date_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "arms": {p: m for p, m in ARMS},
            "variants": {v: {"sha256": variants[v]["sha256"], "chars": variants[v]["chars"],
                             "role": variants[v]["role"]} for v in VARIANTS},
            "condition_A_source": {
                "openai": "canonical/canonical_ai_scores_dataset1.json + stage6_repeatability_run3.json",
                "gemini": "canonical/cross_provider_gemini_raw.json (2 runs)"},
            "runs_per_cell": args.runs,
            "temperature": 0.0,
            "seed_requested": 42,
            "seed_reached_provider": {"openai": True, "gemini": False},
            "gemini_thinking_level": GEMINI_THINKING,
            "max_tokens": captured.get("max_tokens", 600),
            "invest_weight": INVEST_WEIGHT,
            "evidence_source": "src/data/extracted/<pid>__openai/dimensions_v2.json (Jun-16, untouched)",
            "frozen_inputs": "qa_echo, dim_weights, confidence from canonical_ai_scores_dataset1.json",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "failures": failures,
            "not_modified": ["canonical/ (existing files)", "results/", "resultsNew/", "src/data/"],
        },
        "cells": results,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {args.out}")
    for k, v in results.items():
        print(f"     {k}: {len(v)} proposals, failures={failures[k]}")
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
