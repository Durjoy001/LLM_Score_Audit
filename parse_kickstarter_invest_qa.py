#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recover per-dimension invest/qa echoes for the Kickstarter set from
results/kickstarter_full_run.log, for the B4 blend sweep (component 3).

Why fingerprint matching rather than positional parsing
-------------------------------------------------------
The Kickstarter log is a CONCURRENT multi-proposal run: `[LIFECYCLE] base_pid=`
markers and per-stage `[INPUT] proposal_id=` lines interleave across proposals,
so "the pid most recently seen before this [INVEST] line" is NOT reliable.

Instead each parsed block is matched to a pid by FINGERPRINT: the block's five
`[DIM_OUTPUT] ... score=` values are the final blended dimension scores, and
canonical/canonical_kickstarter_scores.json holds those same five values per
proposal. A block is assigned to the unique pid whose five canonical dimensions
match within tolerance. This is independent of log ordering and validates the
recovery at the same time.

A block is only accepted if:
  * it has exactly 5 DIM_OUTPUT lines with qa= and invest=, and
  * exactly ONE canonical pid matches all five score= values within TOL, and
  * blended@0.90 = 0.90*invest + 0.10*qa reproduces the canonical dimension.

Anything ambiguous is dropped and reported, never guessed.

Output: canonical/b4_kickstarter_invest_qa.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "results" / "kickstarter_full_run.log"
CANON = json.loads((ROOT / "canonical" / "canonical_kickstarter_scores.json").read_text())["scores"]
OUT = ROOT / "canonical" / "b4_kickstarter_invest_qa.json"

LOG_DIMS = ["team", "objectives", "strategy", "innovation", "feasibility"]
H2L = {"team": "team", "objective": "objectives", "strategy": "strategy",
       "advantages": "innovation", "feasibility": "feasibility"}
TOL = 0.0015          # canonical dims are rounded to 3dp
IW = 0.90

INVEST_RE = re.compile(r"\[INVEST\] investment scores: (\{[^\n]+\})")
DIM_RE = re.compile(r"\[DIM_OUTPUT\] (\w+): .*?score=([0-9.]+) \(qa=([0-9.]+) invest=([0-9.]+)\)")


def parse_blocks(text: str):
    """Yield (invest_dict, {dim: (score, qa, invest)}) blocks in log order."""
    lines = text.split("\n")
    blocks, cur_invest, cur_dims = [], None, {}
    for line in lines:
        mi = INVEST_RE.search(line)
        if mi:
            if cur_invest is not None and len(cur_dims) == 5:
                blocks.append((cur_invest, cur_dims))
            try:
                cur_invest = json.loads(mi.group(1).replace("'", '"'))
            except Exception:
                cur_invest = None
            cur_dims = {}
            continue
        md = DIM_RE.search(line)
        if md and cur_invest is not None:
            cur_dims[md.group(1)] = (float(md.group(2)), float(md.group(3)), float(md.group(4)))
    if cur_invest is not None and len(cur_dims) == 5:
        blocks.append((cur_invest, cur_dims))
    return blocks


def main() -> int:
    if not LOG.exists():
        print(f"[FAIL] log not found: {LOG}")
        return 2
    blocks = parse_blocks(LOG.read_text(errors="replace"))
    print(f"[INFO] parsed {len(blocks)} complete (INVEST + 5x DIM_OUTPUT) blocks")

    recovered, ambiguous, unmatched, blend_fail = {}, [], [], []
    for bi, (invest, dims) in enumerate(blocks):
        if set(dims) != set(LOG_DIMS) or set(invest) != set(LOG_DIMS):
            unmatched.append({"block": bi, "reason": "dimension set mismatch"})
            continue
        fp = {ld: dims[ld][0] for ld in LOG_DIMS}
        hits = []
        for pid, c in CANON.items():
            if all(abs(fp[H2L[h]] - float(c[h])) <= TOL for h in H2L):
                hits.append(pid)
        if len(hits) == 0:
            unmatched.append({"block": bi, "reason": "no canonical pid matches fingerprint",
                              "fingerprint": fp})
            continue
        if len(hits) > 1:
            ambiguous.append({"block": bi, "candidates": hits, "fingerprint": fp})
            continue
        pid = hits[0]
        if pid in recovered:
            ambiguous.append({"block": bi, "reason": "pid already assigned", "pid": pid})
            continue
        # independent check: blended@0.90 must reproduce the canonical dimension
        worst = max(abs((IW * invest[H2L[h]] + (1 - IW) * dims[H2L[h]][1]) - float(CANON[pid][h]))
                    for h in H2L)
        if worst > 0.006:
            blend_fail.append({"pid": pid, "max_blend_residual": round(worst, 5)})
            continue
        recovered[pid] = {
            "invest": {ld: invest[ld] for ld in LOG_DIMS},
            "qa": {ld: round(dims[ld][1], 4) for ld in LOG_DIMS},
            "blended_from_log": {ld: dims[ld][0] for ld in LOG_DIMS},
            "max_blend_residual_vs_canonical": round(worst, 5),
            "confidence": CANON[pid]["confidence"],
            "overall_ranking": CANON[pid]["overall_ranking"],
            "funded_pct": CANON[pid]["funded_pct"],
        }

    n_canon = len(CANON)
    print(f"[INFO] recovered {len(recovered)}/{n_canon} pids")
    print(f"[INFO] ambiguous={len(ambiguous)} unmatched={len(unmatched)} blend_fail={len(blend_fail)}")
    if recovered:
        print(f"[INFO] worst blend residual across recovered pids: "
              f"{max(v['max_blend_residual_vs_canonical'] for v in recovered.values()):.5f}")

    ok = len(recovered) >= 100 and not blend_fail
    payload = {
        "study": "b4_kickstarter_invest_qa_recovery",
        "validation": {
            "method": ("fingerprint match of the five [DIM_OUTPUT] score= values against "
                       "canonical/canonical_kickstarter_scores.json, plus an independent "
                       "blended@0.90 reproduction check"),
            "tolerance": TOL,
            "blend_check_tolerance": 0.006,
            "n_canonical": n_canon,
            "n_blocks_parsed": len(blocks),
            "n_recovered": len(recovered),
            "n_ambiguous": len(ambiguous),
            "n_unmatched": len(unmatched),
            "n_blend_fail": len(blend_fail),
            "max_blend_residual": (max(v["max_blend_residual_vs_canonical"]
                                       for v in recovered.values()) if recovered else None),
            "usable_for_blend_sweep": ok,
            "note": ("pids not recovered are EXCLUDED from the Kickstarter blend sweep and "
                     "reported; none are imputed."),
        },
        "ambiguous": ambiguous,
        "unmatched": unmatched[:20],
        "blend_fail": blend_fail,
        "recovered": recovered,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{'OK' if ok else 'WARN'}] wrote {OUT}  usable_for_blend_sweep={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
