#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3 (extra) — AI-vs-consensus Spearman vs the leave-one-rater-out (LORO)
band of human agreement with the panel.

For each target (overall ranking + 5 dimensions):
  - AI vs FULL consensus (median dims / mean overall of all 3 raters)
  - For each rater r: rater r vs the consensus of the OTHER two raters
    (leave-one-rater-out, so a rater is never correlated against itself)

The three LORO values form the human agreement band. If the AI-vs-consensus
rho sits inside [min, max] of that band, the AI agrees with the panel about as
well as a human panelist does; if below, it is worse than the human floor.

Note the deliberate asymmetry (documented, not a bug): the AI is scored against
the 3-rater panel while each human is scored against a 2-rater panel — the
standard way to place a model on the human inter-rater ceiling.

Reads only cached artifacts (per_item AI scores + raters_long.csv). No LLM calls.
Output: resultsNew/Dataset1/evaluation/loro_spearman.json
"""
import csv
import json
import statistics
from pathlib import Path

import openpyxl
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
RES = BASE / "resultsNew" / "Dataset1"
PER_ITEM = RES / "evaluation" / "evaluation_report_consensus.xlsx"
LONG_CSV = RES / "raters_long.csv"

DIMS = ["team", "objective", "strategy", "advantages", "feasibility"]
TARGETS = ["overall_ranking"] + DIMS
# per_item row order (verified) -> pid
ROW_PIDS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]


def load_ai():
    """{target -> {pid -> ai_score_0_1}} from the consensus per_item sheet."""
    ws = openpyxl.load_workbook(PER_ITEM, data_only=True)["per_item"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = list(rows[0])
    out = {t: {} for t in TARGETS}
    for pid, r in zip(ROW_PIDS, rows[1:]):
        for t in TARGETS:
            v = r[hdr.index(f"ai_{t}")]
            out[t][pid] = float(v) if v not in (None, "") else None
    return out


def load_raters():
    """{target -> {rater_id -> {pid -> raw_score}}}."""
    out = {t: {1: {}, 2: {}, 3: {}} for t in TARGETS}
    id2pid = {i: p for i, p in enumerate(ROW_PIDS, start=1)}
    with LONG_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["dimension"]
            pid = id2pid[int(row["proposal_id"])]
            out[t][int(row["rater_id"])][pid] = float(row["score"])
    return out


def _sp(xs, ys):
    if len(xs) < 3:
        return None
    rho, _ = spearmanr(xs, ys)
    return None if rho != rho else round(float(rho), 4)


def main():
    ai = load_ai()
    rat = load_raters()
    result = {"targets": {}}

    for t in TARGETS:
        pids = ROW_PIDS
        # full consensus per pid (median dims, mean overall)
        def consensus(rater_ids, pid):
            vals = [rat[t][r][pid] for r in rater_ids]
            return statistics.mean(vals) if t == "overall_ranking" else statistics.median(vals)

        full_cons = [consensus([1, 2, 3], p) for p in pids]
        ai_vals = [ai[t][p] for p in pids]
        ai_rho = _sp(ai_vals, full_cons) if None not in ai_vals else None

        loro = {}
        for r in (1, 2, 3):
            others = [x for x in (1, 2, 3) if x != r]
            loro_cons = [consensus(others, p) for p in pids]
            rater_vals = [rat[t][r][p] for p in pids]
            loro[f"rater{r}_vs_others"] = _sp(rater_vals, loro_cons)

        band = [v for v in loro.values() if v is not None]
        lo, hi = (min(band), max(band)) if band else (None, None)
        if ai_rho is None or not band:
            position = "undefined"
        elif ai_rho < lo:
            position = "BELOW human band (AI worse than human floor)"
        elif ai_rho > hi:
            position = "ABOVE human band (AI better than human ceiling)"
        else:
            position = "INSIDE human band"

        result["targets"][t] = {
            "ai_vs_consensus_rho": ai_rho,
            "loro_rater_vs_others_rho": loro,
            "human_band": [lo, hi],
            "human_band_mean": round(statistics.mean(band), 4) if band else None,
            "ai_position": position,
        }

    (RES / "evaluation" / "loro_spearman.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    print("target            AI-vs-cons   human LORO band          position")
    print("-" * 78)
    for t in TARGETS:
        m = result["targets"][t]
        band = m["human_band"]
        print(f"{t:16s}  {str(m['ai_vs_consensus_rho']):>8s}   "
              f"[{band[0]}, {band[1]}]  {' '*max(0,8-len(str(band[0])+str(band[1])))}{m['ai_position']}")
    print("\n[OK] loro_spearman.json written")


if __name__ == "__main__":
    main()
