#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical AI score set for Dataset1 (12 biotech/semiconductor proposals).

Source of truth = the FROZEN Jun-16 evaluation in results/Dataset1, NOT the stale
Jun-18 cache in src/data/reports (which drifted via LLM non-determinism). Two
independent frozen sources are cross-checked:

  A. results/Dataset1/evaluation/evaluation_report.xlsx  (per_item sheet)
       -> final blended AI dimension + overall scores (0-1, at INVEST_WEIGHT=0.90)
  B. results/Dataset1/rerun_stage6_7_iw90.log
       -> per-dimension invest_score_echo / qa_score_echo (for calibration sweeps)

Confidence comes from src/data/refined_answers/<pid>/postproc/metrics.json, whose
mtimes are Jun-16 (Stage-5 output was never regenerated, so confidence did NOT
drift). Canonical AI verdict = the published OR-logic rule recorded in
evaluation_summary.json: GO iff overall >= 0.725 OR confidence >= 0.811. (The
per_item `verdict_ai` column holds stale raw report verdicts and is NOT used.)

Writes resultsNew/Dataset1/canonical_ai_scores.json and asserts the canonical set
reproduces every published number in evaluation_summary.json before anything
downstream is allowed to use it.
"""
import json
import re
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.tools.evaluate_cohens_kappa import _column_metrics, _rank_order_metrics  # noqa: E402

PIDS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]
DIMS = ["team", "objective", "strategy", "advantages", "feasibility"]
H2L = {"team": "team", "objective": "objectives", "strategy": "strategy",
       "advantages": "innovation", "feasibility": "feasibility"}
RANK = "overall_ranking"

EVAL_XLSX = ROOT / "results" / "Dataset1" / "evaluation" / "evaluation_report.xlsx"
EVAL_SUMMARY = ROOT / "results" / "Dataset1" / "evaluation" / "evaluation_summary.json"
IW90_LOG = ROOT / "results" / "Dataset1" / "rerun_stage6_7_iw90.log"
METRICS = lambda pid: ROOT / "src" / "data" / "refined_answers" / f"{pid}__openai" / "postproc" / "metrics.json"
OUT = ROOT / "resultsNew" / "Dataset1" / "canonical_ai_scores.json"

TAU_R, TAU_C = 0.725, 0.811


def load_peritem():
    ws = openpyxl.load_workbook(EVAL_XLSX, data_only=True)["per_item"]
    rows = list(ws.iter_rows(values_only=True))
    h = list(rows[0]); c = lambda n: h.index(n)
    out = {}
    for pid, r in zip(PIDS, rows[1:]):
        out[pid] = {
            "file_name": r[c("file_name")],
            "pid_matched": r[c("pid")],
            **{d: r[c(f"ai_{d}")] for d in DIMS},
            RANK: r[c(f"ai_{RANK}")],
            "human_dims": {d: r[c(f"human_{d}")] for d in DIMS},
            "human_overall": r[c(f"human_{RANK}")],
            "verdict_human": r[c("verdict_human")],
            "verdict_ai_stale": r[c("verdict_ai")],
        }
    return out


def load_log_invest_qa():
    log = IW90_LOG.read_text()
    blocks = re.split(r"\[STAGE6\+7\] Starting pid=", log)
    out = {}
    for b in blocks[1:]:
        pid = b.split("\n", 1)[0].strip()
        if pid not in PIDS:
            continue
        invest, qa = {}, {}
        m = re.search(r"\[INVEST\] investment scores: (\{[^\n]+\})", b)
        if m:
            invest = json.loads(m.group(1).replace("'", '"'))
        for mm in re.finditer(r"\[DIM_OUTPUT\] (\w+): .*score=([0-9.]+) \(qa=([0-9.]+) invest=([0-9.]+)\)", b):
            qa[mm.group(1)] = float(mm.group(3))
        out[pid] = {"invest": invest, "qa": qa}
    return out


def confidence(pid):
    m = json.loads(METRICS(pid).read_text())
    return round(float((m.get("overall") or {}).get("overall_confidence", 0) or 0), 4)


def dim_weights(pid):
    m = json.loads(METRICS(pid).read_text())
    return (m.get("config_used") or {}).get("dimension_weight") or {}


def main():
    peritem = load_peritem()
    logiq = load_log_invest_qa()

    # ---- cross-check A vs B: per_item dim == blended(invest,qa)@0.90 ----
    IW = 0.90
    maxdiff = 0.0
    for pid in PIDS:
        for d in DIMS:
            ld = H2L[d]
            blended = IW * logiq[pid]["invest"][ld] + (1 - IW) * logiq[pid]["qa"][ld]
            maxdiff = max(maxdiff, abs(peritem[pid][d] - blended))
    assert maxdiff <= 0.006, f"SOURCE A/B disagree (max {maxdiff}) — STOP"

    # ---- canonical record per proposal ----
    canonical = {}
    for pid in PIDS:
        conf = confidence(pid)
        ov = peritem[pid][RANK]
        canonical[pid] = {
            "pid": pid,
            "file_name": peritem[pid]["file_name"],
            **{d: peritem[pid][d] for d in DIMS},
            RANK: ov,
            "confidence": conf,
            "verdict": "Y" if (ov >= TAU_R or conf >= TAU_C) else "N",
            "invest_echo": {H2L[d]: logiq[pid]["invest"][H2L[d]] for d in DIMS},
            "qa_echo": {H2L[d]: logiq[pid]["qa"][H2L[d]] for d in DIMS},
            "dim_weights": dim_weights(pid),
        }

    # ---- VALIDATION: reproduce every published number from evaluation_summary ----
    summ = json.loads(EVAL_SUMMARY.read_text())
    # per-dimension metrics from the frozen per_item human/ai pairs
    per_item_rows = []
    pairs = {c: [] for c in DIMS + [RANK]}
    for pid in PIDS:
        row = {"file_name": peritem[pid]["file_name"], "pid": pid}
        for d in DIMS:
            hv, av = peritem[pid]["human_dims"][d], peritem[pid][d]
            row[f"human_{d}"], row[f"ai_{d}"] = hv, av
            if hv not in (None, "") and av not in (None, ""):
                pairs[d].append((float(hv), float(av)))
        hv, av = peritem[pid]["human_overall"], peritem[pid][RANK]
        row[f"human_{RANK}"], row[f"ai_{RANK}"] = hv, av
        if hv not in (None, "") and av not in (None, ""):
            pairs[RANK].append((float(hv), float(av)))
        per_item_rows.append(row)

    mismatches = []
    for d in DIMS:
        got = _column_metrics(pairs[d], is_ranking=False)
        for k in ("weighted_kappa", "spearman_r"):
            if abs((got.get(k) or 0) - (summ["metrics"][d].get(k) or 0)) > 1e-4:
                mismatches.append(f"{d}.{k}: got {got.get(k)} vs published {summ['metrics'][d].get(k)}")
    got_r = _column_metrics(pairs[RANK], is_ranking=True)
    for k in ("icc", "spearman_r"):
        if abs((got_r.get(k) or 0) - (summ["metrics"][RANK].get(k) or 0)) > 1e-4:
            mismatches.append(f"{RANK}.{k}: got {got_r.get(k)} vs published {summ['metrics'][RANK].get(k)}")
    # rank concordance
    ro = _rank_order_metrics(per_item_rows, RANK)
    if abs((ro.get("pairwise_rank_concordance") or 0) - (summ["rank_order"]["pairwise_rank_concordance"] or 0)) > 1e-4:
        mismatches.append(f"rank_concordance: got {ro.get('pairwise_rank_concordance')} vs {summ['rank_order']['pairwise_rank_concordance']}")
    # OR-logic verdict accuracy
    vacc = sum(1 for pid in PIDS if canonical[pid]["verdict"] == peritem[pid]["verdict_human"]) / 12
    if abs(vacc - summ["verdict_accuracy"]) > 1e-4:
        mismatches.append(f"verdict_accuracy(OR-logic): got {round(vacc,4)} vs published {summ['verdict_accuracy']}")

    if mismatches:
        print("VALIDATION FAILED — canonical set does NOT reproduce published numbers:")
        for m in mismatches:
            print("  -", m)
        raise SystemExit("STOP: canonical extraction does not match evaluation_summary.json")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "provenance": {
            "scores": "results/Dataset1/evaluation/evaluation_report.xlsx (per_item) == rerun_stage6_7_iw90.log blended@0.90",
            "confidence": "src/data/refined_answers/<pid>/postproc/metrics.json (Jun-16, never regenerated)",
            "verdict_rule": "OR-logic score>=0.725 OR confidence>=0.811 (published in evaluation_summary.json)",
            "invest_qa_echo": "rerun_stage6_7_iw90.log (Jun-16)",
            "cross_check_max_source_diff": round(maxdiff, 4),
            "note": "src/data/reports Jun-18 cache is stale/non-canonical and is NOT used.",
        },
        "canonical": canonical,
    }, indent=2), encoding="utf-8")

    print(f"[OK] canonical_ai_scores.json written ({OUT})")
    print(f"cross-check A/B max dim diff = {maxdiff:.4f}")
    print(f"VALIDATION PASSED ✅ — reproduces published rho={summ['metrics'][RANK]['spearman_r']}, "
          f"verdict_acc={summ['verdict_accuracy']}, and all per-dim kappa/spearman/icc.")
    print("pA canonical overall =", canonical["pA"][RANK], "(correct pA__openai, not kt4 0.571)")


if __name__ == "__main__":
    main()
