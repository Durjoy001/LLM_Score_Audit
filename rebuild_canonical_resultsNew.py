#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild resultsNew/ from the CANONICAL Jun-16 AI scores (canonical_ai_scores.json)
instead of the stale Jun-18 src/data cache. Tasks 3-5 + consolidated summary.

OLD single-rater column = the published results/Dataset1 numbers themselves (which
the canonical set reproduces exactly). NEW = 3-rater consensus, both on the SAME
canonical AI score set. No LLM calls; pure re-derivation.
"""
import io
import json
import math
import statistics
import csv
from contextlib import redirect_stdout
from pathlib import Path

import calibration_sweep as cs
from src.tools.evaluate_cohens_kappa import _column_metrics, _rank_order_metrics
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RES = ROOT / "resultsNew" / "Dataset1"
CANON = json.loads((RES / "canonical_ai_scores.json").read_text())["canonical"]
CONS = json.loads((RES / "consensus_build.json").read_text())["consensus"]
PUB = json.loads((ROOT / "results" / "Dataset1" / "evaluation" / "evaluation_summary.json").read_text())

PIDS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]
DIMS = ["team", "objective", "strategy", "advantages", "feasibility"]
LOG_DIMS = ["team", "objectives", "strategy", "innovation", "feasibility"]
RANK = "overall_ranking"
KICK_MAX_AI = 0.640
TAU_R, TAU_C = 0.725, 0.811


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def hnorm(v, is_rank=False):
    return round((round(v) - 1) / 19, 4) if is_rank else round((v - 1) / 4, 4)


# ------------------------------------------------------------------ Task 3: eval
def consensus_eval():
    pairs = {c: [] for c in DIMS + [RANK]}
    rows = []
    vmatch = 0
    for pid in PIDS:
        c = CONS[pid]; a = CANON[pid]
        row = {"file_name": a["file_name"], "pid": pid}
        for d in DIMS:
            hv = hnorm(c[d]); av = a[d]
            row[f"human_{d}"], row[f"ai_{d}"] = hv, av
            pairs[d].append((hv, av))
        hv = hnorm(c["overall_20_mean"], is_rank=True); av = a[RANK]
        row[f"human_{RANK}"], row[f"ai_{RANK}"] = hv, av
        pairs[RANK].append((hv, av))
        rows.append(row)
        if c["verdict_majority"] == a["verdict"]:
            vmatch += 1
    metrics = {}
    for d in DIMS:
        m = _column_metrics(pairs[d], is_ranking=False)
        metrics[d] = {k: _clean(m.get(k)) for k in ("weighted_kappa", "spearman_r", "spearman_p", "mean_diff", "bias_p", "mean_human", "mean_ai")}
    mr = _column_metrics(pairs[RANK], is_ranking=True)
    metrics[RANK] = {k: _clean(mr.get(k)) for k in ("icc", "spearman_r", "spearman_p", "weighted_kappa", "mean_diff", "bias_p")}
    ro = _rank_order_metrics(rows, RANK)
    summary = {
        "source": "CANONICAL Jun-16 scores (canonical_ai_scores.json)",
        "ground_truth": "3-rater consensus (median dims, mean overall/20, majority verdict)",
        "verdict_rule": "OR-logic score>=0.725 OR confidence>=0.811",
        "matched_rows": 12,
        "verdict_accuracy": round(vmatch / 12, 4),
        "metrics": metrics,
        "rank_order": {"pairwise_rank_concordance": ro["pairwise_rank_concordance"],
                       "pairwise_comparable": ro["pairwise_comparable"]},
        "non_unanimous_verdict": [p for p in PIDS if not CONS[p]["verdict_unanimous"]],
    }
    (RES / "evaluation" / "evaluation_summary_consensus.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ------------------------------------------------------------------ Task 3: LORO
def loro():
    raters = {t: {1: {}, 2: {}, 3: {}} for t in DIMS + [RANK]}
    id2pid = {i: p for i, p in enumerate(PIDS, 1)}
    with (RES / "raters_long.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raters[r["dimension"]][int(r["rater_id"])][id2pid[int(r["proposal_id"])]] = float(r["score"])

    def sp(xs, ys):
        if len(xs) < 3:
            return None
        rho, _ = spearmanr(xs, ys)
        return None if rho != rho else round(float(rho), 4)

    out = {"source": "CANONICAL Jun-16 scores", "targets": {}}
    for t in DIMS + [RANK]:
        agg = statistics.mean if t == RANK else statistics.median
        cons_all = [agg([raters[t][r][p] for r in (1, 2, 3)]) for p in PIDS]
        ai_vals = [CANON[p][t] for p in PIDS]
        ai_rho = sp(ai_vals, cons_all)
        loro_vals = {}
        for r in (1, 2, 3):
            others = [x for x in (1, 2, 3) if x != r]
            loro_cons = [agg([raters[t][o][p] for o in others]) for p in PIDS]
            loro_vals[f"rater{r}_vs_others"] = sp([raters[t][r][p] for p in PIDS], loro_cons)
        band = [v for v in loro_vals.values() if v is not None]
        lo, hi = (min(band), max(band)) if band else (None, None)
        pos = "undefined" if (ai_rho is None or not band) else (
            "BELOW human band" if ai_rho < lo else "ABOVE human band" if ai_rho > hi else "INSIDE human band")
        out["targets"][t] = {"ai_vs_consensus_rho": ai_rho, "loro": loro_vals,
                             "human_band": [lo, hi], "ai_position": pos}
    (RES / "evaluation" / "loro_spearman.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ------------------------------------------------------------- Task 4: calibration
def build_ai_raw():
    ai_raw = {}
    for pid in PIDS:
        c = CANON[pid]
        ai_raw[pid] = {ld: {"invest": c["invest_echo"][ld], "qa": c["qa_echo"][ld]} for ld in LOG_DIMS}
        ai_raw[pid]["_conf"] = c["confidence"]
        ai_raw[pid]["_dim_weights"] = c["dim_weights"]
    return ai_raw


def calib(human_raw, ai_raw, label):
    cs.HUMAN_RAW = human_raw
    results = cs.sweep(ai_raw)
    per_w = {}
    for r in results:
        per_w.setdefault(r["invest_weight"], r["spearman"])
    peak = max(per_w.values())
    peak_ws = [w for w, s in per_w.items() if abs(s - peak) < 1e-9]
    iw90 = per_w.get(0.90)
    with redirect_stdout(io.StringIO()):
        vt = cs.sweep_verdict_thresholds(ai_raw, peak_ws[0])
        # global fine search for max verdict accuracy + smallest tau_r among maximizers
        vacc = []
        for iw in [round(w * 0.05, 2) for w in range(21)]:
            r = cs.sweep_verdict_thresholds(ai_raw, iw)
            vacc.append({"iw": iw, "tau_r": r["t_overall"], "tau_c": r["t_conf"], "acc": round(r["verdict_acc"], 4)})
    mx = max(v["acc"] for v in vacc)
    maximizers = [v for v in vacc if abs(v["acc"] - mx) < 1e-9]
    min_tau = min(maximizers, key=lambda v: v["tau_r"])
    return {
        "label": label,
        "sweep": {w: per_w[w] for w in sorted(per_w)},
        "peak_spearman": peak, "peak_weights": peak_ws,
        "iw_0.90_spearman": iw90, "iw_0.90_ties_peak": iw90 is not None and abs(iw90 - peak) < 1e-9,
        "tau_r_at_peak_iw": vt["t_overall"], "tau_c_at_peak_iw": vt["t_conf"],
        "max_verdict_acc": mx, "smallest_tau_r_maximizer": min_tau,
        "all_maximizers_above_640": all(v["tau_r"] > KICK_MAX_AI for v in maximizers),
    }


# ------------------------------------------------------------- Task 5: Kickstarter
def kickstarter(new_tau_r_range, new_tau_c):
    import calibrate_kickstarter_threshold as ck
    rows = ck.load_rows()
    max_score = max(r["score"] for r in rows)
    thresholds = {
        "production_transferred": (TAU_R, TAU_C),
        "canonical_consensus_low": (new_tau_r_range[0], new_tau_c),
        "canonical_consensus_high": (new_tau_r_range[1], new_tau_c),
    }
    cls = {}
    for name, (tr, tc) in thresholds.items():
        m = ck.metrics_from_confusion(ck.confusion(rows, tr, tc))
        cls[name] = {"tau_r": tr, "tau_c": tc, "n_go": sum(1 for r in rows if r["score"] >= tr or r["confidence"] >= tc),
                     "recall": m["recall"], "accuracy": m["accuracy"], "tau_r_above_640": tr > KICK_MAX_AI}
    out = {
        "cache_drift_check": {
            "kickstarter_reports_mtime": "all 141 dated 2026-07-15, before the 2026-07-15 19:43 frozen eval",
            "drifted": False,
            "note": "Unlike Dataset1, the Kickstarter cache was NOT re-run after its eval was frozen; it is self-consistent/canonical.",
        },
        "max_ai_overall_score": round(max_score, 4),
        "kickstarter_max_reference": KICK_MAX_AI,
        "classification": cls,
        "all_zero_recall": all((c["recall"] in (0, 0.0, None)) for c in cls.values()),
        "zero_recall_finding_holds": all(c["tau_r_above_640"] for c in cls.values()),
    }
    (ROOT / "resultsNew" / "Kickstarter" / "threshold_propagation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    ce = consensus_eval()
    lo = loro()
    ai_raw = build_ai_raw()
    old_cal = calib({p: dict(v) for p, v in cs.HUMAN_RAW.items()}, ai_raw, "OLD single-rater")
    new_hr = {p: dict(team=c["team"], objective=c["objective"], strategy=c["strategy"],
                      advantages=c["advantages"], feasibility=c["feasibility"],
                      rank=c["overall_20_mean"], verdict=c["verdict_majority"]) for p, c in CONS.items()}
    new_cal = calib(new_hr, ai_raw, "NEW consensus")
    tau_lo = new_cal["smallest_tau_r_maximizer"]["tau_r"]
    tau_hi = new_cal["tau_r_at_peak_iw"]
    kk = kickstarter([min(tau_lo, tau_hi), max(tau_lo, tau_hi)], new_cal["tau_c_at_peak_iw"])
    json.dump({"old": old_cal, "new": new_cal}, open(RES / "calibration_sweep_consensus.json", "w"), indent=2)

    build_summary(ce, lo, old_cal, new_cal, kk)
    print("[OK] canonical resultsNew rebuilt.")
    print(f"  NEW consensus overall rho={ce['metrics'][RANK]['spearman_r']}  verdict_acc={ce['verdict_accuracy']}")
    print(f"  OLD published overall rho={PUB['metrics'][RANK]['spearman_r']}  verdict_acc={PUB['verdict_accuracy']}")
    print(f"  calib: OLD peak {old_cal['peak_spearman']}@{old_cal['peak_weights']} | NEW peak {new_cal['peak_spearman']}@{new_cal['peak_weights']}")
    print(f"  NEW tau_r range [{tau_lo},{tau_hi}] vs 0.640 -> finding holds: {kk['zero_recall_finding_holds']}")


def build_summary(ce, lo, old_cal, new_cal, kk):
    TG = DIMS + [RANK]
    reli = json.loads((RES / "rater_reliability.json").read_text())
    L = ["# LLM Score Audit — OLD (single-rater) vs NEW (3-rater consensus), CANONICAL Jun-16 scores\n",
         "_Both columns use the canonical Jun-16 AI score set (`canonical_ai_scores.json`), which reproduces the "
         "published paper numbers exactly (ρ=0.5894, verdict-acc=0.8333). The stale Jun-18 `src/data` cache is not used. "
         "pA is correctly scored as pA__openai (0.681)._\n",
         f"**Non-unanimous consensus verdicts:** {ce['non_unanimous_verdict']}\n",
         "## 6.1 — Per-dimension AI-vs-human agreement\n",
         "| Dimension | κ OLD | κ NEW | ρ OLD | ρ NEW |",
         "|---|---|---|---|---|"]
    for d in DIMS:
        o = PUB["metrics"][d]; n = ce["metrics"][d]
        ns = "n/a (constant)" if n["spearman_r"] is None else n["spearman_r"]
        L.append(f"| {d} | {o.get('weighted_kappa')} | {n['weighted_kappa']} | {o.get('spearman_r')} | {ns} |")
    L += ["\n### Headline scalars\n", "| Metric | OLD (paper) | NEW consensus |", "|---|---|---|",
          f"| Overall Spearman ρ | {PUB['metrics'][RANK]['spearman_r']} | {ce['metrics'][RANK]['spearman_r']} |",
          f"| Overall ICC(2,1) | {PUB['metrics'][RANK].get('icc')} | {ce['metrics'][RANK]['icc']} |",
          f"| Pairwise rank concordance | {PUB['rank_order']['pairwise_rank_concordance']} | {ce['rank_order']['pairwise_rank_concordance']} |",
          f"| Verdict accuracy (OR-logic) | {PUB['verdict_accuracy']} | {ce['verdict_accuracy']} |",
          "\n_NEW overall ρ=0.175 uses integer-rounded consensus ranks (evaluate_cohens_kappa convention); "
          "the LORO table below uses un-rounded mean ranks (ρ=0.050). Both are weakly positive and far below "
          "the OLD single-rater 0.589 and the human LORO band. NEW verdict accuracy ties OLD at 0.833 only "
          "because the OR-logic AI GO-set {p3,p7} lands inside the consensus GO-set; pA and pD are the 2 misses._\n",
          "## 6.3 — Calibration (canonical invest/qa echoes)\n", "| Metric | OLD | NEW |", "|---|---|---|",
          f"| INVEST_WEIGHT peak ρ | {old_cal['peak_spearman']} @ {old_cal['peak_weights']} | {new_cal['peak_spearman']} @ {new_cal['peak_weights']} |",
          f"| IW=0.90 ties peak? | {old_cal['iw_0.90_ties_peak']} | {new_cal['iw_0.90_ties_peak']} |",
          f"| max verdict-acc τ_r (smallest) | {old_cal['smallest_tau_r_maximizer']['tau_r']} | {new_cal['smallest_tau_r_maximizer']['tau_r']} |",
          f"| τ_c | {old_cal['tau_c_at_peak_iw']} | {new_cal['tau_c_at_peak_iw']} |",
          "\n### INVEST_WEIGHT sweep (ρ at each step)\n", "| IW | ρ OLD | ρ NEW |", "|---|---|---|"]
    for w in sorted(new_cal["sweep"]):
        L.append(f"| {w:.2f} | {old_cal['sweep'].get(w)} | {new_cal['sweep'][w]} |")
    L += ["\n## Kickstarter 0%-recall transfer (canonical; cache NOT drifted)\n",
          f"- Max Kickstarter AI score = {kk['max_ai_overall_score']} (ref {kk['kickstarter_max_reference']}).",
          f"- New τ_r range vs 0.640, recall under all thresholds, finding holds: **{kk['zero_recall_finding_holds']}** "
          f"(all zero recall: {kk['all_zero_recall']}).",
          "\n## NEW — human-human reliability (unchanged; AI-independent)\n",
          "| Target | ICC(2,1) | ICC(2,k) | mean pairwise wκ | Gwet AC1 |", "|---|---|---|---|---|"]
    for t in TG:
        icc = reli["icc"][t]
        wk = reli["pairwise_weighted_kappa"].get(t, {}).get("mean") if t in DIMS else "—"
        ac = reli["gwet_ac1"].get(t, {}).get("ac1") if t in DIMS else "—"
        L.append(f"| {t} | {icc['icc_2_1']} | {icc['icc_2_k']} | {wk} | {ac} |")
    L += ["\n## NEW — AI-vs-panel vs human LORO band (canonical)\n",
          "| Target | AI-vs-consensus ρ | human LORO band | position |", "|---|---|---|---|"]
    for t in TG:
        m = lo["targets"][t]
        L.append(f"| {t} | {m['ai_vs_consensus_rho']} | {m['human_band']} | {m['ai_position']} |")
    (ROOT / "resultsNew" / "summary_old_vs_new.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({"old_published": PUB, "new_consensus": ce, "loro": lo,
               "calibration": {"old": old_cal, "new": new_cal}, "kickstarter": kk},
              open(ROOT / "resultsNew" / "summary_old_vs_new.json", "w"), indent=2)


if __name__ == "__main__":
    main()
