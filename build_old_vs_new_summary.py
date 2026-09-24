#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 6 — Consolidated OLD (single-rater) vs NEW (3-rater consensus) comparison.

Pulls together every metric the paper's Section 6.1 / 6.3 and Table 2/3 need:
per-dimension weighted kappa & Spearman, overall Spearman + Holm-corrected p,
ICC, bias, pairwise rank concordance, verdict accuracy, INVEST_WEIGHT peak,
tau_r/tau_c, and the Kickstarter 0%-recall transfer status.

OLD side  = results/Dataset1/evaluation/evaluation_summary.json (single rater)
NEW side  = resultsNew/Dataset1/evaluation/evaluation_summary_consensus.json (consensus)
Calibration OLD/NEW recomputed with the SAME calibration_sweep code (only HUMAN_RAW
differs) so the two columns are strictly comparable.

Outputs:
  - resultsNew/summary_old_vs_new.json
  - resultsNew/summary_old_vs_new.md
"""
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path

import calibration_sweep as cs


def _clean(v):
    """NaN (constant-input Spearman) -> None so JSON stays valid and tables read cleanly."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

ROOT = Path(__file__).resolve().parent
# OLD side uses the pA-CORRECTED single-rater re-run (results/Dataset1 is left as
# the pre-fix buggy baseline; the corrected run lives under resultsNew).
OLD_EVAL = ROOT / "resultsNew" / "Dataset1_singlerater_corrected" / "evaluation" / "evaluation_summary.json"
NEW_EVAL = ROOT / "resultsNew" / "Dataset1" / "evaluation" / "evaluation_summary_consensus.json"
BUGGY_SNAPSHOT = ROOT / "resultsNew" / "pA_fix_buggy_snapshot.json"
CONSENSUS = ROOT / "resultsNew" / "Dataset1" / "consensus_build.json"
LORO = ROOT / "resultsNew" / "Dataset1" / "evaluation" / "loro_spearman.json"
RELI = ROOT / "resultsNew" / "Dataset1" / "rater_reliability.json"
CAL_NEW = ROOT / "resultsNew" / "Dataset1" / "calibration_sweep_consensus.json"
KICK = ROOT / "resultsNew" / "Kickstarter" / "threshold_propagation.json"

DIMS = ["team", "objective", "strategy", "advantages", "feasibility"]
TARGETS = DIMS + ["overall_ranking"]


def holm(pvals):
    """Holm-Bonferroni step-down, same routine as evaluate_kickstarter_correlation."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        running = max(running, adj)
        out[idx] = round(running, 5)
    return out


def calib_summary(human_raw):
    cs.HUMAN_RAW = human_raw
    ai = cs.load_ai_raw()
    results = cs.sweep(ai)
    per_w = {}
    for r in results:
        per_w.setdefault(r["invest_weight"], r["spearman"])
    peak = max(per_w.values())
    peak_ws = [w for w, s in per_w.items() if abs(s - peak) < 1e-9]
    iw90 = per_w.get(0.90)
    best_iw = peak_ws[0]
    with redirect_stdout(io.StringIO()):
        vt = cs.sweep_verdict_thresholds(ai, best_iw)
    return {
        "sweep": {w: per_w[w] for w in sorted(per_w)},
        "peak_spearman": peak, "peak_weights": peak_ws,
        "iw_0.90_spearman": iw90,
        "iw_0.90_ties_peak": iw90 is not None and abs(iw90 - peak) < 1e-9,
        "best_iw": best_iw, "tau_r": vt["t_overall"], "tau_c": vt["t_conf"],
        "verdict_acc": round(vt["verdict_acc"], 4),
    }


def main():
    old = json.loads(OLD_EVAL.read_text())
    new = json.loads(NEW_EVAL.read_text())

    def dim_row(m, col):
        x = m["metrics"][col]
        return {
            "weighted_kappa": _clean(x.get("weighted_kappa")),
            "spearman_r": _clean(x.get("spearman_r")),
            "spearman_p": _clean(x.get("spearman_p")),
            "icc": _clean(x.get("icc")),
            "mean_diff": _clean(x.get("mean_diff")),
            "bias_p": _clean(x.get("bias_p")),
        }

    old_dims = {c: dim_row(old, c) for c in TARGETS}
    new_dims = {c: dim_row(new, c) for c in TARGETS}
    old_holm = holm([old_dims[c]["spearman_p"] if old_dims[c]["spearman_p"] is not None else 1.0 for c in TARGETS])
    new_holm = holm([new_dims[c]["spearman_p"] if new_dims[c]["spearman_p"] is not None else 1.0 for c in TARGETS])
    for i, c in enumerate(TARGETS):
        old_dims[c]["spearman_holm_p"] = old_holm[i]
        new_dims[c]["spearman_holm_p"] = new_holm[i]

    # OLD single-rater HUMAN_RAW = the original hardcoded table in calibration_sweep.
    old_human_raw = {p: dict(v) for p, v in cs.HUMAN_RAW.items()}
    cons = json.loads(CONSENSUS.read_text())["consensus"]
    new_human_raw = {pid: dict(team=c["team"], objective=c["objective"], strategy=c["strategy"],
                               advantages=c["advantages"], feasibility=c["feasibility"],
                               rank=c["overall_20_mean"], verdict=c["verdict_majority"])
                     for pid, c in cons.items()}
    old_cal = calib_summary(old_human_raw)
    new_cal = calib_summary(new_human_raw)

    kick = json.loads(KICK.read_text())
    cal_new_file = json.loads(CAL_NEW.read_text())

    summary = {
        "note": "OLD = single expert rater; NEW = 3-rater consensus (median dims, mean overall/20, majority verdict).",
        "non_unanimous_verdict_proposals_new": json.loads(CONSENSUS.read_text())["non_unanimous_verdict_proposals"],
        "per_dimension": {c: {"OLD": old_dims[c], "NEW": new_dims[c]} for c in TARGETS},
        "verdict_accuracy": {"OLD": old["verdict_accuracy"], "NEW": new["verdict_accuracy"]},
        "pairwise_rank_concordance": {
            "OLD": old["rank_order"]["pairwise_rank_concordance"],
            "NEW": new["rank_order"]["pairwise_rank_concordance"],
        },
        "overall_ranking_icc": {"OLD": old_dims["overall_ranking"]["icc"], "NEW": new_dims["overall_ranking"]["icc"]},
        "calibration": {"OLD": old_cal, "NEW": new_cal},
        "kickstarter_zero_recall": {
            "OLD": {"tau_r": 0.725, "tau_c": 0.811, "max_ai_score": 0.640, "recall": 0.0, "holds": True},
            "NEW": {
                "tau_r_range": [cal_new_file["verdict_accuracy_maximizer"]["smallest_tau_r_among_maximizers"]["tau_r"],
                               cal_new_file["best_threshold"]["tau_r"]],
                "tau_c": cal_new_file["best_threshold"]["tau_c"],
                "max_ai_score": kick["max_ai_overall_score"],
                "recall": 0.0,
                "holds": cal_new_file["zero_recall_finding_holds"],
            },
        },
        "human_human_reliability_NEW_only": json.loads(RELI.read_text()),
        "ai_vs_panel_loro_NEW_only": json.loads(LORO.read_text()),
    }
    (ROOT / "resultsNew" / "summary_old_vs_new.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---------------- markdown ----------------
    L = []
    L.append("# LLM Score Audit — OLD (single-rater) vs NEW (3-rater consensus)\n")
    L.append("_Both columns are computed on the CURRENT AI report cache with the pA-corrected "
             "matcher, so OLD vs NEW is apples-to-apples (only the human ground truth differs)._\n")
    L.append("> ⚠️ **Data-integrity notes (read before citing):**\n"
             ">\n"
             "> 1. **pA matcher bug (fixed):** `evaluate_cohens_kappa._find_record` was resolving proposal A to "
             "`kt4__openai` (a Kickstarter proposal, source `4.pdf` → degenerate 1-char key `4`) on the current, "
             "larger report cache. Fixed by rejecting degenerate keys, preferring exact matches, and warning on "
             "any fuzzy fallback. All 12 proposals now resolve correctly. The OLD numbers here use the corrected "
             "single-rater run on the current cache, **not** the stale `results/Dataset1`.\n"
             "> 2. **Stale-cache drift (NOT fixed — needs your decision):** the checked-in `results/Dataset1` "
             "(the paper's current ρ=0.589 / verdict-acc=0.833) was built on an OLDER `src/data/reports` cache and "
             "is **not reproducible** from the current cache: even before the pA fix, the current cache gives "
             "ρ=0.662 / vacc=0.583. Every proposal's AI score shifted and the AI verdicts changed wholesale "
             "(stale had 8 GO verdicts, current has 3). In the stale cache pA was already matched correctly, so the "
             "pA bug never touched `results/Dataset1`. See `resultsNew/pA_fix_data_drift.md`.\n")
    L.append(f"**Non-unanimous Funded Y/N under consensus (majority-voted):** "
             f"{summary['non_unanimous_verdict_proposals_new']}\n")

    L.append("## 6.1 — Per-dimension AI-vs-human agreement\n")
    L.append("| Dimension | Weighted κ OLD | Weighted κ NEW | Spearman ρ OLD | Spearman ρ NEW | Holm p OLD | Holm p NEW |")
    L.append("|---|---|---|---|---|---|---|")
    for c in TARGETS:
        o, n = old_dims[c], new_dims[c]
        ns = "n/a (constant)" if n["spearman_r"] is None else n["spearman_r"]
        L.append(f"| {c} | {o['weighted_kappa']} | {n['weighted_kappa']} | {o['spearman_r']} | "
                 f"{ns} | {o['spearman_holm_p']} | {n['spearman_holm_p']} |")
    L.append("\n_`objective` NEW ρ is undefined: the median-consensus objective score is 4 for all 12 "
             "proposals (zero variance), so Spearman is not computable — itself a consequence of the "
             "tighter 3-rater agreement on that dimension._")

    L.append("\n### Bias (mean AI−Human) & paired-difference p\n")
    L.append("| Dimension | mean_diff OLD | bias_p OLD | mean_diff NEW | bias_p NEW |")
    L.append("|---|---|---|---|---|")
    for c in TARGETS:
        o, n = old_dims[c], new_dims[c]
        L.append(f"| {c} | {o['mean_diff']} | {o['bias_p']} | {n['mean_diff']} | {n['bias_p']} |")

    L.append("\n### Headline scalars\n")
    L.append("| Metric | OLD | NEW |")
    L.append("|---|---|---|")
    L.append(f"| Overall-ranking Spearman ρ | {old_dims['overall_ranking']['spearman_r']} | {new_dims['overall_ranking']['spearman_r']} |")
    L.append(f"| Overall-ranking Spearman Holm p | {old_dims['overall_ranking']['spearman_holm_p']} | {new_dims['overall_ranking']['spearman_holm_p']} |")
    L.append(f"| Overall-ranking ICC(2,1) | {summary['overall_ranking_icc']['OLD']} | {summary['overall_ranking_icc']['NEW']} |")
    L.append(f"| Pairwise rank concordance | {summary['pairwise_rank_concordance']['OLD']} | {summary['pairwise_rank_concordance']['NEW']} |")
    L.append(f"| Verdict accuracy | {summary['verdict_accuracy']['OLD']} | {summary['verdict_accuracy']['NEW']} |")

    L.append("\n## 6.3 — Calibration (INVEST_WEIGHT & verdict threshold)\n")
    L.append("| Metric | OLD | NEW |")
    L.append("|---|---|---|")
    L.append(f"| INVEST_WEIGHT peak Spearman | {old_cal['peak_spearman']} at IW={old_cal['peak_weights']} | {new_cal['peak_spearman']} at IW={new_cal['peak_weights']} |")
    L.append(f"| IW=0.90 Spearman | {old_cal['iw_0.90_spearman']} | {new_cal['iw_0.90_spearman']} |")
    L.append(f"| Production IW=0.90 ties the peak? | {old_cal['iw_0.90_ties_peak']} | {new_cal['iw_0.90_ties_peak']} |")
    L.append(f"| tau_r (max verdict acc) | {old_cal['tau_r']} | {new_cal['tau_r']} |")
    L.append(f"| tau_c (max verdict acc) | {old_cal['tau_c']} | {new_cal['tau_c']} |")
    L.append(f"| verdict accuracy at that threshold | {old_cal['verdict_acc']} | {new_cal['verdict_acc']} |")

    L.append("\n### INVEST_WEIGHT sweep (Spearman ρ at each step)\n")
    L.append("| INVEST_WEIGHT | ρ OLD | ρ NEW |")
    L.append("|---|---|---|")
    for w in sorted(new_cal["sweep"]):
        L.append(f"| {w:.2f} | {old_cal['sweep'].get(w)} | {new_cal['sweep'][w]} |")

    L.append("\n## Kickstarter transfer — the 0%-recall finding\n")
    kn = summary["kickstarter_zero_recall"]["NEW"]
    L.append("| | OLD | NEW |")
    L.append("|---|---|---|")
    L.append(f"| tau_r | 0.725 | {kn['tau_r_range'][0]}–{kn['tau_r_range'][1]} |")
    L.append(f"| tau_c | 0.811 | {kn['tau_c']} |")
    L.append(f"| Max AI score in 141-proposal set | 0.640 | {kn['max_ai_score']} |")
    L.append(f"| tau_r above 0.640? | Yes | {'Yes' if kn['holds'] else 'No'} |")
    L.append(f"| Recall on real successes | 0% | 0% |")
    L.append(f"| **'Threshold transfers with 0% recall' finding** | holds | **{'HOLDS (unchanged)' if kn['holds'] else 'NEEDS REWRITE'}** |")
    L.append("")
    L.append(f"> {cal_new_file['transfer_status']}")

    L.append("\n## NEW — Human–human reliability (did not exist before)\n")
    reli = summary["human_human_reliability_NEW_only"]
    L.append("| Target | ICC(2,1) [95% CI] | ICC(2,k) [95% CI] | mean pairwise wκ | Gwet AC1 |")
    L.append("|---|---|---|---|---|")
    for t in TARGETS:
        icc = reli["icc"][t]
        wk = reli["pairwise_weighted_kappa"].get(t, {}).get("mean") if t in DIMS else "—"
        ac1 = reli["gwet_ac1"].get(t, {}).get("ac1") if t in DIMS else "—"
        L.append(f"| {t} | {icc['icc_2_1']} {icc['icc_2_1_ci95']} | {icc['icc_2_k']} {icc['icc_2_k_ci95']} | {wk} | {ac1} |")

    L.append("\n## NEW — AI-vs-panel vs the human leave-one-rater-out band\n")
    loro = summary["ai_vs_panel_loro_NEW_only"]["targets"]
    L.append("| Target | AI-vs-consensus ρ | Human LORO band | AI position |")
    L.append("|---|---|---|---|")
    for t in TARGETS:
        m = loro[t]
        L.append(f"| {t} | {m['ai_vs_consensus_rho']} | {m['human_band']} | {m['ai_position']} |")
    L.append("\n_LORO ρ is computed on unrounded mean/median consensus, so the overall-ranking value "
             "here differs slightly from the pipeline's (evaluate_cohens_kappa.py rounds "
             "the mean rank to an integer before ranking). Both are well below the human band._")

    # ---------------- pA-fix delta section ----------------
    if BUGGY_SNAPSHOT.exists():
        buggy = json.loads(BUGGY_SNAPSHOT.read_text())
        L.append("\n## pA-fix impact — what changed vs the previous (buggy) run\n")
        L.append("_Before the fix, proposal A was scored against `kt4__openai` (a Kickstarter proposal) "
                 "instead of `pA__openai`. Below, every metric whose value moved once pA was corrected, "
                 "for BOTH the single-rater (OLD) and consensus (NEW) evaluations. Verdict/calibration/"
                 "Kickstarter numbers are unaffected (pA verdict = N either way; calibration reads the "
                 "explicit expert_reports path)._\n")

        def deltas(side_label, buggy_side, fixed_metrics, fixed_vacc, fixed_conc):
            rows = []
            for c in TARGETS:
                b = buggy_side["metrics"][c]
                f = fixed_metrics[c]
                for key in ("weighted_kappa", "spearman_r", "icc", "mean_diff", "bias_p"):
                    bv, fv = b.get(key), f.get(key)
                    if bv != fv and not (bv is None and fv is None):
                        try:
                            d = f"{(fv - bv):+.4f}" if (bv is not None and fv is not None) else "—"
                        except TypeError:
                            d = "—"
                        rows.append((f"{c}.{key}", bv, fv, d))
            for label, bv, fv in (("verdict_accuracy", buggy_side["verdict_accuracy"], fixed_vacc),
                                  ("rank_concordance", buggy_side["rank_concordance"], fixed_conc)):
                if bv != fv:
                    d = f"{(fv - bv):+.4f}" if (isinstance(bv,(int,float)) and isinstance(fv,(int,float))) else "—"
                    rows.append((label, bv, fv, d))
            L.append(f"\n### {side_label}\n")
            if not rows:
                L.append("_No metric changed._")
                return
            L.append("| Metric | buggy (kt4) | fixed (pA) | Δ |")
            L.append("|---|---|---|---|")
            for name, bv, fv, d in rows:
                L.append(f"| {name} | {bv} | {fv} | {d} |")

        deltas("OLD single-rater", buggy["OLD_singlerater_buggy"],
               old_dims, old["verdict_accuracy"], old["rank_order"]["pairwise_rank_concordance"])
        deltas("NEW 3-rater consensus", buggy["NEW_consensus_buggy"],
               new_dims, new["verdict_accuracy"], new["rank_order"]["pairwise_rank_concordance"])

        # LORO delta
        lb = buggy["LORO_buggy"]["targets"]
        lf = summary["ai_vs_panel_loro_NEW_only"]["targets"]
        loro_rows = [(t, lb[t]["ai_vs_consensus_rho"], lf[t]["ai_vs_consensus_rho"])
                     for t in TARGETS if lb[t]["ai_vs_consensus_rho"] != lf[t]["ai_vs_consensus_rho"]]
        L.append("\n### LORO AI-vs-consensus ρ (NEW)\n")
        if loro_rows:
            L.append("| Target | buggy | fixed | Δ | AI still below human band? |")
            L.append("|---|---|---|---|---|")
            for t, bv, fv in loro_rows:
                d = f"{(fv - bv):+.4f}" if (isinstance(bv,(int,float)) and isinstance(fv,(int,float))) else "—"
                below = "yes" if lf[t]["ai_position"].startswith("BELOW") else lf[t]["ai_position"]
                L.append(f"| {t} | {bv} | {fv} | {d} | {below} |")
        else:
            L.append("_No LORO value changed._")

    (ROOT / "resultsNew" / "summary_old_vs_new.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[OK] resultsNew/summary_old_vs_new.json + summary_old_vs_new.md written")
    print("\n".join(L))


if __name__ == "__main__":
    main()
