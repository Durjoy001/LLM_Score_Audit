#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the self-contained data package for the paper section on
score validity vs. threshold validity (Kickstarter, n=141).

Contains ONLY well-powered results. Deliberately excludes every n=12 analysis
(Dataset1 verdict-rule comparison, the Dataset1 score/threshold inversion, the
cap ablation whose OpenAI cohort is n=1) and the cross-provider Gemini arms.

Zero LLM calls. Reads the frozen canonical Kickstarter scores only.
Output: canonical/component4_kickstarter.json
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "canonical" / "canonical_kickstarter_scores.json"
OUT = ROOT / "canonical" / "component4_kickstarter.json"

SEED = 42
N_BOOT = 10000
TAU_R, TAU_C = 0.725, 0.811

KSJ = json.loads(SRC.read_text())
KS = KSJ["scores"]
PIDS = sorted(KS)
score = [KS[p]["overall_ranking"] for p in PIDS]
conf = [KS[p]["confidence"] for p in PIDS]
funded_pct = [float(KS[p]["funded_pct"]) for p in PIDS]
y = [1 if v >= 100 else 0 for v in funded_pct]
N, NPOS = len(PIDS), sum(y)


def auc(sc, lab):
    P = [a for a, b in zip(sc, lab) if b == 1]
    Ng = [a for a, b in zip(sc, lab) if b == 0]
    if not P or not Ng:
        return None
    return sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in P for q in Ng) / (len(P) * len(Ng))


def boot_ci():
    rng = random.Random(SEED)
    bA, bR = [], []
    for _ in range(N_BOOT):
        idx = [rng.randrange(N) for _ in range(N)]
        ss = [score[i] for i in idx]
        yy = [y[i] for i in idx]
        pp = [funded_pct[i] for i in idx]
        if 0 < sum(yy) < len(yy):
            a = auc(ss, yy)
            if a is not None:
                bA.append(a)
        r = spearmanr(ss, pp).statistic
        if r == r:
            bR.append(r)
    q = lambda v, p: round(sorted(v)[int(p * len(v))], 4)
    return (q(bA, .025), q(bA, .975), len(bA)), (q(bR, .025), q(bR, .975), len(bR))


RULES = {
    "score_only": lambda s, c: s >= TAU_R,
    "confidence_only": lambda s, c: c >= TAU_C,
    "score_AND_confidence": lambda s, c: (s >= TAU_R) and (c >= TAU_C),
    "score_OR_confidence": lambda s, c: (s >= TAU_R) or (c >= TAU_C),
}


def confusion(pred):
    tp = sum(1 for p, a in zip(pred, y) if p == 1 and a == 1)
    fp = sum(1 for p, a in zip(pred, y) if p == 1 and a == 0)
    fn = sum(1 for p, a in zip(pred, y) if p == 0 and a == 1)
    tn = sum(1 for p, a in zip(pred, y) if p == 0 and a == 0)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if prec is not None else None)
    return {"accuracy": round((tp + tn) / N, 4),
            "precision": None if prec is None else round(prec, 4),
            "recall": None if rec is None else round(rec, 4),
            "f1": None if f1 is None else round(f1, 4),
            "n_predicted_go": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    A = round(auc(score, y), 4)
    R = round(float(spearmanr(score, funded_pct).statistic), 4)
    (aLo, aHi, aN), (rLo, rHi, rN) = boot_ci()
    u = mannwhitneyu([s for s, l in zip(score, y) if l == 1],
                     [s for s, l in zip(score, y) if l == 0], alternative="greater")

    order = sorted(range(N), key=lambda i: -score[i])
    pak = {}
    for k in (10, 15, 20, 25, 30, 40, 50):
        hit = sum(y[i] for i in order[:k])
        pak[k] = {"funded_in_top_k": hit, "precision_at_k": round(hit / k, 4),
                  "lift_vs_base_rate": round((hit / k) / (NPOS / N), 3),
                  "recall_at_k": round(hit / NPOS, 4)}

    sweep = []
    for t in sorted(set(score)):
        c = confusion([1 if x >= t else 0 for x in score])
        tpr = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0
        fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else 0
        sweep.append({"tau": round(t, 4), "youden_j": round(tpr - fpr, 4), **c})
    best_f1 = max(sweep, key=lambda x: x["f1"] or 0)
    best_j = max(sweep, key=lambda x: x["youden_j"])

    payload = {
        "title": "Score validity vs. threshold validity — Kickstarter (n=141)",
        "one_line_finding": (
            "The AI score ranks real crowdfunding outcomes at AUC 0.764 (95% CI 0.667-0.853), "
            "yet at the production thresholds it predicts zero GO decisions, yielding an "
            "accuracy of 0.8298 that is numerically identical to the negative base rate and a "
            "recall of 0.000."),
        "provenance": {
            "scores": "canonical/canonical_kickstarter_scores.json (frozen 2026-07-15 cache; "
                      "verified consistent with results/Kickstarter, no post-eval re-run, no drift)",
            "ground_truth": "funded_pct >= 100 (real crowdfunding outcome)",
            "scorer": "gpt-4o-mini, Stage 6 Call B, temperature 0.0, seed 42, INVEST_WEIGHT 0.90",
            "llm_calls": 0,
            "analysis": "pure re-derivation from the frozen file",
            "bootstrap": {"seed": SEED, "resamples": N_BOOT,
                          "valid_auc_resamples": aN, "valid_rho_resamples": rN},
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "sample": {"n": N, "n_funded": NPOS, "n_not_funded": N - NPOS,
                   "positive_base_rate": round(NPOS / N, 4),
                   "negative_base_rate": round((N - NPOS) / N, 4)},
        "score_validity": {
            "auc": A, "auc_ci95": [aLo, aHi],
            "spearman_rho_vs_funded_pct": R, "spearman_ci95": [rLo, rHi],
            "mann_whitney_u": round(float(u.statistic), 1),
            "mann_whitney_p_one_sided": float(u.pvalue),
            "interpretation": "discrimination is substantially above chance; CI excludes 0.5",
        },
        "ranking_utility": {
            "precision_at_k": pak,
            "headline": ("screening the top 25 proposals by AI score yields 40.0% funded against "
                         "a 17.0% base rate — 2.35x lift, capturing 41.7% of all funded projects"),
        },
        "threshold_validity_at_production": {
            "tau_r": TAU_R, "tau_c": TAU_C,
            "max_observed_score": round(max(score), 4),
            "max_observed_confidence": round(max(conf), 4),
            "n_reaching_tau_r": sum(1 for x in score if x >= TAU_R),
            "n_reaching_tau_c": sum(1 for x in conf if x >= TAU_C),
            "rules": {name: confusion([1 if fn(s, c) else 0 for s, c in zip(score, conf)])
                      for name, fn in RULES.items()},
            "structural_explanation": (
                "BOTH thresholds are unreachable: no proposal attains tau_r (max score 0.640) "
                "and none attains tau_c (max confidence 0.661). The OR rule therefore has two "
                "dead branches, so every Boolean combination collapses to the same constant "
                "classifier. The four rules are identical by construction, not by coincidence."),
        },
        "best_achievable_threshold": {
            "max_f1": {k: best_f1[k] for k in ("tau", "f1", "precision", "recall", "accuracy", "n_predicted_go")},
            "max_youden_j": {k: best_j[k] for k in ("tau", "youden_j", "precision", "recall", "accuracy", "n_predicted_go")},
            "interpretation": ("a threshold that works exists and sits well inside the observed "
                               "score range; the failure is calibration of the decision layer, "
                               "not absence of signal in the score"),
        },
        "claims_supported": [
            "A deployed screening configuration can report 83% accuracy while achieving zero "
            "recall and discarding a score with AUC 0.764.",
            "Thresholded accuracy alone cannot certify that a screening system ranks proposals "
            "usefully (existence proof).",
            "All four verdict rules are inert here, and this is structural: both thresholds "
            "exceed the maximum observed value of their respective quantities.",
            "The AI score has genuine, statistically significant ranking utility "
            "(AUC CI excludes 0.5; Mann-Whitney p = 2.3e-05; 2.35x lift at k=25).",
            "A working threshold exists at tau ~= 0.53-0.55, giving recall 0.88-0.92.",
        ],
        "claims_NOT_supported": [
            "Do NOT claim this pattern is typical or general — it is an existence proof from one "
            "dataset and one configuration.",
            "Do NOT describe the score as having 'strong predictive power'. The AUC lower bound "
            "is 0.667; 'substantially above chance' is the defensible phrasing.",
            "Do NOT present the system as a precise predictor. At the best threshold precision "
            "is 0.318 — it is a screening filter that halves review workload, not a classifier.",
            "Do NOT claim the AI would have picked winners. Recall@k is bounded: even the top 50 "
            "of 141 capture only 58% of funded projects.",
        ],
        "required_caveats": [
            "tau_r = 0.725 and tau_c = 0.811 were calibrated on Dataset1 (12 expert-reviewed "
            "proposals) and transferred to Kickstarter unchanged. This transfer IS the mechanism "
            "of the failure and must be stated, not hidden.",
            "funded_pct >= 100 measures crowdfunding success, not venture quality. It is a real "
            "outcome, which is the point, but it is not the construct the expert panel scores.",
            "n = 141 with only 24 positives; all interval estimates are correspondingly wide.",
        ],
        "suggested_figure": (
            "Single panel: AI overall score on the x-axis, funded vs. not-funded as jittered "
            "points or stacked histogram, with three vertical reference lines — max observed "
            "score 0.640, best achievable tau 0.530, production tau_r 0.725. The production "
            "line falls outside the data range, which carries the argument visually."),
        "full_threshold_sweep": sweep,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {OUT}")
    print(f"  AUC {A} CI [{aLo}, {aHi}] | rho {R} CI [{rLo}, {rHi}] | MWU p={u.pvalue:.3e}")
    print(f"  best F1 tau={best_f1['tau']} recall={best_f1['recall']} | "
          f"best J tau={best_j['tau']} recall={best_j['recall']}")
    print(f"  reaching tau_r: {payload['threshold_validity_at_production']['n_reaching_tau_r']}/{N} | "
          f"reaching tau_c: {payload['threshold_validity_at_production']['n_reaching_tau_c']}/{N}")


if __name__ == "__main__":
    main()
