#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 — Human-human agreement among the 3 expert raters.

Reads the long-format table produced by build_consensus_ground_truth.py
(resultsNew/Dataset1/raters_long.csv) and computes, per dimension and for the
overall ranking:

  - ICC(2,1)  two-way random, absolute agreement, single rater
  - ICC(2,k)  two-way random, absolute agreement, mean of k raters (k=3)
    both with 95% CIs (McGraw & Wong 1996 F-distribution intervals)
  - Pairwise quadratic-weighted Cohen's kappa between each rater pair (5 dims)
  - Gwet's AC1 (unweighted, multi-rater) per dimension

No LLM calls; pure re-analysis of the cached expert Excel scores.

Outputs:
  - resultsNew/Dataset1/rater_reliability.json
  - resultsNew/Dataset1/rater_reliability_summary.md
"""
import csv
import json
import statistics
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "resultsNew" / "Dataset1"
LONG_CSV = RESULTS_DIR / "raters_long.csv"

DIMS = ["team", "objective", "strategy", "advantages", "feasibility"]
ALL_TARGETS = DIMS + ["overall_ranking"]
SCALE_MIN, SCALE_MAX = 1, 5  # 1-5 ordinal scale for kappa weight matrix / AC1


def load_matrix():
    """Return {dimension -> {proposal_id -> {rater_id -> score}}}."""
    data = {t: {} for t in ALL_TARGETS}
    with LONG_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dim = row["dimension"]
            pid = int(row["proposal_id"])
            rid = int(row["rater_id"])
            data[dim].setdefault(pid, {})[rid] = float(row["score"])
    return data


def _matrix(dim_data):
    """n x k numpy array of scores (rows=proposals, cols=raters 1,2,3)."""
    pids = sorted(dim_data)
    raters = sorted({r for p in dim_data.values() for r in p})
    M = np.array([[dim_data[p][r] for r in raters] for p in pids], dtype=float)
    return M


# ---------------------------------------------------------------------------
# ICC(2,1) and ICC(2,k) with McGraw & Wong (1996) absolute-agreement CIs
# ---------------------------------------------------------------------------
def icc_2(M, alpha=0.05):
    n, k = M.shape
    grand = M.mean()
    row_means = M.mean(axis=1)
    col_means = M.mean(axis=0)

    SST = ((M - grand) ** 2).sum()
    SSB = k * ((row_means - grand) ** 2).sum()          # between targets
    SSC = n * ((col_means - grand) ** 2).sum()          # between raters
    SSE = SST - SSB - SSC

    MSB = SSB / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    icc1 = (MSB - MSE) / (MSB + (k - 1) * MSE + k * (MSC - MSE) / n)
    icck = (MSB - MSE) / (MSB + (MSC - MSE) / n)

    # 95% CI for ICC(A,1) — McGraw & Wong 1996
    fj = MSC / MSE
    vn = (k - 1) * (n - 1) * (k * icc1 * fj + n * (1 + (k - 1) * icc1) - k * icc1) ** 2
    vd = ((n - 1) * k ** 2 * icc1 ** 2 * fj ** 2
          + (n * (1 + (k - 1) * icc1) - k * icc1) ** 2)
    v = vn / vd
    f_u = f_dist.ppf(1 - alpha / 2, n - 1, v)
    f_l = f_dist.ppf(1 - alpha / 2, v, n - 1)
    lo1 = n * (MSB - f_u * MSE) / (f_u * (k * MSC + (k * n - k - n) * MSE) + n * MSB)
    hi1 = n * (f_l * MSB - MSE) / (k * MSC + (k * n - k - n) * MSE + n * f_l * MSB)

    # transform single-rater CI to k-rater CI
    lok = lo1 * k / (1 + (k - 1) * lo1)
    hik = hi1 * k / (1 + (k - 1) * hi1)

    r = lambda x: round(float(x), 4)
    return {
        "n": n, "k": k,
        "icc_2_1": r(icc1), "icc_2_1_ci95": [r(lo1), r(hi1)],
        "icc_2_k": r(icck), "icc_2_k_ci95": [r(lok), r(hik)],
        "ms_between": r(MSB), "ms_raters": r(MSC), "ms_error": r(MSE),
    }


# ---------------------------------------------------------------------------
# Pairwise quadratic-weighted Cohen's kappa on the fixed 1-5 scale
# ---------------------------------------------------------------------------
def weighted_kappa(a, b, cmin=SCALE_MIN, cmax=SCALE_MAX):
    cats = list(range(cmin, cmax + 1))
    q = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    W = np.array([[1 - (i - j) ** 2 / (q - 1) ** 2 for j in range(q)] for i in range(q)])
    O = np.zeros((q, q))
    for x, y in zip(a, b):
        O[idx[int(round(x))], idx[int(round(y))]] += 1
    n = O.sum()
    if n == 0:
        return None
    O /= n
    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E = np.outer(row, col)
    num = (W * O).sum()
    den = (W * E).sum()
    if abs(1 - den) < 1e-12:
        return None
    return round(float((num - den) / (1 - den)), 4)


# ---------------------------------------------------------------------------
# Gwet's AC1 (multi-rater, unweighted) using categories observed in the data
# ---------------------------------------------------------------------------
def gwet_ac1(M):
    n, r = M.shape
    cats = sorted({int(round(v)) for v in M.flatten()})
    q = len(cats)
    if q < 2:
        return {"ac1": 1.0, "categories": cats, "p_a": 1.0, "p_e": 0.0, "note": "single category"}
    # r_ik: count of raters assigning subject i to category k
    counts = {c: np.array([np.sum(np.round(M[i]) == c) for i in range(n)], dtype=float) for c in cats}
    # observed agreement per subject (all r raters present)
    pa_i = np.array([
        sum(counts[c][i] * (counts[c][i] - 1) for c in cats) / (r * (r - 1))
        for i in range(n)
    ])
    p_a = float(pa_i.mean())
    pi = {c: float((counts[c] / r).mean()) for c in cats}
    p_e = sum(pi[c] * (1 - pi[c]) for c in cats) / (q - 1)
    ac1 = (p_a - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-12 else None
    return {
        "ac1": round(ac1, 4) if ac1 is not None else None,
        "categories": cats, "p_a": round(p_a, 4), "p_e": round(p_e, 4),
    }


def main():
    data = load_matrix()
    result = {"raters": [1, 2, 3], "k": 3, "icc": {}, "pairwise_weighted_kappa": {}, "gwet_ac1": {}}

    for target in ALL_TARGETS:
        M = _matrix(data[target])
        result["icc"][target] = icc_2(M)

    for dim in DIMS:
        dim_data = data[dim]
        pids = sorted(dim_data)
        pair_kappas = {}
        for r1, r2 in combinations([1, 2, 3], 2):
            a = [dim_data[p][r1] for p in pids]
            b = [dim_data[p][r2] for p in pids]
            pair_kappas[f"rater{r1}_vs_rater{r2}"] = weighted_kappa(a, b)
        vals = [v for v in pair_kappas.values() if v is not None]
        pair_kappas["mean"] = round(statistics.mean(vals), 4) if vals else None
        result["pairwise_weighted_kappa"][dim] = pair_kappas
        result["gwet_ac1"][dim] = gwet_ac1(_matrix(dim_data))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rater_reliability.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    # ---- human-readable summary ----
    lines = []
    lines.append("# Human-Human Rater Reliability (3 experts, k=3)\n")
    lines.append("## ICC(2,1) single-rater & ICC(2,k) panel-mean, absolute agreement\n")
    lines.append("| Target | ICC(2,1) | 95% CI | ICC(2,k) | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for t in ALL_TARGETS:
        m = result["icc"][t]
        lines.append(f"| {t} | {m['icc_2_1']} | [{m['icc_2_1_ci95'][0]}, {m['icc_2_1_ci95'][1]}] "
                     f"| {m['icc_2_k']} | [{m['icc_2_k_ci95'][0]}, {m['icc_2_k_ci95'][1]}] |")
    lines.append("\n## Pairwise quadratic-weighted Cohen's kappa (per dimension)\n")
    lines.append("| Dimension | r1-r2 | r1-r3 | r2-r3 | mean |")
    lines.append("|---|---|---|---|---|")
    for dim in DIMS:
        pk = result["pairwise_weighted_kappa"][dim]
        lines.append(f"| {dim} | {pk['rater1_vs_rater2']} | {pk['rater1_vs_rater3']} "
                     f"| {pk['rater2_vs_rater3']} | {pk['mean']} |")
    lines.append("\n## Gwet's AC1 (unweighted, multi-rater; per dimension)\n")
    lines.append("| Dimension | AC1 | p_a | p_e | categories |")
    lines.append("|---|---|---|---|---|")
    for dim in DIMS:
        g = result["gwet_ac1"][dim]
        lines.append(f"| {dim} | {g['ac1']} | {g['p_a']} | {g['p_e']} | {g['categories']} |")
    (RESULTS_DIR / "rater_reliability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[OK] rater_reliability.json + rater_reliability_summary.md written")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
