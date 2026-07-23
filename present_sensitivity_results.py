#!/usr/bin/env python3
"""
Present cached per-subrubric sensitivity results across all three dimensions.

Shows results grouped by variant — BASELINE first, then PLUS (adding content),
then MINUS (removing content) — each as a proposal × rubric matrix.

Reads from pre-computed JSON files — no LLM calls.

Usage:
    python3 present_sensitivity_results.py
    python3 present_sensitivity_results.py --dimensions strategy advantages
    python3 present_sensitivity_results.py --pids p1 p2 p3
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from run_sensitivity_all_dimensions import DIMENSIONS, ALL_PIDS

ROOT        = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "src" / "data" / "sensitivity_results"

DIMENSION_FILES = {
    "strategy":   RESULTS_DIR / "sensitivity_strategy.json",
    "advantages": RESULTS_DIR / "sensitivity_advantages.json",
    "objectives": RESULTS_DIR / "sensitivity_objectives.json",
}

SEP2 = "═" * 90
SEP1 = "─" * 90
CW   = 10   # cell width for baseline
CWD  = 10   # cell width for delta columns (e.g. "5(+4↑)")
PW   = 6    # proposal id column width


def _stats(passes: int, valid: int, deltas: list) -> dict:
    """Binomial test vs p=0.5 chance baseline, plus effect size metrics."""
    if valid == 0:
        return {}
    rate  = passes / valid
    p_val = binomtest(passes, valid, p=0.5, alternative='greater').pvalue
    ci    = binomtest(passes, valid, p=0.5, alternative='two-sided').proportion_ci(
                confidence_level=0.95, method='wilson')
    h    = 2 * np.arcsin(np.sqrt(rate)) - 2 * np.arcsin(np.sqrt(0.5))
    sig  = ("***" if p_val < 0.001 else
            "**"  if p_val < 0.01  else
            "*"   if p_val < 0.05  else "n.s.")
    return {
        "p": p_val, "sig": sig,
        "ci_low": ci.low, "ci_high": ci.high,
        "h": h,
        "h_label": "large" if h >= 0.5 else ("medium" if h >= 0.2 else "small"),
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


def load_dimension_results(dim: str) -> tuple[dict | None, list, list]:
    path = DIMENSION_FILES[dim]
    if not path.exists():
        print(f"  [WARN] No cached results for '{dim}' at {path}")
        return None, [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    plus_targets  = data.get("plus_targets",  DIMENSIONS[dim]["plus_targets"])
    minus_targets = data.get("minus_targets", DIMENSIONS[dim]["minus_targets"])
    return data.get("results", {}), plus_targets, minus_targets


def _short_header(key: str, plus_targets: list, minus_targets: list) -> str:
    parts = key.split("_")
    r_idx = next((i for i, p in enumerate(parts) if p.startswith("R") and p[1:].isdigit()), None)
    if r_idx is None:
        return key[-8:]
    r_num = parts[r_idx]
    suffix = parts[r_idx + 1][:3] if r_idx + 1 < len(parts) else ""
    label = f"{r_num}_{suffix}"
    if key in plus_targets:
        label += "[+]"
    elif key in minus_targets:
        label += "[-]"
    return label


def _delta_cell(new_val, base_val, cell_width: int) -> str:
    if not isinstance(new_val, int) or not isinstance(base_val, int):
        s = str(new_val) if new_val is not None else "-"
        return s.rjust(cell_width)
    d = new_val - base_val
    if d > 0:
        s = f"{new_val}(+{d}↑)"
    elif d < 0:
        s = f"{new_val}({d}↓)"
    else:
        s = f"{new_val}(→)"
    return s.rjust(cell_width)


def print_grouped(dim: str, cfg: dict, results: dict, pids: list) -> tuple[int, int]:
    rubrics       = cfg["rubrics"]
    plus_targets  = cfg["plus_targets"]
    minus_targets = cfg["minus_targets"]
    keys          = [r[0] for r in rubrics]
    headers       = [_short_header(k, plus_targets, minus_targets) for k in keys]

    active_pids = [p for p in pids if p in results]

    # Header row builder
    def header_row():
        return "  " + " " * PW + "".join(h.rjust(CW) for h in headers)

    print()
    print(SEP2)
    print(f"  DIMENSION: {dim.upper()}")
    print(f"  PLUS  = inject specific evidence (adding content)")
    print(f"  MINUS = strip key language (removing content)")
    print(f"  * [+] = PLUS hypothesis target   [-] = MINUS hypothesis target")
    print(SEP2)

    # ── BASELINE ──────────────────────────────────────────────────────────────
    print()
    print(f"  ── BASELINE {'─'*66}")
    print(header_row())
    print(f"  {'─'*86}")
    for pid in active_pids:
        b = results[pid].get("baseline") or {}
        row = f"  {pid:<{PW}}"
        for k in keys:
            v = b.get(k, "-")
            row += str(v).rjust(CW)
        print(row)

    # ── PLUS ──────────────────────────────────────────────────────────────────
    print()
    print(f"  ── PLUS  (adding content) {'─'*57}")
    print(header_row())
    print(f"  {'─'*86}")
    for pid in active_pids:
        b = results[pid].get("baseline") or {}
        p = results[pid].get("plus")     or {}
        row = f"  {pid:<{PW}}"
        for k in keys:
            row += _delta_cell(p.get(k), b.get(k), CW)
        print(row)

    # ── MINUS ─────────────────────────────────────────────────────────────────
    print()
    print(f"  ── MINUS  (removing content) {'─'*54}")
    print(header_row())
    print(f"  {'─'*86}")
    for pid in active_pids:
        b = results[pid].get("baseline") or {}
        m = results[pid].get("minus")    or {}
        row = f"  {pid:<{PW}}"
        for k in keys:
            row += _delta_cell(m.get(k), b.get(k), CW)
        print(row)

    # ── HYPOTHESIS CHECK ──────────────────────────────────────────────────────
    CWH = 9   # cell width per rubric column

    def _hyp_table(title: str, targets: list, variant_key: str) -> tuple[int, int, int, list]:
        """Returns (passes, valid_checks, excluded_count, deltas)."""
        short = [k.split("_R")[1][:5] for k in targets]
        excl_note = "ceiling excluded (baseline=5)" if variant_key == "plus" else "floor excluded (baseline=1)"
        print()
        print(f"  ── {title} {'─'*(76 - len(title))}")
        print(f"  — = {excl_note}")
        print("  " + f"{'PID':<{PW}}" + "".join(f"{l:>{CWH}}" for l in short) + f"  {'Passed':>10}")
        print(f"  {'─'*PW}  " + "  ".join(["─"*(CWH-2)] * len(targets)) + "  ─────────")
        t_pass = t_valid = t_excl = 0
        deltas = []
        for pid in active_pids:
            b = results[pid].get("baseline") or {}
            v = results[pid].get(variant_key) or {}
            row = f"  {pid:<{PW}}"
            passed = valid = excl = 0
            for key in targets:
                bv = b.get(key)
                vv = v.get(key)
                at_limit = (variant_key == "plus" and bv == 5) or (variant_key == "minus" and bv == 1)
                if at_limit:
                    row += f"{'—':>{CWH}}"
                    excl += 1
                else:
                    if variant_key == "plus":
                        ok = isinstance(vv, int) and isinstance(bv, int) and vv > bv
                    else:
                        ok = isinstance(vv, int) and isinstance(bv, int) and vv < bv
                    row += f"{'✓':>{CWH}}" if ok else f"{'✗':>{CWH}}"
                    passed += ok; valid += 1
                    if isinstance(vv, int) and isinstance(bv, int):
                        deltas.append(vv - bv if variant_key == "plus" else bv - vv)
            pct_row = 100 * passed // valid if valid else 0
            excl_str = f" (-{excl} excl)" if excl else ""
            row += f"  {passed}/{valid}{excl_str} ({pct_row}%)"
            print(row)
            t_pass += passed; t_valid += valid; t_excl += excl
        pct_all = 100 * t_pass // t_valid if t_valid else 0
        print(f"  {'─'*PW}")
        print(f"  {'Overall':<{PW}}  {t_pass}/{t_valid} ({pct_all}%)  [excluded: {t_excl}]")
        return t_pass, t_valid, t_excl, deltas

    plus_pass,  plus_valid,  plus_excl,  plus_deltas  = _hyp_table("PLUS Hypothesis (scores must RISE)",  plus_targets,  "plus")
    minus_pass, minus_valid, minus_excl, minus_deltas = _hyp_table("MINUS Hypothesis (scores must FALL)", minus_targets, "minus")

    total_pass   = plus_pass  + minus_pass
    total_checks = plus_valid + minus_valid
    total_excl   = plus_excl  + minus_excl
    pct = 100 * total_pass // total_checks if total_checks else 0
    print()
    print(f"  Combined: {total_pass}/{total_checks} valid checks passed ({pct}%)  [total excluded: {total_excl}]")

    # ── Statistical Tests ─────────────────────────────────────────────────────
    print()
    print(f"  ── Statistical Tests  (H₀: p = 0.50 chance baseline, one-sided binomial) {'─'*12}")
    print(f"  {'Variant':<9}  {'Passes':>10}  {'p-value':>12}  {'Sig':>4}  {'95% CI':>18}  {'Cohen h':>10}  {'Mean Δ':>8}")
    print(f"  {'─'*9}  {'─'*10}  {'─'*12}  {'─'*4}  {'─'*18}  {'─'*10}  {'─'*8}")
    for label, passes, valid, dlist in [
        ("PLUS",    plus_pass,  plus_valid,  plus_deltas),
        ("MINUS",   minus_pass, minus_valid, minus_deltas),
        ("Combined",total_pass, total_checks, plus_deltas + minus_deltas),
    ]:
        if valid == 0:
            continue
        s = _stats(passes, valid, dlist)
        rate_str = f"{passes}/{valid} ({100*passes//valid}%)"
        p_str    = f"{s['p']:.2e}"
        ci_str   = f"[{100*s['ci_low']:.1f}%, {100*s['ci_high']:.1f}%]"
        h_str    = f"{s['h']:.2f} ({s['h_label']})"
        d_str    = (f"+{s['mean_delta']:.2f}" if s['mean_delta'] >= 0 else f"{s['mean_delta']:.2f}") if label != "Combined" else "—"
        print(f"  {label:<9}  {rate_str:>10}  {p_str:>12}  {s['sig']:>4}  {ci_str:>18}  {h_str:>10}  {d_str:>8}")
    print(SEP2)

    return total_pass, total_checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids",       nargs="+", default=ALL_PIDS)
    ap.add_argument("--dimensions", nargs="+", default=list(DIMENSIONS.keys()),
                    choices=list(DIMENSIONS.keys()))
    args = ap.parse_args()

    totals: dict[str, tuple[int, int]] = {}

    for dim in args.dimensions:
        results, plus_targets, minus_targets = load_dimension_results(dim)
        if results is None:
            continue
        cfg = {**DIMENSIONS[dim], "plus_targets": plus_targets, "minus_targets": minus_targets}
        passed, checks = print_grouped(dim, cfg, results, args.pids)
        totals[dim] = (passed, checks)

    if totals:
        print()
        print(SEP2)
        print(f"  AGGREGATE HYPOTHESIS SUMMARY")
        print(f"  {'─'*50}")
        grand_pass = grand_total = 0
        for dim, (p, t) in totals.items():
            pct = 100 * p // t if t else 0
            print(f"  {dim:<12}:  {p}/{t} checks  ({pct}%)")
            grand_pass  += p
            grand_total += t
        print(f"  {'─'*50}")
        gpct = 100 * grand_pass // grand_total if grand_total else 0
        print(f"  {'TOTAL':<12}:  {grand_pass}/{grand_total} checks  ({gpct}%)")
        print(SEP2)


if __name__ == "__main__":
    main()
