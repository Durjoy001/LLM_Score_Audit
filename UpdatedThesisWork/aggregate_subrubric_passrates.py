#!/usr/bin/env python3
"""
Per-sub-rubric pass rates for the PLUS/MINUS evidence-injection perturbation test.

Source of truth
---------------
  src/data/sensitivity_results/sensitivity_all_dimensions.json   (mode="adaptive")

Produced by `run_sensitivity_all_dimensions.py --adaptive`, which sets both
plus_targets and minus_targets to all 8 rubric keys per dimension, giving
12 proposals x 8 sub-rubrics x 2 directions x 5 dimensions = 960 raw checks.
Team and Feasibility were added on 2026-09-16 with newly drafted sub-rubrics;
they have no published figures, so the paper check below covers the original three.

Pass criterion
--------------
Mirrors generate_sensitivity_report.py:196-226 exactly (the code that rendered
results/Dataset1/sensitivity_report.html, from which the paper's figures were read):

  PLUS  passes when plus  > baseline
  MINUS passes when minus < baseline

Checks at the scale limit are EXCLUDED from the denominator, because the score
physically cannot move in the expected direction:

  PLUS  excluded when baseline == 5   (ceiling effect)
  MINUS excluded when baseline == 1   (floor effect)

This exclusion is what reproduces the published dimension figures. Without it,
Strategy MINUS reads 51% instead of 89%.

Rounding note: the HTML report uses floor division (100 * passed // valid), so it
renders 91/93/95 where the paper reports 92/94/96. This script rounds (the paper's
convention) and prints both so the difference is never silent.

Output
------
  subrubric_pass_rates.csv   80 rows, tidy/long format (40 sub-rubrics x 2 directions)

Usage
-----
  python3 aggregate_subrubric_passrates.py
"""

import ast
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RESULTS_JSON = ROOT / "src" / "data" / "sensitivity_results" / "sensitivity_all_dimensions.json"
CONFIG_PY    = ROOT / "run_sensitivity_all_dimensions.py"
RENDERED_HTML = ROOT / "results" / "Dataset1" / "sensitivity_report.html"
OUT_CSV      = HERE / "subrubric_pass_rates.csv"

# Dimension-level figures as reported in the paper, for the sanity check.
PAPER_FIGURES = {
    "strategy":   {"PLUS": 92, "MINUS": 89},
    "objectives": {"PLUS": 96, "MINUS": 100},
    "advantages": {"PLUS": 67, "MINUS": 94},
}

# A cell with this few testable checks cannot support a meaningful rate.
THIN_CELL_THRESHOLD = 3

DIRECTIONS = ("PLUS", "MINUS")
VARIANT_KEY = {"PLUS": "plus", "MINUS": "minus"}


def load_rubric_metadata(path: Path) -> dict:
    """Pull (key, label, type) per dimension out of the experiment config.

    Parsed via ast rather than imported: importing the runner would execute
    load_dotenv() and pull in the LLM provider, which needs credentials we
    do not want this read-only aggregation to depend on.
    """
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if getattr(node.targets[0], "id", "") != "DIMENSIONS":
            continue
        meta = {}
        for dim_node, cfg_node in zip(node.value.keys, node.value.values):
            for cfg_k, cfg_v in zip(cfg_node.keys, cfg_node.values):
                if cfg_k.value == "rubrics":
                    meta[dim_node.value] = [
                        tuple(x.value for x in elt.elts) for elt in cfg_v.elts
                    ]
        return meta
    raise RuntimeError(f"DIMENSIONS not found in {path}")


def is_excluded(direction: str, baseline: int) -> bool:
    """Ceiling (PLUS at 5) / floor (MINUS at 1) checks are not testable."""
    return (direction == "PLUS" and baseline == 5) or (direction == "MINUS" and baseline == 1)


def passed(direction: str, variant: int, baseline: int) -> bool:
    return variant > baseline if direction == "PLUS" else variant < baseline


def tally(per_pid: dict, rubric_key: str, direction: str) -> tuple[int, int, int]:
    """Return (checks_passed, valid_checks, n_excluded) for one sub-rubric x direction."""
    vkey = VARIANT_KEY[direction]
    n_pass = n_valid = n_excl = 0
    for record in per_pid.values():
        baseline = (record.get("baseline") or {}).get(rubric_key)
        variant = (record.get(vkey) or {}).get(rubric_key)
        if not isinstance(baseline, int):
            continue
        if is_excluded(direction, baseline):
            n_excl += 1
            continue
        n_valid += 1
        if isinstance(variant, int) and passed(direction, variant, baseline):
            n_pass += 1
    return n_pass, n_valid, n_excl


def main() -> int:
    if not RESULTS_JSON.exists():
        print(f"ERROR: results not found: {RESULTS_JSON}", file=sys.stderr)
        return 1

    payload = json.loads(RESULTS_JSON.read_text())
    if payload.get("mode") != "adaptive":
        print(
            f"ERROR: expected mode='adaptive' (all 8 sub-rubrics targeted), "
            f"got {payload.get('mode')!r}. The hardcoded run only perturbs 2+2 "
            f"rubrics per dimension and cannot yield a sub-rubric breakdown.",
            file=sys.stderr,
        )
        return 1

    results = payload["results"]
    rubric_meta = load_rubric_metadata(CONFIG_PY)

    rows = []
    rollup = {}  # (dimension, direction) -> [passed, valid, excluded]

    for dimension, rubrics in rubric_meta.items():
        if dimension not in results:
            print(f"  [WARN] dimension {dimension!r} absent from results, skipped")
            continue
        per_pid = results[dimension]
        for rubric_key, label, rubric_type in rubrics:
            for direction in DIRECTIONS:
                n_pass, n_valid, n_excl = tally(per_pid, rubric_key, direction)
                rollup.setdefault((dimension, direction), [0, 0, 0])
                rollup[(dimension, direction)][0] += n_pass
                rollup[(dimension, direction)][1] += n_valid
                rollup[(dimension, direction)][2] += n_excl

                # n_valid == 0 means every baseline sat at the limit: the rate is
                # undefined, NOT zero. Emitting 0.0 here would read as a total
                # failure to respond when in fact nothing was testable.
                rate = round(n_pass / n_valid, 4) if n_valid else ""
                rows.append({
                    "sub_rubric": rubric_key,
                    "sub_rubric_label": label,
                    "dimension": dimension,
                    "rubric_type": rubric_type,
                    "direction": direction,
                    "total_checks": n_valid,
                    "checks_passed": n_pass,
                    "pass_rate": rate,
                    "n_excluded": n_excl,
                    "n_proposals": len(per_pid),
                    "thin_cell": "YES" if n_valid <= THIN_CELL_THRESHOLD else "",
                })

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Dimension rollups: the sanity check against the paper ────────────────
    print()
    print("  DIMENSION-LEVEL ROLLUPS  (paper-consistent exclusion applied)")
    print("  " + "─" * 84)
    print(f"  {'Dimension':<12}{'Dir':<7}{'passed/valid':>14}{'exact':>9}{'rounded':>9}"
          f"{'floor':>8}{'paper':>8}   check")
    print("  " + "─" * 84)

    all_match = True
    paper_pass = paper_valid = 0
    for dimension in rubric_meta:
        for direction in DIRECTIONS:
            if (dimension, direction) not in rollup:
                continue
            n_pass, n_valid, n_excl = rollup[(dimension, direction)]
            if not n_valid:
                print(f"  {dimension:<12}{direction:<7}{'0/0':>14}{'n/a':>9}")
                continue
            exact = 100 * n_pass / n_valid
            rounded = round(exact)
            floored = 100 * n_pass // n_valid   # what sensitivity_report.html prints
            paper = PAPER_FIGURES.get(dimension, {}).get(direction)
            if paper is None:
                check = "new"
            else:
                ok = rounded == paper
                all_match &= ok
                paper_pass += n_pass
                paper_valid += n_valid
                check = "MATCH" if ok else "MISMATCH"
            print(f"  {dimension:<12}{direction:<7}{f'{n_pass}/{n_valid}':>14}"
                  f"{exact:>8.1f}%{rounded:>8}%{floored:>7}%{paper if paper else '-':>7}%"
                  f"   {check}")
    print("  " + "─" * 84)
    print(f"  All six paper dimension figures reproduce: {all_match}")
    print(f"  Paper dimensions only: {paper_pass}/{paper_valid}")

    grand_pass = sum(v[0] for v in rollup.values())
    grand_valid = sum(v[1] for v in rollup.values())
    print(f"  Grand total: {grand_pass}/{grand_valid} "
          f"({100 * grand_pass / grand_valid:.1f}%, floor {100 * grand_pass // grand_valid}%)")
    if RENDERED_HTML.exists():
        print(f"  Cross-check vs {RENDERED_HTML.relative_to(ROOT)}: "
              f"expect 366/417 in that file's aggregate row.")

    # ── Thin-cell warnings ───────────────────────────────────────────────────
    thin = [r for r in rows if r["thin_cell"] == "YES"]
    if thin:
        print()
        print(f"  THIN CELLS — {len(thin)} of {len(rows)} rows have "
              f"n <= {THIN_CELL_THRESHOLD} testable checks")
        print("  Pass rates on these are not interpretable; do not report them as findings.")
        print("  " + "─" * 84)
        for r in sorted(thin, key=lambda r: (r["total_checks"], r["sub_rubric"])):
            shown = f"{r['pass_rate']:.3f}" if r["pass_rate"] != "" else "undefined"
            print(f"    {r['sub_rubric']:<38}{r['direction']:<7}"
                  f"n={r['total_checks']:<3}(excl {r['n_excluded']:>2})  rate={shown}")

    print()
    print(f"  Wrote {len(rows)} rows -> {OUT_CSV}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
