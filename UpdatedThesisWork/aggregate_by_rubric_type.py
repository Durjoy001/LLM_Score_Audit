#!/usr/bin/env python3
"""
Roll the per-sub-rubric pass rates up by rubric TYPE (doc-verifiable vs judgment-dep).

Reads subrubric_pass_rates.csv (produced by aggregate_subrubric_passrates.py) and
pools checks within each dimension x type x direction cell.

Each dimension has 8 sub-rubrics: 4 doc-verifiable (R1-R4) + 4 judgment-dependent
(R5-R8). Rates are POOLED over checks (sum passed / sum valid), not averaged over
sub-rubric rates, so that cells with different denominators are weighted correctly
and the numbers still sum back to the published dimension figures.

Output: rubric_type_pass_rates.csv  (5 dimensions x 2 types x 2 directions = 20 rows)
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN_CSV = HERE / "subrubric_pass_rates.csv"
OUT_CSV = HERE / "rubric_type_pass_rates.csv"

DIMENSIONS = ["strategy", "objectives", "advantages", "team", "feasibility"]
TYPES = ["doc-verifiable", "judgment-dep"]
DIRECTIONS = ["PLUS", "MINUS"]


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run aggregate_subrubric_passrates.py first.",
              file=sys.stderr)
        return 1

    rows = list(csv.DictReader(IN_CSV.open()))

    pooled = defaultdict(lambda: {"passed": 0, "valid": 0, "excluded": 0, "rubrics": []})
    for r in rows:
        cell = pooled[(r["dimension"], r["rubric_type"], r["direction"])]
        cell["passed"] += int(r["checks_passed"])
        cell["valid"] += int(r["total_checks"])
        cell["excluded"] += int(r["n_excluded"])
        cell["rubrics"].append((r["sub_rubric"], int(r["checks_passed"]), int(r["total_checks"])))

    out = []
    for dim in DIMENSIONS:
        for typ in TYPES:
            for direction in DIRECTIONS:
                cell = pooled.get((dim, typ, direction))
                if cell is None:
                    continue
                valid = cell["valid"]
                # Undefined, not zero, when nothing was testable.
                rate = round(cell["passed"] / valid, 4) if valid else ""
                out.append({
                    "dimension": dim,
                    "rubric_type": typ,
                    "direction": direction,
                    "n_sub_rubrics": len(cell["rubrics"]),
                    "total_checks": valid,
                    "checks_passed": cell["passed"],
                    "pass_rate": rate,
                    "n_excluded": cell["excluded"],
                })

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print()
    print("  PASS RATES BY RUBRIC TYPE  (pooled over checks)")
    print("  " + "-" * 72)
    print(f"  {'Dimension':<12}{'Type':<16}{'Dir':<7}{'pass/valid':>12}{'rate':>9}{'excl':>7}")
    print("  " + "-" * 72)
    for dim in DIMENSIONS:
        for typ in TYPES:
            for direction in DIRECTIONS:
                row = next((r for r in out if r["dimension"] == dim
                            and r["rubric_type"] == typ
                            and r["direction"] == direction), None)
                if row is None:
                    continue
                rate = f"{100 * row['pass_rate']:.1f}%" if row["pass_rate"] != "" else "NA"
                frac = f"{row['checks_passed']}/{row['total_checks']}"
                print(f"  {dim:<12}{typ:<16}{direction:<7}{frac:>12}"
                      f"{rate:>9}{row['n_excluded']:>7}")
        print()

    # Is a type-level gap driven by a single outlier sub-rubric? Report the
    # weakest contributor in each PLUS cell so a gap is never read as a
    # broad property of the type when one rubric causes it.
    print("  WEAKEST SUB-RUBRIC IN EACH PLUS CELL")
    print("  " + "-" * 72)
    for dim in DIMENSIONS:
        for typ in TYPES:
            cell = pooled[(dim, typ, "PLUS")]
            worst = min(cell["rubrics"], key=lambda t: (t[1] / t[2]) if t[2] else 2)
            others = [t for t in cell["rubrics"] if t[0] != worst[0]]
            op, ov = sum(t[1] for t in others), sum(t[2] for t in others)
            rest = f"{100 * op / ov:.1f}%" if ov else "NA"
            wr = f"{100 * worst[1] / worst[2]:.1f}%" if worst[2] else "NA"
            print(f"  {dim:<12}{typ:<16}worst={worst[0]:<36}{wr:>7}   "
                  f"other 3 pooled={rest:>7}")
    print()
    print(f"  Wrote {len(out)} rows -> {OUT_CSV}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
