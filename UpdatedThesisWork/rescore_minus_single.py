#!/usr/bin/env python3
"""
Per-sub-rubric MINUS test: one report per proposal x sub-rubric, with ONLY that sub-rubric's passages removed.

Run rescore_corrected.py first: this reuses its baseline scores (first baseline run), its scorer call and its
neutral report IDs, so the comparison uses identical settings (temperature 0, no expert-opinion JSON).

For each of the cells in removed_text_review.py that have something to remove:
  target     MINUS passes when the removed sub-rubric scores below its baseline (excluded at baseline 1)
  spillover  for the other 7 sub-rubrics in the same report, how often the score dropped although none of
             their evidence was removed; compared with how often scores drop between two identical baseline runs

Output: src/data/sensitivity_results/corrected_v2/minus_single.json

Usage:
    python3 rescore_minus_single.py --build_only
    python3 rescore_minus_single.py
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import rescore_corrected as rc
from removed_text_review import MINUS_REMOVE, MINUS_REPLACEMENT


def build_text(original: str, key: str, finds: list[str]) -> str:
    text = original
    for find in finds:
        if find not in text:
            raise RuntimeError(f"{key}: removal passage not found: {find[:60]}")
        text = text.replace(find, MINUS_REPLACEMENT[key])
    if any(find in text[:rc.SCORER_WINDOW] for find in finds):
        raise RuntimeError(f"{key}: removal passage still visible")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_only", action="store_true", help="Write and check the reports; no LLM calls.")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    runner = rc.load_runner()
    corrected = json.loads((rc.OUT_DIR / "results.json").read_text())
    base_ids = {tuple(k.split("|")): v for k, v in corrected["id_map"].items()}

    cells = {}
    for pid in rc.PIDS:
        original = (rc.REPORT_DIR / f"{pid}__openai_final_report.md").read_text(encoding="utf-8")
        for dim in rc.DIMENSIONS:
            for key, finds in MINUS_REMOVE.get(pid, {}).get(dim, {}).items():
                report_id = rc.neutral_id(pid, dim, f"minus_only|{key}")
                (rc.REPORT_DIR / f"{report_id}_final_report.md").write_text(
                    build_text(original, key, finds), encoding="utf-8")
                cells[(pid, dim, key)] = report_id
    print(f"[OK] wrote {len(cells)} single-removal reports; all removals applied")
    if args.build_only:
        return 0

    todo = [(dim, rid) for (pid, dim, key), rid in cells.items()
            if not rc.score_path(runner, dim, rid).exists()]
    print(f"[..] scoring {len(todo)} reports ({len(cells) - len(todo)} already scored)")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        errors = [(rid, e) for (dim, rid), e in zip(todo, pool.map(lambda t: rc.run_scorer(runner, *t), todo)) if e]
    for rid, err in errors:
        print(f"  [WARN] {rid}: {err}")
    if errors:
        print(f"[FAIL] {len(errors)} reports failed to score; rerun to retry them")
        return 1

    # ── Judge ────────────────────────────────────────────────────────────────
    types = {k: t for d in rc.DIMENSIONS for k, _, t in runner.DIMENSIONS[d]["rubrics"]}
    rows, spill = [], []
    for (pid, dim, key), rid in cells.items():
        base = rc.load_scores(runner, dim, base_ids[(pid, dim, "base")])
        scored = rc.load_scores(runner, dim, rid)
        b = base[key]
        result = ("EXCLUDED" if rc.is_excluded("MINUS", b)
                  else "PASS" if rc.passed("MINUS", scored[key], b) else "FAIL")
        rows.append({"pid": pid, "dimension": dim, "sub_rubric": key, "rubric_type": types[key],
                     "report_id": rid, "baseline": b, "minus_single": scored[key], "result": result})
        for other, ob in base.items():
            if other != key and ob > 1:
                spill.append({"dimension": dim, "dropped": scored[other] < ob})

    # Noise reference: how often a score above 1 drops between the two identical baseline runs.
    noise = [c["baseline_repeat"] < c["baseline"] for c in corrected["checks"] if c["baseline"] > 1]

    (rc.OUT_DIR / "minus_single.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": __doc__,
        "checks": rows,
        "spillover_drop_rate": sum(s["dropped"] for s in spill) / len(spill),
        "spillover_n": len(spill),
        "noise_drop_rate": sum(noise) / len(noise),
        "noise_n": len(noise),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────────────────────
    groups = defaultdict(list)
    for r in rows:
        for g in ((r["dimension"], r["rubric_type"]), ("ALL", r["rubric_type"]), (r["dimension"], "both")):
            groups[g].append(r)
    print(f"\n  {'dimension':<12}{'type':<16}{'MINUS (single)':>18}{'excluded':>10}")
    for dim in rc.DIMENSIONS + ["ALL"]:
        for rtype in ("doc-verifiable", "judgment-dep") + (("both",) if dim != "ALL" else ()):
            g = groups[(dim, rtype)]
            ok = sum(r["result"] == "PASS" for r in g)
            n = sum(r["result"] in ("PASS", "FAIL") for r in g)
            shown = f"{ok}/{n} ({100 * ok / n:.0f}%)" if n else "0/0"
            print(f"  {dim:<12}{rtype:<16}{shown:>18}{sum(r['result'] == 'EXCLUDED' for r in g):>10}")
    s = sum(x["dropped"] for x in spill)
    print(f"\n  spillover: other sub-rubrics dropped in {s}/{len(spill)} cases ({100 * s / len(spill):.0f}%)")
    print(f"  noise:     scores dropped between identical baseline runs in "
          f"{sum(noise)}/{len(noise)} cases ({100 * sum(noise) / len(noise):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
