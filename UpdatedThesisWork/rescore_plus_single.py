#!/usr/bin/env python3
"""
Per-sub-rubric PLUS test: one report per proposal x sub-rubric, with ONLY that sub-rubric's added sentence.

Run rescore_corrected.py first: this reuses its baseline scores (first baseline run), its scorer call, its
insertion point and its neutral report IDs, so the comparison uses identical settings.

For each proposal x sub-rubric:
  target     PLUS passes when that sub-rubric scores above its baseline (excluded at baseline 5)
  spillover  for the other 7 sub-rubrics in the same report, how often the score rose although nothing aimed at
             them was added; compared with how often scores rise between two identical baseline runs

Output: src/data/sensitivity_results/corrected_v2/plus_single.json

Usage:
    python3 rescore_plus_single.py --build_only
    python3 rescore_plus_single.py
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import rescore_corrected as rc


def build_text(runner, original: str, sentence: str) -> str:
    text = runner.adaptive_inject_plus(original, sentence, rc.PLUS_MARKER)
    if text == original + "\n\n" + sentence:
        raise RuntimeError("PLUS marker not found")
    if text.find(sentence) + len(sentence) > rc.SCORER_WINDOW:
        raise RuntimeError("PLUS sentence is past the scorer window")
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
            sentences = rc.plus_sentences(runner, pid, dim)
            for (key, _, _), sentence in zip(runner.DIMENSIONS[dim]["rubrics"], sentences):
                report_id = rc.neutral_id(pid, dim, f"plus_only|{key}")
                (rc.REPORT_DIR / f"{report_id}_final_report.md").write_text(
                    build_text(runner, original, sentence), encoding="utf-8")
                cells[(pid, dim, key)] = report_id
    print(f"[OK] wrote {len(cells)} single-sentence PLUS reports; all inside the scorer window")
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
        result = ("EXCLUDED" if rc.is_excluded("PLUS", b)
                  else "PASS" if rc.passed("PLUS", scored[key], b) else "FAIL")
        rows.append({"pid": pid, "dimension": dim, "sub_rubric": key, "rubric_type": types[key],
                     "report_id": rid, "baseline": b, "plus_single": scored[key], "result": result})
        for other, ob in base.items():
            if other != key and ob < 5:
                spill.append({"dimension": dim, "rose": scored[other] > ob})

    # Noise reference: how often a score below 5 rises between the two identical baseline runs.
    noise = [c["baseline_repeat"] > c["baseline"] for c in corrected["checks"] if c["baseline"] < 5]

    (rc.OUT_DIR / "plus_single.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": __doc__,
        "checks": rows,
        "spillover_rise_rate": sum(s["rose"] for s in spill) / len(spill),
        "spillover_n": len(spill),
        "noise_rise_rate": sum(noise) / len(noise),
        "noise_n": len(noise),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────────────────────
    groups = defaultdict(list)
    for r in rows:
        for g in ((r["dimension"], r["rubric_type"]), ("ALL", r["rubric_type"]), (r["dimension"], "both")):
            groups[g].append(r)
    print(f"\n  {'dimension':<12}{'type':<16}{'PLUS (single)':>18}{'reached 5':>11}{'excluded':>10}")
    for dim in rc.DIMENSIONS + ["ALL"]:
        for rtype in ("doc-verifiable", "judgment-dep") + (("both",) if dim != "ALL" else ()):
            g = groups[(dim, rtype)]
            ok = sum(r["result"] == "PASS" for r in g)
            n = sum(r["result"] in ("PASS", "FAIL") for r in g)
            top = sum(r["plus_single"] == 5 for r in g)
            shown = f"{ok}/{n} ({100 * ok / n:.0f}%)" if n else "0/0"
            print(f"  {dim:<12}{rtype:<16}{shown:>18}{f'{top}/{len(g)}':>11}"
                  f"{sum(r['result'] == 'EXCLUDED' for r in g):>10}")
    s = sum(x["rose"] for x in spill)
    print(f"\n  spillover: other sub-rubrics rose in {s}/{len(spill)} cases ({100 * s / len(spill):.0f}%)")
    print(f"  noise:     scores rose between identical baseline runs in "
          f"{sum(noise)}/{len(noise)} cases ({100 * sum(noise) / len(noise):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
