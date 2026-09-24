#!/usr/bin/env python3
"""
Rescore the PLUS/MINUS perturbation test with the manually corrected edits.

What changes from the original run (run_sensitivity_all_dimensions.py --adaptive)
---------------------------------------------------------------------------------
PLUS   the injected text is exactly the 8 sentences shown in the ADDED text column: the corrected sentence from
       added_text_review.py where the original was misaligned, otherwise the original rubric_coverage sentence.
       It is inserted before "##### Team and governance" for every dimension, so the scorer's 12,000-character
       window always contains all of it.
MINUS  the passages in removed_text_review.py are replaced (every occurrence) with the neutral per-sub-rubric
       sentence. Sub-rubrics with nothing to remove are reported as untestable, not as passes or fails.
Scorer the unchanged sub-rubric scorers (src/tools/generate_<dim>_rubric_scores.py), called identically for the
       baseline and both variants: temperature 0.0 and no AI expert-opinion JSON (it repeats the report's
       commentary, so giving it to the baseline only would undo the MINUS removals). The baseline is scored twice
       to measure run-to-run noise; the first run is used for pass/fail.
IDs    every scored report gets a neutral hashed id, so the proposal_id the scorer sees does not reveal the variant.

The pass rule is unchanged (aggregate_subrubric_passrates.py): PLUS passes when plus > baseline, MINUS when
minus < baseline; PLUS is excluded at baseline 5 and MINUS at baseline 1.

Outputs
-------
  src/data/reports/r2x<hash>__openai_final_report.md          the scored reports
  src/data/<dim>_rubric_scores/r2x<hash>__openai/...          scores written by the scorers
  src/data/sensitivity_results/corrected_v2/results.json      id map, scores, pass/fail per check

Usage
-----
  python3 rescore_corrected.py --build_only    # write reports and check them, no LLM calls
  python3 rescore_corrected.py                 # build, score (skips reports already scored), summarise
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "src" / "data"
REPORT_DIR = DATA / "reports"
CONTENT_DIR = DATA / "sensitivity_results" / "adaptive_content"
OUT_DIR = DATA / "sensitivity_results" / "corrected_v2"

sys.path.insert(0, str(HERE))
from added_text_review import REVIEW  # noqa: E402
from aggregate_subrubric_passrates import is_excluded, passed  # noqa: E402
from removed_text_review import MINUS_REMOVE, MINUS_REPLACEMENT  # noqa: E402

DIMENSIONS = ["strategy", "objectives", "advantages", "team", "feasibility"]
PIDS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "pA", "pB", "pC", "pD"]
VARIANTS = ["base", "base_repeat", "plus", "minus"]
PLUS_MARKER = "##### Team and governance"
SCORER_WINDOW = 12000
TEMPERATURE = "0.0"


def load_runner():
    spec = importlib.util.spec_from_file_location("runner", ROOT / "run_sensitivity_all_dimensions.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def neutral_id(pid: str, dim: str, variant: str) -> str:
    digest = hashlib.sha1(f"corrected_v2|{pid}|{dim}|{variant}".encode()).hexdigest()[:10]
    return f"r2x{digest}__openai"


def as_sentence(text: str) -> str:
    text = text.strip().rstrip(".").strip()
    return text[:1].upper() + text[1:] + "."


def plus_sentences(runner, pid: str, dim: str) -> list[str]:
    coverage = json.loads((CONTENT_DIR / f"{pid}_{dim}.json").read_text())["plus"].get("rubric_coverage") or {}
    sentences = []
    for key, _, _ in runner.DIMENSIONS[dim]["rubrics"]:
        corrected = REVIEW.get(dim, {}).get(key, {}).get(pid)
        sentences.append(as_sentence(corrected[1] if corrected else coverage[key]))
    return sentences


def build_texts(runner, pid: str, dim: str, original: str) -> dict:
    sentences = plus_sentences(runner, pid, dim)
    injection = " ".join(sentences[:4]) + "\n\n" + " ".join(sentences[4:])
    plus = runner.adaptive_inject_plus(original, injection, PLUS_MARKER)
    if plus == original + "\n\n" + injection:
        raise RuntimeError(f"{pid}/{dim}: PLUS marker not found")
    start = plus.find(injection)
    if start + len(injection) > SCORER_WINDOW:
        raise RuntimeError(f"{pid}/{dim}: PLUS text ends at {start + len(injection)}, past the scorer window")

    minus = original
    for key, finds in MINUS_REMOVE.get(pid, {}).get(dim, {}).items():
        for find in finds:
            if find not in minus:
                raise RuntimeError(f"{pid}/{dim}/{key}: removal passage not found")
            minus = minus.replace(find, MINUS_REPLACEMENT[key])
    for finds in MINUS_REMOVE.get(pid, {}).get(dim, {}).values():
        for find in finds:
            if find in minus[:SCORER_WINDOW]:
                raise RuntimeError(f"{pid}/{dim}: removal passage still visible: {find[:60]}")
    return {"base": original, "base_repeat": original, "plus": plus, "minus": minus}


def score_path(runner, dim: str, report_id: str) -> Path:
    cfg = runner.DIMENSIONS[dim]
    return ROOT / cfg["scores_dir"] / report_id / cfg["result_file"]


def run_scorer(runner, dim: str, report_id: str) -> str | None:
    """Call the unchanged scorer. Returns an error string, or None on success."""
    tool = ROOT / runner.DIMENSIONS[dim]["tool"]
    cmd = [sys.executable, str(tool), "--pid", report_id, "--temperature", TEMPERATURE, "--use_llm"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        return result.stderr.strip()[-300:]
    return None


def load_scores(runner, dim: str, report_id: str) -> dict | None:
    path = score_path(runner, dim, report_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data["meta"].get("expert_path"):
        raise RuntimeError(f"{report_id}: scored with expert JSON, expected none")
    return {k: v["score"] for k, v in data[runner.DIMENSIONS[dim]["scores_key"]].items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_only", action="store_true", help="Write and check the reports; no LLM calls.")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    runner = load_runner()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    id_map = {}
    for pid in PIDS:
        original = (REPORT_DIR / f"{pid}__openai_final_report.md").read_text(encoding="utf-8")
        for dim in DIMENSIONS:
            for variant, text in build_texts(runner, pid, dim, original).items():
                report_id = neutral_id(pid, dim, variant)
                (REPORT_DIR / f"{report_id}_final_report.md").write_text(text, encoding="utf-8")
                id_map[(pid, dim, variant)] = report_id
    print(f"[OK] wrote {len(id_map)} reports; all PLUS text inside the scorer window; all removals applied")
    if args.build_only:
        return 0

    todo = [(dim, rid) for (pid, dim, variant), rid in id_map.items() if not score_path(runner, dim, rid).exists()]
    print(f"[..] scoring {len(todo)} reports ({len(id_map) - len(todo)} already scored)")
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (dim, rid), err in zip(todo, pool.map(lambda t: run_scorer(runner, *t), todo)):
            if err:
                errors.append((rid, err))
                print(f"  [WARN] {rid}: {err}")
    if errors:
        print(f"[FAIL] {len(errors)} reports failed to score; rerun to retry them")
        return 1

    # ── Collect and judge ────────────────────────────────────────────────────
    checks = []
    for pid in PIDS:
        for dim in DIMENSIONS:
            scores = {v: load_scores(runner, dim, id_map[(pid, dim, v)]) for v in VARIANTS}
            removals = MINUS_REMOVE.get(pid, {}).get(dim, {})
            for key, label, rtype in runner.DIMENSIONS[dim]["rubrics"]:
                b = scores["base"][key]
                row = {"pid": pid, "dimension": dim, "sub_rubric": key, "label": label, "rubric_type": rtype,
                       "baseline": b, "baseline_repeat": scores["base_repeat"][key],
                       "plus": scores["plus"][key], "minus": scores["minus"][key]}
                row["plus_result"] = ("EXCLUDED" if is_excluded("PLUS", b)
                                      else "PASS" if passed("PLUS", row["plus"], b) else "FAIL")
                if not removals.get(key):
                    row["minus_result"] = "UNTESTABLE"
                elif is_excluded("MINUS", b):
                    row["minus_result"] = "EXCLUDED"
                else:
                    row["minus_result"] = "PASS" if passed("MINUS", row["minus"], b) else "FAIL"
                checks.append(row)

    (OUT_DIR / "results.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": __doc__,
        "plus_marker": PLUS_MARKER,
        "temperature": float(TEMPERATURE),
        "id_map": {f"{p}|{d}|{v}": rid for (p, d, v), rid in id_map.items()},
        "checks": checks,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────────────────────
    def rate(rows, col):
        tested = [r for r in rows if r[col] in ("PASS", "FAIL")]
        ok = sum(r[col] == "PASS" for r in tested)
        return f"{ok}/{len(tested)}" + (f" ({100 * ok / len(tested):.0f}%)" if tested else "")

    groups = defaultdict(list)
    for r in checks:
        groups[(r["dimension"], r["rubric_type"])].append(r)
        groups[("ALL", r["rubric_type"])].append(r)
        groups[(r["dimension"], "both")].append(r)
    print(f"\n  {'dimension':<12}{'type':<16}{'PLUS':>16}{'MINUS':>16}")
    for dim in DIMENSIONS + ["ALL"]:
        for rtype in ("doc-verifiable", "judgment-dep") + (("both",) if dim != "ALL" else ()):
            rows = groups[(dim, rtype)]
            print(f"  {dim:<12}{rtype:<16}{rate(rows, 'plus_result'):>16}{rate(rows, 'minus_result'):>16}")
    same = sum(r["baseline"] == r["baseline_repeat"] for r in checks)
    print(f"\n  baseline run-to-run agreement: {same}/{len(checks)} sub-rubric scores identical")
    print(f"  MINUS untestable (nothing to remove): {sum(r['minus_result'] == 'UNTESTABLE' for r in checks)}")
    print(f"  results -> {OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
