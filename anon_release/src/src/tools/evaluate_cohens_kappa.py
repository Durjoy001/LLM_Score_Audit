# -*- coding: utf-8 -*-
"""
Evaluation: human vs LLM score agreement.

All scores operate on a unified 0-1 scale:
  - Machine scores: extracted directly from final_report.md as 0-1 floats.
    If llm_scores.json exists (from --use_llm mode), its integer scores are
    normalized: (score - 1) / 4 for ordinal fields, (score - 1) / 19 for ranking.
  - Human scores: normalized from the Excel rubric scale before comparison:
    1-5 fields → (score - 1) / 4, 1-20 ranking → (score - 1) / 19.

Metrics per dimension:
  - Weighted Cohen's kappa  (both sides binned to 5-point 0-1 grid)
  - ICC(2,1)                (raw 0-1 floats)
  - Spearman rank correlation + p-value  (raw 0-1 floats)
  - 95% bootstrap CI for kappa / ICC
  - Bias: mean difference (AI - Human) + paired t-test p-value

Inputs:
  - Human Excel file (default: src/data/evaluations/human/human_scores.xlsx)
  - Machine scores from src/data/llm_scores/<pid>/llm_scores.json  (primary)
    or  src/data/reports/<pid>_final_report.md                      (fallback)

Outputs:
  - src/data/evaluations/runs/<run_id>/evaluation_summary.json
  - src/data/evaluations/runs/<run_id>/evaluation_report.xlsx
"""

import json
import math
import re
import sys
import uuid
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, ttest_rel
from openpyxl import load_workbook, Workbook

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.utils.score_extraction import extract_report_scores

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
EVAL_DIR = DATA_DIR / "evaluations"
HUMAN_DIR = EVAL_DIR / "human"
RUNS_DIR = EVAL_DIR / "runs"
LLM_SCORES_DIR = DATA_DIR / "llm_scores"
REPORT_DIR = DATA_DIR / "reports"
PROPOSALS_DIR = DATA_DIR / "proposals"

# All numeric score dimensions on 0-1 scale
ORDINAL_COLS = ["team", "objective", "strategy", "advantages", "feasibility"]
RANKING_COL = "overall_ranking"
COLUMNS = ORDINAL_COLS + [RANKING_COL]

# Kappa discretization: 5 equal bins on [0, 1] → {0.0, 0.25, 0.5, 0.75, 1.0}
_N_BINS = 5
_BIN_STEP = 1.0 / (_N_BINS - 1)


def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _bin_0_1(v: float) -> float:
    """Snap a 0-1 float to the nearest point on the 5-bin equal grid."""
    clamped = max(0.0, min(1.0, v))
    return round(round(clamped / _BIN_STEP) * _BIN_STEP, 9)


def _human_to_01(raw: int, is_ranking: bool) -> float:
    """Normalize a human integer score to 0-1."""
    if is_ranking:
        return round((raw - 1) / 19, 4)
    return round((raw - 1) / 4, 4)


def _llm_int_to_01(raw: Any, is_ranking: bool) -> Optional[float]:
    """
    Normalize a score from llm_scores.json to 0-1.

    If the value is already in [0, 1] (e.g., from an old extraction run), it is
    returned as-is. Otherwise it is assumed to be a 1-5 or 1-20 integer and
    linearly mapped to [0, 1].
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if 0.0 < v < 1.0:
        return round(v, 4)
    if is_ranking:
        return round((v - 1) / 19, 4)
    return round((v - 1) / 4, 4)


# ---------------------------------------------------------------------------
# Excel parsing helpers
# ---------------------------------------------------------------------------

def _clean_header(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = " ".join(s.replace("\n", " ").split())
    s = s.replace("/", " ").replace("(", " ").replace(")", " ")
    return s


def _normalize_name(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _stem_name(value: str) -> str:
    name = (value or "").strip()
    if not name:
        return ""
    return Path(name).stem.strip().lower()


def _normalize_key(value: str) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\.(pdf|pptx|ppt|docx|doc|txt|md)$", "", s)
    s = re.sub(r"[\(\[（]\s*\d+\s*[\)\]）]$", "", s).strip()
    s = re.sub(r"^[0-9a-z]+\s*[-_\.、:：]\s*", "", s)
    # normalize full-width/Chinese brackets and ASCII parens to spaces so that
    # human filenames with （...）/【...】/(...)  match AI keys where those chars
    # were replaced with underscores during PID generation.
    # Note: trailing (N) stripping already ran above, so () is safe to flatten here.
    s = re.sub(r"[（）【】〔〕《》〈〉「」『』｛｝\(\)]", " ", s)
    s = re.sub(r"[\s\-_\.、:：]+", " ", s)
    return s.strip()


def _detect_header_row(rows: List[List[Any]]) -> Tuple[int, Dict[str, int]]:
    header_candidates = []
    for idx, row in enumerate(rows[:10]):
        normalized = [_clean_header(c) for c in row]
        mapping: Dict[str, int] = {}
        for i, cell in enumerate(normalized):
            if not cell:
                continue
            if "file" in cell and "name" in cell:
                mapping["file_name"] = i
            elif cell.startswith("team"):
                mapping["team"] = i
            elif "objective" in cell:
                mapping["objective"] = i
            elif "strategy" in cell:
                mapping["strategy"] = i
            elif "advantage" in cell:
                mapping["advantages"] = i
            elif "feasibility" in cell:
                mapping["feasibility"] = i
            elif "overall" in cell and "ranking" in cell:
                mapping["overall_ranking"] = i
            elif "funded" in cell or "verdict" in cell or ("y" in cell and "n" in cell):
                mapping["verdict"] = i
        header_candidates.append((idx, mapping))
    header_candidates.sort(key=lambda x: len(x[1]), reverse=True)
    if not header_candidates or len(header_candidates[0][1]) < 4:
        raise RuntimeError("Could not detect header row in the human score sheet.")
    return header_candidates[0][0], header_candidates[0][1]


def _looks_like_filename(value: Any) -> bool:
    s = str(value or "").strip().lower()
    if not s:
        return False
    return bool(re.search(r"\.(pdf|pptx|ppt|docx|doc|txt|md)$", s))


def _infer_file_name_column(rows: List[List[Any]], header_row_idx: int) -> Optional[int]:
    data_rows = rows[header_row_idx + 1: header_row_idx + 21]
    if not data_rows:
        return None
    max_cols = max(len(r) for r in data_rows if r) if data_rows else 0
    best_idx, best_hits = None, 0
    for col in range(max_cols):
        hits = sum(1 for row in data_rows if col < len(row) and _looks_like_filename(row[col]))
        if hits > best_hits:
            best_hits, best_idx = hits, col
    return best_idx if best_hits >= 2 else None


def _pick_sheet(wb) -> Tuple[str, Any]:
    names = wb.sheetnames
    _log("DEBUG", f"Workbook sheets: {names}")
    active_name = wb.active.title if wb.active else ""
    for name in names:
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        non_empty = sum(
            1 for row in rows[:20] if any(c is not None and str(c).strip() for c in row)
        )
        if non_empty >= 2:
            _log("DEBUG", f"Selected sheet: {name} (non_empty_rows={non_empty})")
            return name, ws
    _log("DEBUG", "No non-empty sheet found; falling back to active sheet.")
    return active_name, wb.active


def _parse_int(value: Any) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return -1


def _parse_verdict(value: Any) -> str:
    s = str(value or "").strip().upper()
    if not s:
        return ""
    return "Y" if s.startswith("Y") else "N"


def _collect_ai_scores(provider_filter: str = "") -> Dict[str, Dict[str, Any]]:
    """
    Collect machine scores, normalized to 0-1.

    Primary source: final_report.md; scores are already continuous 0-1 floats
    from Stage 5/6 and carry the expert verdict (GO/HOLD/NO-GO).
    Fallback: llm_scores.json (written by --use_llm mode); integer scores are
    normalized via (score-1)/4 and (score-1)/19, used only when no report exists.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    provider_filter = (provider_filter or "").strip().lower()

    # --- Primary: final_report.md ---
    if REPORT_DIR.exists():
        report_files = 0
        for report_path in REPORT_DIR.glob("*_final_report.md"):
            pid = report_path.name.replace("_final_report.md", "")
            pid_base = pid.rsplit("_", 1)[0] if "_" in pid else pid
            try:
                report_text = report_path.read_text(encoding="utf-8")
                scores = extract_report_scores(report_text)
            except Exception:
                continue
            report_files += 1

            # Enrich with meta.json if available
            original, stem = "", ""
            meta_path = PROPOSALS_DIR / f"{pid}.meta.json"
            if meta_path.exists():
                try:
                    meta = read_json(meta_path)
                    original = meta.get("original_filename", "")
                    stem = meta.get("original_stem", "")
                except Exception:
                    pass

            record = {
                "pid": pid,
                "pid_base": pid_base,
                "source_filename": original,
                "source_stem": stem,
                "provider": "report",
                "scores_path": str(report_path),
                "scores": scores,
            }
            for key in {_normalize_name(original), _normalize_key(original),
                        _normalize_key(stem), stem, pid, pid_base}:
                if key:
                    mapping[key] = record
        _log("DEBUG", f"Report files loaded: {report_files}")

    # --- Fallback: llm_scores.json (only when no report found for this pid) ---
    if LLM_SCORES_DIR.exists():
        score_files = 0
        score_paths = list(LLM_SCORES_DIR.glob("*/llm_scores.json"))
        score_paths += list(LLM_SCORES_DIR.glob("*/*/llm_scores.json"))
        for score_path in score_paths:
            score_files += 1
            try:
                data = read_json(score_path)
            except Exception:
                continue
            meta = data.get("meta", {}) or {}
            raw_scores = data.get("scores", {}) or {}
            provider = (meta.get("provider") or "").strip().lower()
            if provider_filter and provider != provider_filter:
                continue
            pid = meta.get("pid") or score_path.parent.name
            # Skip if a report already covers this pid
            if pid in mapping:
                continue
            pid_base = pid.rsplit("_", 1)[0] if "_" in pid else pid
            original = meta.get("source_filename") or ""
            stem = meta.get("source_filename_stem") or _stem_name(original)

            # Normalize integer scores to 0-1
            scores: Dict[str, Any] = {}
            for col in ORDINAL_COLS:
                v = _llm_int_to_01(raw_scores.get(col), is_ranking=False)
                if v is not None:
                    scores[col] = v
            v = _llm_int_to_01(raw_scores.get(RANKING_COL), is_ranking=True)
            if v is not None:
                scores[RANKING_COL] = v
            scores["verdict"] = raw_scores.get("verdict", "N")

            record = {
                "pid": pid,
                "pid_base": pid_base,
                "source_filename": original,
                "source_stem": stem,
                "provider": provider,
                "scores_path": str(score_path),
                "scores": scores,
            }
            for key in {_normalize_name(original), _normalize_key(original),
                        _normalize_key(stem), stem, pid, pid_base}:
                if key and key not in mapping:  # never overwrite a report-sourced entry
                    mapping[key] = record
        _log("DEBUG", f"LLM score fallback files loaded: {score_files}")

    return mapping


# A substring-match key must be at least this many characters (after stripping any
# file extension) to be trusted. This blocks degenerate keys such as a bare "4"
# produced by a source document literally named "4.pdf" (a Kickstarter proposal),
# which would otherwise substring-match almost any human filename containing a "4".
_MIN_FUZZY_KEY_LEN = 4


def _is_degenerate_key(key: str) -> bool:
    """True if a key is too short or numerically trivial to be safe for substring
    matching (e.g. '4' or '4.pdf'). Such keys are only ever used for EXACT matches,
    never for the fuzzy fallback."""
    core = re.sub(r"\.(pdf|pptx|ppt|docx|doc|txt|md)$", "", str(key or "")).strip()
    return len(core) < _MIN_FUZZY_KEY_LEN or core.isdigit()


def _ai_map_from_canonical(json_path: Path) -> Dict[str, Dict[str, Any]]:
    """Build the AI score map from a frozen canonical scores JSON instead of
    scanning src/data/reports (which is gitignored and known to drift).

    Expects the schema written by build_canonical_ai_scores.py:
        {"canonical": {"<pid>": {team, objective, strategy, advantages,
                                 feasibility, overall_ranking, verdict,
                                 confidence, file_name, ...}, ...}}
    All scores are already on the 0-1 scale; the recorded verdict is the
    published OR-logic verdict. Keys are the same normalized filename / pid
    forms _find_record() looks up.
    """
    data = read_json(json_path).get("canonical", {})
    mapping: Dict[str, Dict[str, Any]] = {}
    for pid, rec in data.items():
        scores: Dict[str, Any] = {c: rec.get(c) for c in COLUMNS}
        scores["verdict"] = rec.get("verdict", "N")
        fn = rec.get("file_name", "") or ""
        record = {
            "pid": pid,
            "pid_base": pid,
            "source_filename": fn,
            "source_stem": _stem_name(fn),
            "provider": "canonical",
            "scores_path": str(json_path),
            "scores": scores,
        }
        for key in {_normalize_name(fn), _normalize_key(fn), _stem_name(fn), pid}:
            if key:
                mapping[key] = record
    _log("DEBUG", f"Canonical AI scores loaded: {len(data)} proposals from {json_path}")
    return mapping


def _find_record(ai_map: Dict[str, Dict[str, Any]], human_raw: str) -> Optional[Dict[str, Any]]:
    """Resolve a human filename to an AI record.

    Priority order:
      1. EXACT match on any normalized form (these include the meta-derived
         original_filename / stem keys, so meta-based identity always wins first).
      2. Guarded substring fallback: degenerate keys (see _is_degenerate_key) are
         skipped, the most specific (longest-overlap) candidate is chosen for
         determinism, and a WARNING is logged so the fuzzy path is never silent.
    """
    # 1) exact match — highest priority, covers meta-derived keys
    for key in (_normalize_name(human_raw), _stem_name(human_raw), _normalize_key(human_raw)):
        if key and key in ai_map:
            return ai_map[key]

    # 2) guarded substring fallback
    human_key = _normalize_key(human_raw)
    if human_key and not _is_degenerate_key(human_key):
        best: Optional[Tuple[int, str, Dict[str, Any]]] = None
        for key, record in ai_map.items():
            if _is_degenerate_key(key):
                continue
            if human_key in key or key in human_key:
                specificity = min(len(human_key), len(key))
                if best is None or specificity > best[0]:
                    best = (specificity, key, record)
        if best is not None:
            _log("WARN", f"fuzzy fallback matched human file {human_raw!r} -> "
                         f"pid={best[2].get('pid')!r} via key {best[1]!r} "
                         f"(no exact match found)")
            return best[2]
    return None


# ---------------------------------------------------------------------------
# Statistical metrics — all operate on 0-1 floats
# ---------------------------------------------------------------------------

def _weighted_kappa_01(pairs: List[Tuple[float, float]]) -> float:
    """Quadratic-weighted Cohen's kappa on 0-1 scores discretized to 5 bins."""
    if len(pairs) < 2:
        return 0.0
    nc = _N_BINS
    W = np.array(
        [[1.0 - (i - j) ** 2 / (nc - 1) ** 2 for j in range(nc)] for i in range(nc)]
    )

    O = np.zeros((nc, nc))
    valid = 0
    for h, a in pairs:
        hi_idx = int(_bin_0_1(h) / _BIN_STEP + 0.5)
        ai_idx = int(_bin_0_1(a) / _BIN_STEP + 0.5)
        if 0 <= hi_idx < nc and 0 <= ai_idx < nc:
            O[hi_idx, ai_idx] += 1
            valid += 1
    if valid == 0:
        return 0.0
    O /= valid

    row_sum = O.sum(axis=1)
    col_sum = O.sum(axis=0)
    E = np.outer(row_sum, col_sum)

    num = float((W * O).sum())
    den = float((W * E).sum())
    if abs(1.0 - den) < 1e-12:
        return 0.0
    return round(float(np.clip((num - den) / (1.0 - den), -1.0, 1.0)), 4)


def _unweighted_kappa_01(pairs: List[Tuple[float, float]]) -> float:
    """Unweighted kappa on 0-1 binned scores."""
    if not pairs:
        return 0.0
    n = len(pairs)
    counts_h: Dict[float, int] = {}
    counts_a: Dict[float, int] = {}
    agree = 0
    for h, a in pairs:
        hs, as_ = _bin_0_1(h), _bin_0_1(a)
        counts_h[hs] = counts_h.get(hs, 0) + 1
        counts_a[as_] = counts_a.get(as_, 0) + 1
        if abs(hs - as_) < 1e-9:
            agree += 1
    po = agree / n
    pe = sum(
        (counts_h.get(c, 0) / n) * (counts_a.get(c, 0) / n)
        for c in set(counts_h) | set(counts_a)
    )
    if 1.0 - pe <= 1e-9:
        return 0.0
    return round((po - pe) / (1.0 - pe), 4)


def _icc(pairs: List[Tuple[float, float]]) -> float:
    """ICC(2,1): two-way mixed, single measures, absolute agreement."""
    n = len(pairs)
    if n < 3:
        return 0.0
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    data = np.column_stack([a, b])
    k = 2
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    SS_total = np.sum((data - grand_mean) ** 2)
    SS_rows = k * np.sum((row_means - grand_mean) ** 2)
    SS_cols = n * np.sum((col_means - grand_mean) ** 2)
    SS_error = SS_total - SS_rows - SS_cols
    if n <= 1:
        return 0.0
    MS_rows = SS_rows / (n - 1)
    MS_cols = SS_cols / (k - 1)
    MS_error = SS_error / ((n - 1) * (k - 1))
    denom = MS_rows + (k - 1) * MS_error + k * (MS_cols - MS_error) / n
    if abs(denom) < 1e-12:
        return 0.0
    return round(float(np.clip((MS_rows - MS_error) / denom, -1.0, 1.0)), 4)


def _spearman(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    if len(pairs) < 3:
        return 0.0, 1.0
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    rho, pval = spearmanr(a, b)
    rho = round(float(rho), 4)
    pval = float(pval)
    # scipy uses a t-distribution approximation: when |r|=1, t→±∞ and p→0,
    # which is wrong for small n. The exact minimum two-tailed p is 2/n!
    n = len(pairs)
    if n <= 20 and abs(rho) >= 1.0 - 1e-9:
        pval = max(pval, 2.0 / math.factorial(n))
    return rho, round(pval, 4)


def _bootstrap_ci(
    pairs: List[Tuple[float, float]],
    stat_fn,
    n_boot: int = 1000,
) -> Tuple[Optional[float], Optional[float]]:
    n = len(pairs)
    if n < 3:
        return None, None
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [pairs[int(i)] for i in idx]
        try:
            boots.append(stat_fn(sample))
        except Exception:
            pass
    if len(boots) < 20:
        return None, None
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return round(lo, 4), round(hi, 4)


def _bias_stats(pairs: List[Tuple[float, float]]) -> Tuple[float, Optional[float]]:
    if len(pairs) < 2:
        return 0.0, None
    h_vals = [p[0] for p in pairs]
    a_vals = [p[1] for p in pairs]
    mean_diff = round(float(np.mean(np.array(a_vals, float) - np.array(h_vals, float))), 4)
    if len(pairs) < 3 or len(set(a_vals)) == 1 or len(set(h_vals)) == 1:
        return mean_diff, None
    try:
        _, pval = ttest_rel(a_vals, h_vals)
        return mean_diff, round(float(pval), 4)
    except Exception:
        return mean_diff, None


def _rank_order_metrics(items: List[Dict[str, Any]], field: str = RANKING_COL) -> Dict[str, Any]:
    """Pairwise rank-order concordance on 0-1 float scores."""
    valid = []
    for item in items:
        hv = item.get(f"human_{field}")
        av = item.get(f"ai_{field}")
        try:
            hf, af = float(hv), float(av)
            valid.append({
                "file_name": item.get("file_name", ""),
                "pid": item.get("pid", ""),
                "human": hf,
                "ai": af,
            })
        except (TypeError, ValueError):
            continue

    comparable = 0
    concordant = 0
    discordant = 0
    human_ties = 0
    ai_ties_when_human_ordered = 0
    examples = []

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            human_delta = a["human"] - b["human"]
            ai_delta = a["ai"] - b["ai"]
            if abs(human_delta) < 1e-9:
                human_ties += 1
                continue
            comparable += 1
            if abs(ai_delta) < 1e-9:
                ai_ties_when_human_ordered += 1
                discordant += 1
                ok = False
            else:
                ok = (human_delta > 0 and ai_delta > 0) or (human_delta < 0 and ai_delta < 0)
                if ok:
                    concordant += 1
                else:
                    discordant += 1
            if not ok and len(examples) < 10:
                examples.append({
                    "a": a["file_name"],
                    "b": b["file_name"],
                    "human_a": a["human"],
                    "human_b": b["human"],
                    "ai_a": a["ai"],
                    "ai_b": b["ai"],
                })

    rate = round(concordant / comparable, 4) if comparable else None
    return {
        "field": field,
        "n_items": len(valid),
        "pairwise_comparable": comparable,
        "pairwise_concordant": concordant,
        "pairwise_discordant": discordant,
        "human_tie_pairs_excluded": human_ties,
        "ai_ties_when_human_ordered": ai_ties_when_human_ordered,
        "pairwise_rank_concordance": rate,
        "discordant_examples": examples,
    }


def _column_metrics(
    pairs: List[Tuple[float, float]],
    is_ranking: bool = False,
) -> Dict[str, Any]:
    """Compute all statistical metrics for one score column (0-1 scale)."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    rho, spearman_p = _spearman(pairs)
    spearman_ci_lo, spearman_ci_hi = _bootstrap_ci(pairs, lambda p: _spearman(p)[0])
    mean_diff, bias_p = _bias_stats(pairs)

    if is_ranking:
        primary_val = _icc(pairs)
        ci_lo, ci_hi = _bootstrap_ci(pairs, _icc)
        result: Dict[str, Any] = {
            "n": n,
            "metric_type": "continuous_0_1_icc",
            "icc": primary_val,
            "icc_ci95": [ci_lo, ci_hi],
            "weighted_kappa": _weighted_kappa_01(pairs),
        }
    else:
        primary_val = _weighted_kappa_01(pairs)
        ci_lo, ci_hi = _bootstrap_ci(pairs, _weighted_kappa_01)
        result = {
            "n": n,
            "metric_type": "continuous_0_1_weighted_kappa",
            "weighted_kappa": primary_val,
            "weighted_kappa_ci95": [ci_lo, ci_hi],
            "unweighted_kappa": _unweighted_kappa_01(pairs),
        }

    result["spearman_r"] = rho
    result["spearman_p"] = spearman_p
    result["spearman_ci95"] = [spearman_ci_lo, spearman_ci_hi]
    result["mean_diff"] = mean_diff
    result["bias_p"] = bias_p

    h_vals = [p[0] for p in pairs]
    a_vals = [p[1] for p in pairs]
    result["mean_human"] = round(float(np.mean(h_vals)), 4)
    result["mean_ai"] = round(float(np.mean(a_vals)), 4)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate human vs AI score agreement.")
    ap.add_argument("--human_xlsx", type=str, default=str(HUMAN_DIR / "human_scores.xlsx"),
                    help="Human ground-truth Excel file (single-rater or consensus).")
    ap.add_argument("--ai_scores_json", type=str, default="",
                    help="Frozen canonical AI scores JSON (e.g. canonical/canonical_ai_scores_dataset1.json). "
                         "When set, AI scores are read from it instead of scanning the gitignored src/data cache, "
                         "making the run reproducible against the pinned canonical set.")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--provider", type=str, default="", help="Optional provider filter (ignored with --ai_scores_json).")
    args = ap.parse_args()

    human_path = Path(args.human_xlsx)
    if not human_path.exists():
        raise FileNotFoundError(f"Human Excel file not found: {human_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = Path(args.out_dir) if args.out_dir else (RUNS_DIR / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    wb = load_workbook(human_path, data_only=True)
    _, ws = _pick_sheet(wb)

    rows = list(ws.iter_rows(values_only=True))
    _log("DEBUG", f"Total rows read: {len(rows)}")
    header_row_idx, headers = _detect_header_row(rows)
    _log("DEBUG", f"Header row index: {header_row_idx}, mapping: {headers}")

    if "file_name" not in headers:
        inferred = _infer_file_name_column(rows, header_row_idx)
        if inferred is not None:
            headers["file_name"] = inferred
            _log("DEBUG", f"Inferred file_name column index: {inferred}")

    data_rows = rows[header_row_idx + 1:]
    if args.ai_scores_json:
        ai_scores_path = Path(args.ai_scores_json)
        if not ai_scores_path.exists():
            raise FileNotFoundError(f"Canonical AI scores JSON not found: {ai_scores_path}")
        ai_map = _ai_map_from_canonical(ai_scores_path)
    else:
        ai_map = _collect_ai_scores(args.provider)

    per_item: List[Dict[str, Any]] = []
    pairs_by_col: Dict[str, List[Tuple[float, float]]] = {c: [] for c in COLUMNS}
    verdict_pairs: List[Tuple[str, str]] = []
    matched = 0
    skipped = 0
    unmatched_logged = 0

    for row in data_rows:
        file_idx = headers.get("file_name")
        if file_idx is None:
            continue
        raw_file = str(row[file_idx] or "").strip()
        if not raw_file:
            continue

        record = _find_record(ai_map, raw_file)

        if not record:
            skipped += 1
            if unmatched_logged < 5:
                _log("DEBUG", f"No AI match for file_name={raw_file!r}")
                unmatched_logged += 1
            per_item.append({"file_name": raw_file, "notes": "no_ai_match"})
            continue

        matched += 1
        scores = record.get("scores", {}) or {}
        verdict_h = _parse_verdict(row[headers["verdict"]] if "verdict" in headers and headers["verdict"] < len(row) else "")
        verdict_a = _parse_verdict(scores.get("verdict"))

        row_out: Dict[str, Any] = {
            "file_name": raw_file,
            "pid": record.get("pid"),
            "provider": record.get("provider", ""),
            "verdict_human": verdict_h,
            "verdict_ai": verdict_a,
            "verdict_match": "Y" if verdict_h and verdict_h == verdict_a else "N",
            "notes": "",
        }

        for col in COLUMNS:
            is_ranking = col == RANKING_COL
            raw_hv = _parse_int(row[headers[col]] if col in headers and headers[col] < len(row) else "")
            hv = _human_to_01(raw_hv, is_ranking) if raw_hv >= 1 else None

            raw_av = scores.get(col)
            try:
                av = round(float(raw_av), 4) if raw_av is not None else None
            except (TypeError, ValueError):
                av = None

            row_out[f"human_{col}"] = hv if hv is not None else ""
            row_out[f"ai_{col}"] = av if av is not None else ""
            diff = round(av - hv, 4) if (hv is not None and av is not None) else ""
            row_out[f"diff_{col}"] = diff
            if hv is not None and av is not None:
                pairs_by_col[col].append((hv, av))

        if verdict_h and verdict_a:
            verdict_pairs.append((verdict_h, verdict_a))

        per_item.append(row_out)

    # Compute metrics
    metrics: Dict[str, Any] = {}
    for col in ORDINAL_COLS:
        metrics[col] = _column_metrics(pairs_by_col[col], is_ranking=False)
    metrics[RANKING_COL] = _column_metrics(pairs_by_col[RANKING_COL], is_ranking=True)

    verdict_accuracy = 0.0
    if verdict_pairs:
        verdict_accuracy = round(sum(1 for h, a in verdict_pairs if h == a) / len(verdict_pairs), 4)

    kappa_compat: Dict[str, float] = {}
    for col in ORDINAL_COLS:
        kappa_compat[col] = metrics[col].get("weighted_kappa", 0.0)
    kappa_compat[RANKING_COL] = metrics[RANKING_COL].get("icc", 0.0)

    rank_order = _rank_order_metrics(per_item, RANKING_COL)

    summary = {
        "run_id": run_id,
        "generated_at": now_str(),
        "human_file": str(human_path),
        "provider_filter": args.provider,
        "score_scale": "0-1 (human scores normalized; kappa uses 5-bin grid)",
        "matched_rows": matched,
        "skipped_rows": skipped,
        "verdict_accuracy": verdict_accuracy,
        "metrics": metrics,
        "rank_order": rank_order,
        "kappa": kappa_compat,
    }
    write_json(out_dir / "evaluation_summary.json", summary)

    # ----- Excel report -----
    out_wb = Workbook()

    ws_sum = out_wb.active
    ws_sum.title = "summary"
    for row in [
        ["metric", "value"],
        ["run_id", run_id],
        ["generated_at", summary["generated_at"]],
        ["human_file", summary["human_file"]],
        ["score_scale", summary["score_scale"]],
        ["provider_filter", args.provider or "<any>"],
        ["matched_rows", matched],
        ["skipped_rows", skipped],
        ["verdict_accuracy", verdict_accuracy],
        ["rank_pairwise_concordance", rank_order.get("pairwise_rank_concordance")],
        ["rank_pairwise_comparable", rank_order.get("pairwise_comparable")],
    ]:
        ws_sum.append(row)

    ws_met = out_wb.create_sheet("metrics")
    ws_met.append([
        "column", "n", "metric_type",
        "weighted_kappa", "wk_ci_lo", "wk_ci_hi",
        "unweighted_kappa",
        "icc", "icc_ci_lo", "icc_ci_hi",
        "spearman_r", "spearman_p", "sp_ci_lo", "sp_ci_hi",
        "mean_human", "mean_ai", "mean_diff", "bias_p",
    ])
    for col in COLUMNS:
        m = metrics.get(col, {})
        wk_ci = m.get("weighted_kappa_ci95") or [None, None]
        icc_ci = m.get("icc_ci95") or [None, None]
        sp_ci = m.get("spearman_ci95") or [None, None]
        ws_met.append([
            col,
            m.get("n"),
            m.get("metric_type"),
            m.get("weighted_kappa"),
            wk_ci[0],
            wk_ci[1],
            m.get("unweighted_kappa"),
            m.get("icc"),
            icc_ci[0],
            icc_ci[1],
            m.get("spearman_r"),
            m.get("spearman_p"),
            sp_ci[0],
            sp_ci[1],
            m.get("mean_human"),
            m.get("mean_ai"),
            m.get("mean_diff"),
            m.get("bias_p"),
        ])

    ws_bias = out_wb.create_sheet("bias")
    ws_bias.append(["column", "mean_human (0-1)", "mean_ai (0-1)", "mean_diff (AI-Human)", "bias_p", "interpretation"])
    for col in COLUMNS:
        m = metrics.get(col, {})
        diff = m.get("mean_diff", 0.0) or 0.0
        pval = m.get("bias_p")
        if pval is None:
            interp = "n/a"
        elif pval < 0.05:
            interp = "AI overrates" if diff > 0 else "AI underrates"
        else:
            interp = "no significant bias"
        ws_bias.append([col, m.get("mean_human"), m.get("mean_ai"), diff, pval, interp])

    ws_rank = out_wb.create_sheet("rank_order")
    ws_rank.append(["metric", "value"])
    for key, value in rank_order.items():
        if key == "discordant_examples":
            continue
        ws_rank.append([key, value])
    ws_rank.append([])
    ws_rank.append(["discordant_examples"])
    ws_rank.append(["proposal_a", "proposal_b", "human_a", "human_b", "ai_a", "ai_b"])
    for ex in rank_order.get("discordant_examples", []):
        ws_rank.append([ex.get("a"), ex.get("b"), ex.get("human_a"), ex.get("human_b"), ex.get("ai_a"), ex.get("ai_b")])

    ws_items = out_wb.create_sheet("per_item")
    item_header = ["file_name", "pid", "provider", "verdict_human", "verdict_ai", "verdict_match"]
    for col in COLUMNS:
        item_header += [f"human_{col}", f"ai_{col}", f"diff_{col}"]
    item_header.append("notes")
    ws_items.append(item_header)
    for item in per_item:
        row_vals = [
            item.get("file_name", ""),
            item.get("pid", ""),
            item.get("provider", ""),
            item.get("verdict_human", ""),
            item.get("verdict_ai", ""),
            item.get("verdict_match", ""),
        ]
        for col in COLUMNS:
            row_vals += [item.get(f"human_{col}", ""), item.get(f"ai_{col}", ""), item.get(f"diff_{col}", "")]
        row_vals.append(item.get("notes", ""))
        ws_items.append(row_vals)

    report_path = out_dir / "evaluation_report.xlsx"
    out_wb.save(report_path)
    _log("OK", f"summary written: {out_dir / 'evaluation_summary.json'}")
    _log("OK", f"report written: {report_path}")


if __name__ == "__main__":
    main()
