# -*- coding: utf-8 -*-
"""
Stage 5 - Structured candidate post-processing (v3.2, no web hard gates)

Input:
  - Expected schema: "llm_answering.v2" or "refined_items.v2.proposal_aware_with_general_insights"
  - Top-level shape: {"meta": {...}, "items": [ {dimension, q_index, question, candidates: [...]}, ... ]}

Outputs:
  - src/data/refined_answers/<pid>/postproc/metrics.json
  - src/data/refined_answers/<pid>/postproc/selected_by_question.json
  - src/data/refined_answers/<pid>/postproc/final_payload.json
  - src/data/refined_answers/<pid>/postproc/report.md
  - src/data/refined_answers/<pid>/postproc/drops_debug.json
"""

import os
import re
import json
import math
import string
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from functools import lru_cache
import re as _re

# ============================ Paths and defaults ============================

ROOT = Path(__file__).resolve().parents[1]  # .../src
DATA_DIR = ROOT / "data"
REFINED_ROOT = DATA_DIR / "refined_answers"
PROGRESS_FILE = DATA_DIR / "step_progress.json"


def _write_progress(done: int, total: int, pid: str = "") -> None:
    try:
        path = PROGRESS_FILE.parent / f"step_progress_{pid}.json" if pid else PROGRESS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"done": done, "total": total}), encoding="utf-8")
    except Exception:
        pass
CONF_DIR = DATA_DIR / "config" / "postproc"
QS_CONF_PATH = DATA_DIR / "config" / "question_sets" / "generated_questions.json"

# Authority vocabulary aligned with llm_answering. "hc" is intentionally excluded
# to avoid false positives for Health Canada; use "health canada" instead.
AUTHORITY_TOKENS = [
    "fda","ema","ich q8","ich q9","ich q10","21 cfr part 11",
    "iso 13485","iso 14971","iso 27001","who","gamp5","gamp 5", "pic/s",
    "clinicaltrials.gov","eudract","nct","doi","orcid","pubmed","scopus",
    "uspto","epo","cnipa",
    # Jurisdictions and abbreviations.
    "nmpa","cfda","mhra","pmda","tga","health canada","nice",
    "eudralex","mdr","ivdr",
    "iec 62304","iec 62366","iso 62304","iso 62366",
    "gcp","gmp","glp","gxp",
    # Privacy, security, and compliance.
    "gdpr","hipaa","phipa","pipeda","nist","soc 2","iso 27017","iso 27018"
]

COVERAGE_BANK = {
    "regulatory": ["fda","ema","ich","21 cfr","iso","pic/s","gmp","gcp","glp","gxp",
                   "nmpa","mhra","pmda","tga","health canada","mdr","ivdr",
                   "gdpr","pipeda","phipa","hipaa","nist","soc 2","iso 27017","iso 27018"],
    "trial": ["clinicaltrials.gov","eudract","nct"],
    "publication": ["pubmed","scopus","doi","orcid"],
    "patent": ["uspto","epo","cnipa","wo"],
    "repo": ["github","gitlab","model card","data card"]
}

# === Alias and tokenizer helpers ===
# Compressed strings mapped to canonical spacing, e.g. 21CFRPart11 / ISO13485 / ICHQ10.
ALIAS_MAP = {
    # Regulatory and standards normalization.
    "21cfrpart11": "21 cfr part 11",
    "21cfr11": "21 cfr part 11",
    "iso13485": "iso 13485",
    "iso14971": "iso 14971",
    "iso27001": "iso 27001",
    "iso27017": "iso 27017",
    "iso27018": "iso 27018",
    "iec62304": "iec 62304",
    "iec62366": "iec 62366",
    "gamp5": "gamp 5",
    "ichq8": "ich q8",
    "ichq9": "ich q9",
    "ichq10": "ich q10",
    "pics": "pic/s",

    # Site/domain normalization.
    "clinicaltrialsgov": "clinicaltrials gov",
}

# Alignment weighting: authority tokens receive more weight.
AUTHORITY_KEYWORDS = {
    "fda": 2.0, "ema": 2.0, "ich": 2.0, "q8": 1.4, "q9": 1.4, "q10": 1.6,
    "iso": 2.0, "13485": 2.0, "14971": 2.0, "27001": 1.6, "27017": 1.4, "27018": 1.4,
    "21": 1.1, "cfr": 2.0, "part": 1.1, "11": 1.1,
    "clinicaltrials": 2.0, "gov": 1.0, "eudract": 2.0,
    "pubmed": 2.0, "scopus": 2.0, "orcid": 1.5,
    "uspto": 2.0, "epo": 2.0, "cnipa": 2.0,
    "gdpr": 2.0, "hipaa": 2.0, "phipa": 2.0, "pipeda": 2.0,
    "gxp": 1.4, "gmp": 1.4, "glp": 1.4, "gcp": 1.4, "doi": 1.4,
}

STOPWORDS_ALIGN = {
    "the","and","of","to","for","in","on","by","with","a","an","is","are",
}

_RE_NON_ALNUM = _re.compile(r"[^a-z0-9]+", _re.IGNORECASE)

def _apply_aliases(s: str) -> str:
    """Normalize spacing/punctuation, split digit-letter boundaries, and expand aliases."""
    if not s:
        return ""
    base = str(s).lower()
    base = base.replace("\u00a0", " ").replace("\u3000", " ")

    # Build a compact punctuation-free string for alias matching.
    compact = _RE_NON_ALNUM.sub("", base)
    if compact in ALIAS_MAP:
        base = ALIAS_MAP[compact]

    # Add spaces at digit-letter boundaries: 21CFRPart11 -> 21 CFR Part 11.
    base = _re.sub(r"([a-z])([0-9])", r"\1 \2", base)
    base = _re.sub(r"([0-9])([a-z])", r"\1 \2", base)

    # Fallback for ICHQ10 -> ich q10.
    base = _re.sub(r"\b(ich)\s*q\s*(\d+)\b", r"\1 q\2", base)
    # clinicaltrials.gov -> clinicaltrials gov
    base = base.replace("clinicaltrials.gov", "clinicaltrials gov")
    return base

def _tokens_for_alignment(s: str):
    r"""Extract [a-z]+ and digit tokens, then construct bigrams."""
    if not s:
        return set(), set()
    s = _apply_aliases(s)

    # Replace non-alphanumeric characters with spaces.
    s_en = _RE_NON_ALNUM.sub(" ", s)

    # English / numeric tokens.
    toks_en = [t for t in _re.findall(r"[a-z]+|\d+", s_en) if t not in STOPWORDS_ALIGN]

    toks = toks_en

    unigrams = set(toks)
    bigrams = set()
    for i in range(len(toks)-1):
        bigrams.add(toks[i] + " " + toks[i+1])
    return unigrams, bigrams


def _weighted_overlap(q_uni, q_bi, c_uni, c_bi):
    """Weighted overlap between hint/question tokens and candidate tokens, with a small bigram bonus."""
    if not q_uni:
        return 0.0

    def w(t):
        if t in AUTHORITY_KEYWORDS:
            return AUTHORITY_KEYWORDS[t]
        return 1.0

    den = sum(w(t) for t in q_uni)
    hit = sum(w(t) for t in (q_uni & c_uni))
    bonus = 0.0
    if q_bi and c_bi:
        bi_hit = len(q_bi & c_bi)
        bonus = min(0.30, 0.10 * bi_hit)
    return min(1.0, (hit / max(1e-9, den)) * (1.0 + bonus))
DEFAULT_CONF = {
    # Per-question normalization parameters.
    "length_ref_chars": 280,
    "claims_ref": 3,
    "evidence_ref": 4,
    "jaccard_threshold": 0.30,

    # Weight configuration. Evidence signals are active; weights sum to 1.0.
    "consistency_weight": 0.10,
    "fields_weight": {
        "length": 0.16,
        "claims": 0.20,
        "evidence_count": 0.10,
        "evidence_authority": 0.02,
        "evidence_coverage": 0.02,
        "structure": 0.17,
        "alignment": 0.28,
        "calibrated_confidence": 0.05
    },

    # Dimension weights.
    "dimension_weight": {
        "team": 1.00,
        "objectives": 1.00,
        "strategy": 1.00,
        "innovation": 1.10,
        "feasibility": 1.20
    },

    # Penalties.
    "penalties": {
        "contradiction": 0.05,
        "overclaim": 0.03,
        "understructure": 0.03,
        "redline_residual": 0.06,
        "dimension_drift": 0.04
    },

    # Filtering thresholds. Authority/coverage thresholds are retained for compatibility only.
    "filters": {
        # Evidence-related fields are diagnostic only.
        "min_evidence_count": 0,

        # Basic structure requirements.
        "min_bullet_lines": 3,
        "min_median_bullet_len": 6,
        "min_structured_score": 0.05,
        "soft_window": True,

        # Alignment is used only to remove severely off-topic answers.
        "min_alignment_for_keep": 0.10,

        # Compatibility fields, no longer active filters.
        "min_auth_hits_for_keep": 0,
        "min_coverage_bins_for_keep": 0,
        "min_authority_ratio": 0.0,
        "min_coverage_ratio": 0.0,
        "dyn_align_relax_trigger": 0.0,
        "dyn_align_relax_delta": 0.0
    },

    # Consistency correction parameters.
    "consistency_correction": {
        "k_contradiction": 0.15,
        "beta": {
            "jaccard_sweetspot": [0.25, 0.65],
            "provider_calibration": {
                "deepseek": {"beta_delta": -0.01},
                "openai":   {"beta_delta":  0.00},
                "default":  {"beta_delta":  0.00}
            }
        }
    },

    # Dimension-specific filtering parameters.
    "dimension_specific": {
        "objectives": {"min_bullet_lines": 3, "min_alignment_for_keep": 0.18},
        "strategy":   {"min_bullet_lines": 3, "min_alignment_for_keep": 0.24},
        "innovation": {"min_bullet_lines": 3, "min_alignment_for_keep": 0.18},
        "feasibility":{"min_bullet_lines": 3, "min_alignment_for_keep": 0.12}
    },

    # Report output.
    "bar_symbols": 20,
    "adv_topk": 5,
    "report_width": 92,
    "unknown_warn_ratio": 0.10
}

PUNC = set(string.punctuation)
DIM_ORDER = ["team", "objectives", "strategy", "innovation", "feasibility"]

RE_DATE   = re.compile(r"\b(20\d{2}|19\d{2})([-/.])\d{1,2}([-/\.])\d{1,2}\b|\b(Q[1-4]\s*-\s*20\d{2})\b", re.I)
RE_MONEY  = re.compile(r"\b(\$|USD|EUR|CNY|RMB|CAD)\s*\d{2,}(,\d{3})*(\.\d+)?\b|\b\d+(\.\d+)?\s*(million|billion)\b", re.I)
RE_TRIAL  = re.compile(r"\bNCT\d{8}\b|\bEUCTR-\d{4}-\d{6}-\d{2}\b", re.I)
RE_DOI    = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
RE_PATENT = re.compile(r"\b(US|EP|CN)\d{5,}\b|\bWO\d{7,}\b", re.I)
RE_ISO    = re.compile(r"\b(ISO|IEC)\s?\d{4,5}(-\d+)?\b", re.I)
RE_STDNUM = re.compile(r"\bEN\s?\d{3,5}\b|\bASTM\s?[A-Z]?\d{2,5}\b", re.I)
RE_ID_ANY = re.compile(r"\b(registration|approval|filing)\s+(number|id)\b", re.I)

# ====== Line cleanup and soft joining ======

DOMAIN_FIXES = [
    (re.compile(r"clinicaltrials\s*[\.\-]?\s*(\d+\s*[\.\-]?\s*)?gov", re.I), "ClinicalTrials.gov"),
    (re.compile(r"eudract\s*[\.\-]?\s*eu", re.I), "EudraCT EU"),
    (re.compile(r"pubmed\s*[\.\-]?\s*ncbi\s*[\.\-]?\s*nlm\s*[\.\-]?\s*nih\s*[\.\-]?\s*gov", re.I), "PubMed"),
]
BULLET_NORM = re.compile(r"^\s*(?:[-*]|(\d+)[\.\)])\s*")

def _soft_join(text: str) -> str:
    if not text:
        return ""
    s = text
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"([A-Za-z0-9])\n([A-Za-z0-9])", r"\1 \2", s)
    s = re.sub(r"\s*\.\s*(gov|com|org|net|io)\b", r".\1", s, flags=re.I)
    for pat, rep in DOMAIN_FIXES:
        s = pat.sub(rep, s)
    return s

def _normalize_bullets(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out = []
    for ln in lines:
        base = BULLET_NORM.sub("", ln).strip()
        if not base:
            continue
        out.append(base)
    return "\n".join(out)

# ========= Two cleanup modes =========

def sanitize_for_scoring(raw: str) -> str:
    s = raw or ""
    s = _soft_join(s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def sanitize_for_display(raw: str) -> str:
    s = raw or ""
    s = _soft_join(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\b([A-Za-z])(?:\s+[A-Za-z]){1,3}\b",
               lambda m: m.group(0).replace(" ", ""), s)
    s = _normalize_bullets(s)
    return s.strip()

def sanitize_answer(raw: str) -> str:
    return sanitize_for_display(raw)

# ====== Placeholder-line detection ======
PLACEHOLDER_LINE = re.compile(
    r"""^(
        [\-=~\._]{2,}$                         |
        [\(\)\[\]\{\}]$                        |
        \d+\s*[\.\)]\s*\d?$                    |
        (?:principle|range|example|clause)(?:\s*/\s*(?:principle|range))?$ |
        [A-Za-z]$                              |
        [\u3000\s]*$
    )""",
    re.X
)

def _placeholder_ratio(cleaned_text: str) -> float:
    if not cleaned_text:
        return 1.0
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    bad = sum(1 for ln in lines if PLACEHOLDER_LINE.match(ln))
    return bad / max(1, len(lines))

# ============================ Utility functions with caching ============================

def load_config():
    cfg_path = CONF_DIR / "config.json"
    if cfg_path.exists():
        try:
            user_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return _merge_conf(DEFAULT_CONF, user_cfg)
        except Exception:
            pass
    return DEFAULT_CONF

def _merge_conf(base, user):
    out = dict(base)
    for k, v in (user or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            nv = dict(base[k]); nv.update(v); out[k] = nv
        else:
            out[k] = v
    return out

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def detect_latest_pid() -> str:
    if not REFINED_ROOT.exists():
        return ""
    cands = []
    for d in REFINED_ROOT.iterdir():
        if d.is_dir() and (d / "all_refined_items.json").exists():
            cands.append((d.name, (d / "all_refined_items.json").stat().st_mtime))
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[0][0] if cands else ""

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def norm01(x, lo, hi):
    """
    Linearly map x into [0, 1].
    If hi <= lo, return 0. Values outside the interval are clipped.
    """
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return float(v)

def safe_float(x, default=0.6):
    try:
        f = float(x)
        if math.isnan(f):
            return default
        return max(0.0, min(1.0, f))
    except Exception:
        return default

@lru_cache(maxsize=8192)
def _sanitize_cached(raw: str) -> str:
    return sanitize_for_display(raw)

@lru_cache(maxsize=8192)
def _word_tokens_cached(text: str):
    """
    Word-level tokens using [a-z0-9]+.
    This matches the semantic-alignment logic and does not stem or remove stopwords.
    """
    s = (sanitize_for_scoring(text) or "").lower()
    en = re.findall(r"[a-z0-9]+", s)
    return tuple(t for t in en if t)

_tokenize_cached = _word_tokens_cached

def tokenize(text: str):
    return list(_word_tokens_cached(text))


def jaccard(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    A, B = set(a_tokens), set(b_tokens)
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni else 0.0

def looks_structured(ans: str) -> float:
    cleaned = sanitize_for_scoring(ans)
    lines = [ln.strip() for ln in (cleaned or "").splitlines() if ln.strip()]
    if not lines:
        return 0.0
    bulletish = re.compile(r"^(\d+[\.\)]\s+|[-*]\s+|[A-Za-z]\))")
    bullets = sum(1 for ln in lines if bulletish.match(ln))
    ratio_b = bullets / max(1, len(lines))
    # Accept lines up to 200 chars to accommodate English bullet points
    # (Chinese bullets are ~50 chars; English are naturally 100–200 chars)
    shortish = sum(1 for ln in lines if 6 <= len(ln) <= 200)
    ratio_s = shortish / max(1, len(lines))
    return max(0.0, min(1.0, 0.4 * ratio_b + 0.6 * ratio_s))

def overclaim_score(ans: str) -> float:
    terms = ["must", "guaranteed", "absolute", "completely", "zero risk", "certainly", "cannot fail", "100%"]
    low = (ans or "").lower()
    cnt = sum(low.count(t) for t in terms)
    length = max(1, len(ans or ""))
    dens = cnt / length
    return max(0.0, min(1.0, dens * 200))

def contradiction_pair(a: str, b: str) -> float:
    neg = ["not", "no", "cannot", "avoid", "prohibit", "without", "lacks", "missing"]
    pos = ["can", "could", "allow", "recommend", "feasible", "through", "supports"]
    a_low = (sanitize_for_scoring(a) or "").lower()
    b_low = (sanitize_for_scoring(b) or "").lower()
    a_neg = sum(a_low.count(t) for t in neg)
    b_neg = sum(b_low.count(t) for t in neg)
    a_pos = sum(a_low.count(t) for t in pos)
    b_pos = sum(b_low.count(t) for t in pos)
    jac = jaccard(_tokenize_cached(sanitize_for_scoring(a)), _tokenize_cached(sanitize_for_scoring(b)))
    diff = abs((a_pos - a_neg) - (b_pos - b_neg))
    base = norm01(diff, 0, 20)
    penal = 1.0 - jac
    return max(0.0, min(1.0, base * penal))

def bar(value: float, n: int = 20) -> str:
    n = max(1, n)
    k = max(0, min(n, int(round(float(value) * n))))
    return "#" * k + "-" * (n - k)

def authority_ratio(hints: list) -> float:
    if not hints:
        return 0.0
    hits = 0
    for h in hints:
        s = (h or "").lower()
        if any(tok in s for tok in AUTHORITY_TOKENS):
            hits += 1
    return hits / max(1, len(hints))

def coverage_score(hints: list) -> float:
    if not hints:
        return 0.0
    cats = set()
    for h in hints:
        s = (h or "").lower()
        for cat, toks in COVERAGE_BANK.items():
            if any(tok in s for tok in toks):
                cats.add(cat)
    return len(cats) / len(COVERAGE_BANK)

def has_redline(text: str) -> bool:
    t = text or ""
    return any(p.search(t) for p in (RE_DATE, RE_MONEY, RE_TRIAL, RE_DOI, RE_PATENT, RE_ISO, RE_STDNUM, RE_ID_ANY))

# ============================ Dimension drift and authority alignment ============================

OTHER_DIMS = set(DIM_ORDER)

def _strip_cross_dim_tags(dim: str, tags: list) -> list:
    keep, seen = [], set()
    for t in tags or []:
        s = str(t or "").strip()
        if not s:
            continue
        low = s.lower()
        if low in OTHER_DIMS and low != dim.lower():
            continue
        if low in seen:
            continue
        seen.add(low)
        keep.append(s)
    return keep

def _authority_hints_from_qs(qs_cfg: dict, dim: str, limit: int = 8) -> list:
    try:
        # Support both flat {dim: {search_hints}} and nested {dimensions: {dim: {search_hints}}}
        dims_section = qs_cfg.get("dimensions", {}) or {}
        block = dims_section.get(dim) or qs_cfg.get(dim, {}) or {}
        hints = (block.get("search_hints", []) if isinstance(block, dict) else []) or []
        hints = _strip_cross_dim_tags(dim, hints)
        def score(h):
            s = str(h).lower()
            return -sum(1 for t in AUTHORITY_TOKENS if t in s)
        hints = list({h: None for h in hints}.keys())
        hints.sort(key=score)
        return hints[:limit]
    except Exception:
        return []

# ===== Semantic alignment =====
def _alignment_ratio(dim: str, auth_hints: list, answer: str, topic_tags: list, evidence_hints: list,
                     question: str = "", claims: list = None) -> float:
    """
    Semantic alignment score between candidate text and optional authority hints.
    Candidate corpus includes answer + topic_tags + evidence_hints twice, plus optional question/claims.
    Empty-token hints are skipped. If no valid hints remain, return a neutral fallback score.
    """
    claims = claims or []
    # Repeat evidence_hints once to increase their alignment weight.
    pool = [
        sanitize_for_scoring(answer or ""),
        " ".join([str(t) for t in (topic_tags or [])]),
        " ".join([str(h) for h in (evidence_hints or [])]),
        " ".join([str(h) for h in (evidence_hints or [])])
    ]
    if question:
        pool.append(str(question))
    if claims:
        pool.append(" ".join([str(c) for c in claims if str(c).strip()]))

    corpus = " \n ".join(pool)
    c_uni, c_bi = _tokens_for_alignment(corpus)

    # No hints: return a neutral fallback if candidate tokens exist.
    if not auth_hints:
        return 0.35 if c_uni else 0.0

    cover_hits = 0
    scores = []
    valid_hints = 0

    for h in auth_hints:
        q_uni, q_bi = _tokens_for_alignment(str(h))
        if not q_uni and not q_bi:
            continue
        valid_hints += 1
        s = _weighted_overlap(q_uni, q_bi, c_uni, c_bi)
        scores.append(s)
        if s >= 0.15:
            cover_hits += 1

    # If all hints are token-empty, treat this as a no-hint scenario.
    if valid_hints == 0:
        return 0.5 if c_uni else 0.0

    if not scores:
        return 0.0

    coverage = cover_hits / max(1, valid_hints)
    mean_hit = sum(scores) / len(scores)
    return max(0.0, min(1.0, 0.5 * coverage + 0.5 * mean_hit))


def _dimension_drift_score(dim: str, answer: str, topic_tags: list, evidence_hints: list) -> float:
    tags = " ".join([str(t) for t in (topic_tags or [])]).lower()
    others = sorted(OTHER_DIMS - {dim})
    hit = sum(1 for od in others if od in tags)
    weak_corpus = (" ".join([
        sanitize_for_scoring(answer or ""),
        " ".join([str(h) for h in (evidence_hints or [])])
    ])).lower()
    weak = sum(1 for od in others if od in weak_corpus)
    score = 0.6 * (hit / max(1, len(others))) + 0.4 * (weak / max(1, len(others)))
    return max(0.0, min(1.0, score))

# =============== Evidence phrases and general-insight extraction for reports ===============

def _top_evidence_phrases(hints: list, topk: int = 3):
    if not hints:
        return []
    def _norm(h: str):
        s = re.sub(r"\s+", " ", (h or "")).strip()
        return s[:120]
    candidates = []
    for h in hints:
        s = _norm(str(h))
        if not s:
            continue
        score = 0
        low = s.lower()
        for tok in AUTHORITY_TOKENS:
            if tok in low:
                score += 1
        candidates.append((s, score))
    counter = Counter([c[0] for c in candidates])
    ranked = sorted(
        counter.items(),
        key=lambda x: (-max([sc for (txt, sc) in candidates if txt == x[0]]), -x[1], x[0])
    )
    return [r[0] for r in ranked[:topk]]

def _uniq_general_insights(gi_list, topk: int = 10, max_len: int = 220):
    """
    Aggregate general_insights into dimension-level expert context:
    - deduplicate
    - normalize whitespace
    - cap item length
    """
    if not gi_list:
        return []
    seen = set()
    out = []
    for g in gi_list:
        s = re.sub(r"\s+", " ", str(g or "").strip())
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        if len(s) > max_len:
            s = s[:max_len].rstrip() + "..."
        out.append(s)
        if len(out) >= topk:
            break
    return out

def _dedupe_preserve_order(items):
    if not items:
        return []
    out = []
    seen = set()
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

# ============================ Scoring logic ============================

def _strong_alignment_bonus(ans: str, evids: list) -> float:
    if not ans and not evids:
        return 0.0
    text = (sanitize_for_scoring(ans) or "") + " " + " ".join([str(x) for x in (evids or [])])
    low = text.lower()
    strong = 0
    if ("clinicaltrials.gov" in low or "eudract" in low) and re.search(r"\b(NCT\d{8}|EUCTR-\d{4}-\d{6}-\d{2})\b", low, re.I):
        strong += 1
    if ("pubmed" in low or "doi" in low) and re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", low, re.I):
        strong += 1
    if ("uspto" in low or "epo" in low or "cnipa" in low) and re.search(r"\b(US|EP|CN)\d{5,}|WO\d{7,}\b", low, re.I):
        strong += 1
    if ("fda" in low or "ema" in low or "21 cfr" in low or "iso" in low) and re.search(r"\b(20\d{2}|19\d{2})\b", low, re.I):
        strong += 1
    return min(0.10, 0.03 * strong)

def score_candidate(ans_item: dict, cfg: dict, peer_tokens_list=None, dim: str = "", auth_hints: list = None, question: str = "") -> dict:
    ans = (ans_item.get("answer") or "").strip()
    claims = _dedupe_preserve_order(ans_item.get("claims") or [])
    evids = _dedupe_preserve_order(ans_item.get("evidence_hints") or [])
    tags  = _dedupe_preserve_order(ans_item.get("topic_tags") or [])
    conf  = safe_float(ans_item.get("confidence"), 0.6)
    diag  = ans_item.get("diag") or {}

    s_len   = norm01(len(sanitize_for_scoring(ans)), 0, cfg["length_ref_chars"])
    s_clm   = norm01(len([c for c in claims if isinstance(c, str) and c.strip()]), 0, cfg["claims_ref"])
    s_evc   = norm01(len([e for e in evids if isinstance(e, str) and e.strip()]), 0, cfg["evidence_ref"])
    s_eva   = authority_ratio(evids)
    s_evg   = coverage_score(evids)
    s_str   = looks_structured(ans)

    # Alignment. question/claims injection can be disabled for ablation.
    use_qc = not cfg.get("_ablate_no_question_claims", False)
    s_aln   = _alignment_ratio(dim, auth_hints or [], ans, tags, evids,
                               question=(question if use_qc else ""),
                               claims=(claims if use_qc else []))

    s_cal   = conf

    # auth_boost / cov_boost are diagnostic only.
    auth_boost = min(1.0, float(diag.get("auth_hits", 0)) / 5.0) if isinstance(diag.get("auth_hits", 0), (int, float)) else 0.0
    cov_boost  = min(1.0, float(len(set(diag.get("coverage_bins") or []))) / 4.0)
    s_eva = max(s_eva, auth_boost)
    s_evg = max(s_evg, cov_boost)

    s_aln = min(1.0, s_aln + _strong_alignment_bonus(ans, evids))

    if peer_tokens_list:
        me = list(_tokenize_cached(sanitize_for_scoring(ans)))
        sims = [jaccard(me, tks) for tks in peer_tokens_list if tks is not None]
        s_con = sum(sims) / max(1, len(sims))
    else:
        s_con = 0.5

    pen = 0.0
    oc = overclaim_score(ans)
    if oc > 0.15:
        pen += cfg["penalties"]["overclaim"]
    if s_str < 0.20:
        pen += cfg["penalties"]["understructure"]
    if any(has_redline(str(c)) for c in claims):
        pen += cfg["penalties"]["redline_residual"]

    drift_raw = _dimension_drift_score(dim, ans, tags, evids)
    if drift_raw > 0:
        scale = 0.4 if drift_raw <= 0.2 else (0.7 if drift_raw <= 0.5 else 1.0)
        pen += min(cfg["penalties"]["dimension_drift"] * scale, drift_raw * cfg["penalties"]["dimension_drift"])
    if (diag.get("cross_dim") is True):
        pen += min(cfg["penalties"]["dimension_drift"], 0.02)

    fw = cfg["fields_weight"]
    fields_part = (
        fw["length"] * s_len +
        fw["claims"] * s_clm +
        fw["evidence_count"] * s_evc +
        fw["evidence_authority"] * s_eva +
        fw["evidence_coverage"] * s_evg +
        fw["structure"] * s_str +
        fw["alignment"] * s_aln +
        fw["calibrated_confidence"] * s_cal
    )

    total = (1.0 - cfg["consistency_weight"]) * fields_part + cfg["consistency_weight"] * s_con
    total = max(0.0, min(1.0, total - pen))

    alpha = max(0.0, min(1.0, fields_part - pen))

    if (dim == "innovation") and (ans_item.get("diag", {}).get("repro_signal") is True):
        total = min(1.0, total + 0.01)
        alpha = min(1.0, alpha + 0.01)

    return {
        "scores": {
            "length": s_len,
            "claims": s_clm,
            "evidence_count": s_evc,
            "evidence_authority": s_eva,
            "evidence_coverage": s_evg,
            "structure": s_str,
            "alignment": s_aln,
            "calibrated_confidence": s_cal,
            "consistency": s_con
        },
        "penalties": {
            "overclaim": oc,
            "dimension_drift": drift_raw,
            "applied": pen
        },
        "alpha": alpha,
        "total": total
    }

# Strong bad-candidate filter. Returns (is_bad, reason).
def _bad_candidate_with_reason(c: dict, cfg: dict, dim_name: str, auth_hints: list, q_text: str = ""):
    ans_raw = (c.get("answer") or "").strip()
    ans = sanitize_for_scoring(ans_raw)

    # 1. Explicit error or too short.
    if c.get("error") is True:
        return True, "error"
    if len(ans) < 12:
        return True, "too_short"

    # 2. Placeholder/noise ratio. Only high ratios are treated as garbage.
    ph_ratio = _placeholder_ratio(sanitize_for_display(ans_raw))
    if ph_ratio > 0.60:
        return True, "placeholder_noise"

    # 3. Dimension-specific parameters.
    dim_conf = (cfg.get("dimension_specific") or {}).get(dim_name, {})
    min_struct  = cfg["filters"]["min_structured_score"]
    min_bullets = dim_conf.get("min_bullet_lines", cfg["filters"]["min_bullet_lines"])
    min_align   = float(dim_conf.get("min_alignment_for_keep", cfg["filters"]["min_alignment_for_keep"]))

    if dim_name == "unknown":
        # Use a lower alignment threshold for unknown dimensions.
        min_align = min(0.15, min_align)

    # 4. Structure score. Very poor structure is dropped; borderline cases may pass soft_window.
    s_struct = looks_structured(ans_raw)
    if s_struct < min_struct:
        if cfg["filters"].get("soft_window", True) and s_struct >= 0.5 * min_struct:
            pass
        else:
            return True, "poor_structure"

    # 5. Deduplicated bullet lines for bullet count and median line length.
    _raw_lines = [ln.strip() for ln in sanitize_for_display(ans_raw).splitlines() if ln.strip()]
    seen = set()
    bullet_lines = []
    for ln in _raw_lines:
        if ln not in seen:
            seen.add(ln)
            bullet_lines.append(ln)

    # 6. Basic bullet-count requirement.
    if len(bullet_lines) < min_bullets:
        if cfg["filters"].get("soft_window", True) and len(bullet_lines) >= max(1, min_bullets - 1):
            pass
        else:
            return True, "few_bullets"

    # 7. Median line length. Filters slogan-like or ultra-short bullet stacks.
    if bullet_lines:
        lens = sorted(len(x) for x in bullet_lines)
        med = lens[len(lens)//2]
        if med < cfg["filters"]["min_median_bullet_len"]:
            if cfg["filters"].get("soft_window", True) and med >= max(2, cfg["filters"]["min_median_bullet_len"] - 2):
                pass
            else:
                return True, "too_short_lines"

    # 8. Semantic alignment, used only to drop severely off-topic candidates.
    use_qc = not cfg.get("_ablate_no_question_claims", False)
    aln = _alignment_ratio(
        dim_name,
        auth_hints or [],
        ans_raw,
        c.get("topic_tags") or [],
        c.get("evidence_hints") or [],
        question=(q_text if use_qc else ""),
        claims=(c.get("claims") or [] if use_qc else [])
    )

    dyn_min_align = float(min_align)
    if aln < dyn_min_align:
        # If structure is good or alignment is only slightly low, allow under soft_window.
        if cfg["filters"].get("soft_window", True) and (aln >= dyn_min_align * 0.6 or s_struct >= 0.5):
            pass
        else:
            return True, "weak_alignment"

    return False, ""

def _fallback_pick(cands: list, dim: str, auth_hints: list):
    def _alignment_ratio_local(answer: str, evids: list):
        # Fast fallback alignment using direct hint containment.
        corpus = (sanitize_for_scoring(answer or "")) + " " + " ".join([str(h) for h in (evids or [])])
        total = len(auth_hints) or 1
        hit = 0
        low = corpus.lower()
        for h in auth_hints or []:
            if (h or "").strip().lower() in low:
                hit += 1
        return hit / total

    def _fallback_score(c):
        ans = (c.get("answer") or "")
        evids = c.get("evidence_hints") or []
        s_str = looks_structured(ans)
        s_eva = authority_ratio(evids)
        s_evg = coverage_score(evids)
        s_evc = norm01(len([e for e in evids if str(e).strip()]), 0, 4)
        s_aln = _alignment_ratio_local(ans, evids)
        red_p = 0.0
        if len((c.get("facts_redlined") or [])) >= 3:
            red_p = 0.08
        # Fallback score: structure and alignment dominate; evidence metrics are light.
        base = 0.50 * s_str + 0.25 * s_aln + 0.15 * s_evc + 0.05 * s_eva + 0.05 * s_evg
        return max(0.0, base - red_p)

    ranked = sorted([(i, _fallback_score(c)) for i, c in enumerate(cands)], key=lambda x: -x[1])
    return ranked[0][0] if ranked else 0

def _provider_name(cand: dict) -> str:
    pv = (cand.get("provider") or "").strip().lower()
    if pv:
        return pv
    mdl = (cand.get("model") or "").lower()
    if "deepseek" in mdl:
        return "deepseek"
    if "gpt" in mdl or "openai" in mdl or "o" == mdl[:1]:
        return "openai"
    return "default"

def _beta_with_sweetspot_and_provider(beta_raw: float, avg_jac: float, avg_ctr: float, cfg: dict, provider: str) -> float:
    lo, hi = cfg["consistency_correction"]["beta"]["jaccard_sweetspot"]
    if avg_jac > hi:
        beta_adj = (1 - 0.05) * beta_raw
    elif avg_jac < lo:
        beta_adj = (1 - 0.05 * (lo - avg_jac) / max(1e-9, lo)) * beta_raw
    else:
        beta_adj = beta_raw
    cal = cfg["consistency_correction"]["beta"]["provider_calibration"].get(
        provider, cfg["consistency_correction"]["beta"]["provider_calibration"]["default"]
    )
    delta = float(cal.get("beta_delta", 0.0))
    beta_final = max(0.0, min(1.0, beta_adj + delta))
    return beta_final

def select_best_candidate(cands: list, cfg: dict, dim: str, auth_hints: list, q_text: str = "", last_provider: str = None):
    if not cands:
        return {"best": None, "all": [], "pairwise": {"avg_jaccard": 0.0, "avg_contradiction": 0.0}, "drop_stats": {}, "selection_mode": "empty"}

    drop_stats = Counter()
    cleaned = []
    for c in cands:
        bad, reason = _bad_candidate_with_reason(c, cfg, dim_name=dim, auth_hints=auth_hints, q_text=q_text)
        if bad:
            drop_stats[reason] += 1
        else:
            cleaned.append(c)

    if not cleaned:
        idx = _fallback_pick(cands, dim, auth_hints)
        cleaned = [cands[idx]]
        selection_mode = "fallback"
    else:
        selection_mode = "filtered"

    cands = cleaned

    tokens = [list(_tokenize_cached(sanitize_for_scoring((c.get("answer") or "")))) for c in cands]
    jac_sum = ctr_sum = 0.0
    pair_cnt = 0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            pair_cnt += 1
            jv = jaccard(tokens[i], tokens[j])
            cv = contradiction_pair(cands[i].get("answer", ""), cands[j].get("answer", ""))
            jac_sum += jv
            ctr_sum += cv

    avg_jac = jac_sum / pair_cnt if pair_cnt else 0.0
    avg_ctr = ctr_sum / pair_cnt if pair_cnt else 0.0

    k1 = cfg["consistency_correction"]["k_contradiction"]
    beta_raw = max(0.0, min(1.0, (1 - k1 * avg_ctr) * (0.5 + 0.5 * avg_jac)))

    scored = []
    for idx, c in enumerate(cands):
        peer = [tokens[k] if k != idx else None for k in range(len(cands))]
        sc = score_candidate(c, cfg, peer_tokens_list=peer, dim=dim, auth_hints=auth_hints, question=q_text)
        pv = _provider_name(c)
        beta = _beta_with_sweetspot_and_provider(beta_raw, avg_jac, avg_ctr, cfg, provider=pv)
        final = max(0.0, min(1.0, sc.get("alpha", sc.get("total", 0.0)) * beta))
        scored.append((idx, final, sc, beta, pv))

    def _tie_key(t):
        final = t[1]
        sd = t[2].get("scores", {})
        return (
            -float(final),
            -float(sd.get("evidence_authority", 0.0)),
            -float(sd.get("evidence_coverage", 0.0)),
            -float(sd.get("alignment", 0.0)),
            -float(sd.get("structure", 0.0)),
            -float(sd.get("claims", 0.0)),
            -float(sd.get("length", 0.0)),
        )

    scored.sort(key=_tie_key)

    # Provider balancing when scores are close.
    def _pick_with_provider_balance(scored_list, last_pv, margin=0.02):
        """
        Prefer a provider different from the previous selected provider when its score
        is within margin of the top candidate. Otherwise keep the top candidate.
        scored_list shape: (idx, final_score, detail_dict, beta, provider)
        """
        if not scored_list:
            return None
        top = scored_list[0]
        if len(scored_list) == 1 or last_pv is None:
            return top
        top_score = top[1]
        # Only inspect the top three to avoid quality instability.
        for cand in scored_list[:3]:
            _, sc_final, _, _, sc_pv = cand
            if sc_pv != last_pv and (top_score - sc_final) < margin:
                return cand
        return top

    picked_tuple = _pick_with_provider_balance(scored, last_provider)
    if picked_tuple is None:
        best_idx, final_score, best_detail, best_beta, best_provider = (
            0, 0.0, {"scores": {}, "penalties": {}, "alpha": 0.0, "total": 0.0}, 0.0, "default"
        )
    else:
        best_idx, final_score, best_detail, best_beta, best_provider = picked_tuple

    conflict_pen = cfg["penalties"]["contradiction"] * avg_ctr
    topic_reliability = max(0.0, min(1.0, best_detail.get("total", 0.0) - conflict_pen))

    out_all = []
    for idx, fscore, detail, b, pv in scored:
        item = dict(cands[idx])
        item["_score_detail"] = detail
        item["_score_total"]  = detail.get("total", 0.0)
        item["_score_alpha"]  = detail.get("alpha", 0.0)
        item["_score_final"]  = fscore
        item["_score_beta"]   = b
        item["_provider_used"] = pv
        out_all.append(item)

    return {
        "best": {
            "index": best_idx,
            "candidate": cands[best_idx] if cands else None,
            "score": {
                "alpha": best_detail.get("alpha", 0.0),
                "beta": best_beta,
                "final": final_score,
                "raw_total": best_detail.get("total", 0.0),
                "after_topic_conflict": topic_reliability,
                "avg_pairwise_jaccard": avg_jac,
                "avg_pairwise_contradiction": avg_ctr
            },
            "detail": best_detail,
            "provider": best_provider
        },
        "all": out_all,
        "pairwise": {"avg_jaccard": avg_jac, "avg_contradiction": avg_ctr},
        "drop_stats": dict(drop_stats),
        "selection_mode": selection_mode
    }

# ============================ Aggregation and report ============================

def aggregate_dimensions(items: list, cfg: dict, qs_cfg: dict):
    per_question = []

    dim_bucket = defaultdict(list)
    dropped_reason_bucket = defaultdict(Counter)
    # Used for provider balancing across close-scoring candidates.
    last_provider_per_dim = {d: None for d in DIM_ORDER + ["unknown"]}

    auth_map = {dim: _authority_hints_from_qs(qs_cfg, dim, limit=8) for dim in DIM_ORDER}
    auth_map.setdefault("unknown", [])

    provider_stats = Counter()
    provider_alpha = defaultdict(list)
    provider_final = defaultdict(list)

    for it in items:
        dim0 = (it.get("dimension") or "").strip().lower()
        dim = dim0 if dim0 in DIM_ORDER else "unknown"
        qidx = it.get("q_index")
        ques = (it.get("question") or "").strip()
        cands = it.get("candidates") or []

        picked = select_best_candidate(
            cands, cfg,
            dim=dim,
            auth_hints=auth_map.get(dim, []),
            q_text=ques,
            last_provider=last_provider_per_dim.get(dim)
        )
        best = picked["best"]

        # Track the previous provider for close-score provider balancing.
        if best and best.get("candidate"):
            last_provider_per_dim[dim] = (best.get("provider") or last_provider_per_dim.get(dim))

        if picked.get("drop_stats"):
            dropped_reason_bucket[dim].update(picked["drop_stats"])

        if best and best["candidate"]:
            selc = best["candidate"]
            fallback_used = picked.get("selection_mode") == "fallback"
            best_item = {
                "dimension": dim,
                "q_index": qidx,
                "question": ques,
                "selected": {
                    "provider": selc.get("provider"),
                    "model": selc.get("model"),
                    "variant_id": selc.get("variant_id"),
                    "answer": selc.get("answer"),
                    "claims": selc.get("claims") or [],
                    "evidence_hints": selc.get("evidence_hints") or [],
                    "topic_tags": selc.get("topic_tags") or [],
                    "confidence": safe_float(selc.get("confidence"), 0.6),
                    "alignment_ratio": best["detail"]["scores"].get("alignment", 0.0),
                    "dimension_drift": best["detail"]["penalties"].get("dimension_drift", 0.0),
                    "facts_redlined": selc.get("facts_redlined", []) or [],
                    "general_insights": _uniq_general_insights(selc.get("general_insights") or [], topk=10),
                    "selection_mode": picked.get("selection_mode", "filtered"),
                    "fallback_used": fallback_used
                },
                "score": best["score"],
                "score_detail": best["detail"],
                "pairwise": picked["pairwise"],
                "all_candidates": picked["all"],
                "auth_hints_used": auth_map.get(dim, []),
                "drop_stats": picked["drop_stats"],
            }
            per_question.append(best_item)
            dim_bucket[dim].append(best_item)

            pv = (picked.get("best") or {}).get("provider") or "default"
            provider_stats[pv] += 1
            provider_alpha[pv].append(float(best["detail"].get("alpha", 0.0)))
            provider_final[pv].append(float(best["score"].get("final", 0.0)))
        else:
            per_question.append({
                "dimension": dim, "q_index": qidx, "question": ques,
                "selected": None,
                "score": {"alpha": 0.0, "beta": 0.0, "final": 0.0,
                          "raw_total": 0.0, "after_topic_conflict": 0.0,
                          "avg_pairwise_jaccard": 0.0, "avg_pairwise_contradiction": 0.0},
                "score_detail": {},
                "pairwise": {"avg_jaccard": 0.0, "avg_contradiction": 0.0},
                "all_candidates": [],
                "auth_hints_used": auth_map.get(dim, []),
                "drop_stats": picked["drop_stats"]
            })

    per_dimension = {}
    dims_for_report = list(DIM_ORDER)
    if "unknown" in dim_bucket:
        dims_for_report.append("unknown")

    drop_reasons_global = Counter()
    for d, cnts in dropped_reason_bucket.items():
        drop_reasons_global.update(cnts)

    for dim in dims_for_report:
        qs = sorted([x for x in per_question if x["dimension"] == dim], key=lambda z: z["q_index"])
        if not qs:
            per_dimension[dim] = {
                "avg": 0.0, "n": 0,
                "avg_alignment": 0.0, "avg_drift": 0.0,
                "strengths": [], "risks": [], "snippets": [],
                "auth_hints": auth_map.get(dim, []),
                "redlined_samples": [],
                "dropped_reasons": {},
                "top_evidence_phrases": [],
                "general_insights": [],
                "explain": {
                    "top_contributors": [],
                    "top_penalties": []
                }
            }
            continue

        # Safely read each question's final score, defaulting to 0.
        scores_final = []
        for q in qs:
            score_block = q.get("score") or {}
            val = score_block.get("final", score_block.get("after_topic_conflict", 0.0))
            try:
                scores_final.append(float(val))
            except (TypeError, ValueError):
                scores_final.append(0.0)

        # Use mean + max weighting to preserve dimension differentiation.
        if scores_final:
            mean_sc = sum(scores_final) / len(scores_final)
            max_sc = max(scores_final)
            avg = 0.6 * mean_sc + 0.4 * max_sc
        else:
            mean_sc = 0.0
            max_sc = 0.0
            avg = 0.0

        strengths, risks, snippets = [], [], []
        aln_vals, drf_vals = [], []
        redlined_samples = []

        evid_pool = []
        gi_pool = []
        contrib_counter = Counter()
        penalty_counter = Counter()

        for q in qs:
            sel = q.get("selected")
            if sel:
                eva = authority_ratio(sel.get("evidence_hints") or [])
                evg = coverage_score(sel.get("evidence_hints") or [])
                aln = float(sel.get("alignment_ratio", 0.0))
                drf = float(sel.get("dimension_drift", 0.0))

                aln_vals.append(aln)
                drf_vals.append(drf)

                # Strength signal: enough claims plus good alignment.
                if len(sel.get("claims") or []) >= 2 and aln >= 0.40:
                    strengths.append(
                        f"Q{q['q_index']}: enough claims with good structure/alignment "
                        f"(auth={eva:.2f}, cover={evg:.2f}, align={aln:.2f})"
                    )
                if drf > 0.0:
                    risks.append(f"Q{q['q_index']}: possible cross-dimension drift (drift={drf:.2f}); review dimension boundary manually")
                if (q.get("score_detail") or {}).get("penalties", {}).get("applied", 0.0) > 0.0:
                    risks.append("Penalty applied for possible overclaim, weak structure, or residual redline content; spot-check recommended")

                ans_txt = sel.get("answer") or ""
                clean_snip = sanitize_for_display(ans_txt)
                snippets.append(f"Q{q['q_index']}: {(clean_snip[:160] + '...') if clean_snip else '<none>'}")

                for s in (sel.get("facts_redlined") or [])[:2]:
                    if s not in redlined_samples and len(redlined_samples) < 8:
                        redlined_samples.append(s)

                evid_pool.extend(sel.get("evidence_hints") or [])
                gi_pool.extend(sel.get("general_insights") or [])

                sd = q.get("score_detail", {}).get("scores", {})
                rank_pairs = sorted(sd.items(), key=lambda x: -float(x[1]))[:2]
                for k, _ in rank_pairs:
                    contrib_counter[k] += 1
                pen = q.get("score_detail", {}).get("penalties", {})
                if pen.get("applied", 0.0) > 0.0:
                    if pen.get("dimension_drift", 0.0) > 0:
                        penalty_counter["dimension_drift"] += 1
                    if pen.get("overclaim", 0.0) > 0.15:
                        penalty_counter["overclaim"] += 1
                    if sd.get("structure", 1.0) < 0.2:
                        penalty_counter["understructure"] += 1
            else:
                risks.append(f"Q{q['q_index']}: no valid answer")
                snippets.append(f"Q{q['q_index']}: <none>")

        top_evid = _top_evidence_phrases(evid_pool, topk=3)
        gi_agg = _uniq_general_insights(gi_pool, topk=10)

        per_dimension[dim] = {
            "avg": avg,
            "n": len(qs),
            "avg_alignment": (sum(aln_vals) / max(1, len(aln_vals))) if aln_vals else 0.0,
            "avg_drift": (sum(drf_vals) / max(1, len(drf_vals))) if drf_vals else 0.0,
            "strengths": strengths[:cfg["adv_topk"]],
            "risks": risks[:cfg["adv_topk"]],
            "snippets": snippets[:6],
            "auth_hints": auth_map.get(dim, []),
            "redlined_samples": redlined_samples,
            "dropped_reasons": dict(dropped_reason_bucket.get(dim, {})),
            "top_evidence_phrases": top_evid,
            "general_insights": gi_agg,
            "explain": {
                "top_contributors": [k for k, _ in contrib_counter.most_common(3)],
                "top_penalties": [k for k, _ in penalty_counter.most_common(3)]
            }
        }

    # overall
    dim_score = 0.0
    weight_sum = 0.0
    total_q_count = len(per_question)
    unknown_q_count = len([1 for q in per_question if q["dimension"] == "unknown"])
    for dim, info in per_dimension.items():
        if dim not in DIM_ORDER:
            continue
        w = (cfg.get("dimension_weight") or {}).get(dim, 1.0)
        dim_score += w * float(info.get("avg", 0.0))
        weight_sum += w
    overall_score = dim_score / max(1e-9, weight_sum if weight_sum > 0 else 1.0)

    all_conf, all_jac, all_ctr, cnt = [], 0.0, 0.0, 0
    fallback_count = 0
    for q in per_question:
        if q.get("selected"):
            all_conf.append(safe_float(q["selected"].get("confidence"), 0.6))
            if q["selected"].get("fallback_used"):
                fallback_count += 1
        all_jac += float(q.get("pairwise", {}).get("avg_jaccard", 0.0))
        all_ctr += float(q.get("pairwise", {}).get("avg_contradiction", 0.0))
        cnt += 1
    mean_conf = sum(all_conf) / max(1, len(all_conf))
    mean_jac = all_jac / max(1, cnt)
    mean_ctr = all_ctr / max(1, cnt)
    overall_confidence = max(0.0, min(1.0, (0.5 * mean_conf + 0.3 * mean_jac + 0.2 * (1 - mean_ctr))))

    provider_summary = {}
    for pv, n in provider_stats.items():
        provider_summary[pv] = {
            "selected_count": n,
            "avg_alpha": sum(provider_alpha[pv]) / max(1, len(provider_alpha[pv])),
            "avg_final": sum(provider_final[pv]) / max(1, len(provider_final[pv]))
        }

    drop_g = dict(drop_reasons_global)
    placeholder_cnt = drop_g.get("placeholder_noise", 0)
    few_bullets_cnt = drop_g.get("few_bullets", 0)

    dim_median_bullets = {}
    for dim in per_dimension:
        qs_d = [q for q in per_question if q["dimension"] == dim and q.get("selected")]
        nums = []
        for q in qs_d:
            cleaned = sanitize_for_display(q["selected"]["answer"] or "")
            nums.append(len([ln for ln in cleaned.splitlines() if ln.strip()]))
        if nums:
            nums.sort()
            dim_median_bullets[dim] = nums[len(nums)//2]
        else:
            dim_median_bullets[dim] = 0

    overall = {
        "overall_score": overall_score,
        "overall_confidence": overall_confidence,
        "mean_pairwise_jaccard": mean_jac,
        "mean_pairwise_contradiction": mean_ctr,
        "unknown_ratio": (unknown_q_count / max(1, total_q_count)),
        "drop_reasons_global": dict(drop_reasons_global),
        "provider_stats": provider_summary,
        "placeholder_ratio_global": placeholder_cnt / max(1, sum(drop_g.values())) if drop_g else 0.0,
        "few_bullets_ratio_global": few_bullets_cnt / max(1, sum(drop_g.values())) if drop_g else 0.0,
        "dim_median_bullets": dim_median_bullets,
        "fallback_ratio_global": fallback_count / max(1, sum(1 for q in per_question if q.get("selected")))
    }

    return per_question, per_dimension, overall

def build_report_md(pid: str, meta: dict, per_dim: dict, overall: dict, cfg: dict) -> str:
    lines = []
    lines.append(f"# Post-Processing Report - {pid}")
    lines.append("")
    lines.append(f"- Generated at: {now_str()}")
    if meta:
        m = {k: meta.get(k) for k in ("generated_at", "pid", "schema") if k in meta}
        if "args" in meta:
            m["args"] = meta["args"]
        lines.append(f"- Metadata: {json.dumps(m, ensure_ascii=False)}")
    lines.append("")
    lines.append("## Overview")
    sc = overall["overall_score"]
    cf = overall["overall_confidence"]
    lines.append(f"- Overall score (0-1): **{sc:.3f}**  {bar(sc, cfg['bar_symbols'])}")
    lines.append(f"- Overall confidence (0-1): **{cf:.3f}**  {bar(cf, cfg['bar_symbols'])}")
    lines.append(f"- Global consistency (mean Jaccard): **{overall['mean_pairwise_jaccard']:.3f}**")
    lines.append(f"- Global contradiction (mean): **{overall['mean_pairwise_contradiction']:.3f}**")

    if "drop_reasons_global" in overall and overall["drop_reasons_global"]:
        drg = overall["drop_reasons_global"]
        disp = ", ".join([f"{k}:{v}" for k, v in sorted(drg.items(), key=lambda x: (-x[1], x[0]))])
        lines.append(f"- Global candidate drop reasons: {disp}")
    lines.append(f"- Estimated placeholder-noise share: {overall.get('placeholder_ratio_global', 0.0):.1%}")
    lines.append(f"- Estimated few-bullets share: {overall.get('few_bullets_ratio_global', 0.0):.1%}")
    lines.append(f"- Fallback selection share: {overall.get('fallback_ratio_global', 0.0):.1%}")

    if "unknown" in per_dim and per_dim["unknown"]["n"] > 0:
        unk_n = per_dim["unknown"]["n"]
        unk_ratio = overall.get("unknown_ratio", 0.0)
        lines.append(f"- Warning: **{unk_n}** question(s) landed in the `unknown` dimension ({unk_ratio:.1%}); review the question set and dimension extraction.")
        if unk_ratio > cfg.get("unknown_warn_ratio", 0.10):
            lines.append(f"- **High-severity warning**: unknown share exceeds the {int(cfg.get('unknown_warn_ratio', 0.10)*100)}% warning threshold.")

    pvstats = overall.get("provider_stats", {})
    if pvstats:
        parts = []
        for k, v in pvstats.items():
            parts.append(f"{k}: selected={v['selected_count']} | avg_alpha={v['avg_alpha']:.3f} | avg_final={v['avg_final']:.3f}")
        lines.append(f"- Provider statistics: {'; '.join(parts)}")

    lines.append("")
    lines.append("## Dimension Breakdown")

    dims_in_report = list(DIM_ORDER) + [d for d in per_dim.keys() if d not in DIM_ORDER]

    for dim in dims_in_report:
        if dim not in per_dim:
            continue
        info = per_dim[dim]
        avg = info["avg"]
        lines.append(f"### {dim} - score {avg:.3f}  {bar(avg, cfg['bar_symbols'])}")
        if info.get("auth_hints"):
            lines.append(f"- Reference hints, not facts: {'; '.join(info['auth_hints'])}")
        lines.append(f"- Mean alignment / drift: **{info.get('avg_alignment', 0.0):.2f} / {info.get('avg_drift', 0.0):.2f}**")
        if "explain" in info:
            ex = info["explain"] or {}
            if ex.get("top_contributors"):
                lines.append(f"- Main score contributors: {', '.join(ex['top_contributors'])}")
            if ex.get("top_penalties"):
                lines.append(f"- Main penalty factors: {', '.join(ex['top_penalties'])}")
        if info.get("top_evidence_phrases"):
            lines.append("- **Top evidence phrases, authority hits first**:")
            for s in info["top_evidence_phrases"]:
                lines.append(f"  - {s}")
        gi_list = info.get("general_insights") or []
        if gi_list:
            lines.append("- **General insights, not project achievements**:")
            for s in gi_list[:5]:
                lines.append(f"  - {s}")
        if info["strengths"]:
            lines.append("- **Strengths**:")
            for s in info["strengths"]:
                lines.append(f"  - {s}")
        if info["risks"]:
            lines.append("- **Risks**:")
            for r in info["risks"]:
                lines.append(f"  - {r}")
        if info["snippets"]:
            lines.append("- **Representative snippets**:")
            for sn in info["snippets"]:
                lines.append(f"  - {sn}")
        redlined_samples = info.get("redlined_samples") or []
        if redlined_samples:
            lines.append("- **Original sentences moved to evidence hints, sample**:")
            for s in redlined_samples[:6]:
                lines.append(f"  - {s}")
        drop_r = info.get("dropped_reasons") or {}
        if drop_r:
            lines.append("- **Dropped candidate stats, reason: count**:")
            disp = ", ".join([f"{k}:{v}" for k, v in sorted(drop_r.items(), key=lambda x: (-x[1], x[0]))])
            lines.append(f"  - {disp}")
        lines.append(f"- Median bullet count, estimated: {overall.get('dim_median_bullets', {}).get(dim, 0)}")
        lines.append("")
    lines.append("> Note: this report is based only on structured candidate-answer metrics, "
                 "including length, claims, evidence hints, authority/coverage diagnostics, structure, consistency, confidence, and dimension alignment. "
                 "General insights are industry-context suggestions and do not mean the project has already met those requirements. "
                 "Before final use, manually check alignment with the original proposal and supporting evidence.")
    return "\n".join(lines)

# ============================ Main flow ============================

def _log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def main():
    stage_start = time.perf_counter()
    ap = argparse.ArgumentParser(description="Post-processing for structured candidates (no LLM calls).")
    ap.add_argument("--pid", type=str, default="", help="Proposal ID. If omitted, the latest refined_answers directory is used.")
    ap.add_argument("--input", type=str, default="", help="Optional path to all_refined_items.json.")
    ap.add_argument("--qs_file", type=str, default="", help="Optional generated_questions.json path for this proposal.")

    # A/B switches.
    ap.add_argument("--ablate_no_question_claims", action="store_true",
                    help="Disable question/claims injection into the alignment corpus for ablation.")
    ap.add_argument("--ablate_no_dyn_relax", action="store_true",
                    help="Deprecated compatibility flag for dynamic alignment relaxation.")

    args = ap.parse_args()

    cfg = load_config()

    # Store A/B switches in cfg so downstream functions can read them.
    cfg["_ablate_no_question_claims"] = bool(args.ablate_no_question_claims)
    cfg["_ablate_no_dyn_relax"] = bool(args.ablate_no_dyn_relax)

    _log("STAGE_5", "Starting Stage 5: post-processing answer candidates")
    _log(
        "STAGE_5",
        "Purpose: score, filter, select, and aggregate Stage 4 answer candidates into final_payload."
    )
    _log(
        "STAGE_5",
        f"pid_arg={args.pid or '<auto>'} input_arg={args.input or '<auto>'} "
        f"ablate_no_question_claims={bool(args.ablate_no_question_claims)}"
    )

    # Read question set for optional authority/topic hint alignment.
    qs_path = Path(args.qs_file) if args.qs_file.strip() else QS_CONF_PATH
    if qs_path.exists():
        try:
            qs_cfg = read_json(qs_path)
            _log("INPUT", f"question_set_path={qs_path} size_bytes={qs_path.stat().st_size}")
        except Exception:
            qs_cfg = {}
            _log("WARN", f"question_set_path={qs_path} could not be read; continuing without hints")
    else:
        qs_cfg = {}
        _log("WARN", f"question_set_path={qs_path} not found; continuing without hints")

    if args.input:
        refined_path = Path(args.input)
        if not refined_path.exists():
            raise FileNotFoundError(f"Input file not found: {refined_path}")
        pid = refined_path.parent.name
    else:
        pid = args.pid.strip() or detect_latest_pid()
        if not pid:
            raise RuntimeError("No latest project directory found under refined_answers, and neither --pid nor --input was provided.")
        refined_path = REFINED_ROOT / pid / "all_refined_items.json"
        if not refined_path.exists():
            raise FileNotFoundError(f"File not found: {refined_path}")

    _log("INPUT", f"proposal_id={pid}")
    _log("INPUT", f"refined_items_path={refined_path} size_bytes={refined_path.stat().st_size}")

    # If no explicit --qs_file, try the per-proposal question set and merge its search_hints.
    if not args.qs_file.strip():
        per_proposal_qs = DATA_DIR / "questions" / pid / "generated_questions.json"
        if per_proposal_qs.exists():
            try:
                per_qs = read_json(per_proposal_qs)
                dims_block = per_qs.get("dimensions", per_qs) or {}
                for _dim in DIM_ORDER:
                    _dim_block = dims_block.get(_dim, {})
                    _hints = (_dim_block.get("search_hints") or [] if isinstance(_dim_block, dict) else [])
                    if _hints:
                        qs_cfg.setdefault(_dim, {})["search_hints"] = _hints
                _log("INPUT", f"per_proposal_qs merged search_hints from {per_proposal_qs}")
            except Exception as _e:
                _log("WARN", f"could not read per-proposal question set: {_e}")

    # Always merge key_points from dimensions_v2.json into search_hints as additional authority anchors.
    # This ensures Chinese verbatim phrases are available for alignment even when English search_hints exist.
    dims_v2_path = DATA_DIR / "extracted" / pid / "dimensions_v2.json"
    if dims_v2_path.exists():
        try:
            dims_data = read_json(dims_v2_path)
            for _dim in DIM_ORDER:
                kps = (dims_data.get(_dim, {}) or {}).get("key_points", [])
                if kps:
                    existing = qs_cfg.get(_dim, {}).get("search_hints", []) or []
                    merged = existing + [str(k)[:120] for k in kps[:6] if str(k)[:120] not in existing]
                    qs_cfg.setdefault(_dim, {})["search_hints"] = merged
            _log("INPUT", f"merged key_points into search_hints from {dims_v2_path}")
        except Exception as _e:
            _log("WARN", f"could not merge key_points from dimensions_v2: {_e}")

    data = read_json(refined_path)
    meta = (data.get("meta") or {})
    schema = meta.get("schema", "")

    # Supports llm_answering.v2 and refined_items.v2.proposal_aware_with_general_insights.
    allowed_schemas = {"", "llm_answering.v2", "refined_items.v2.proposal_aware_with_general_insights"}
    if schema and schema not in allowed_schemas:
        _log("WARN", f"input schema={schema}; expected one of {allowed_schemas}. Continuing anyway.")

    # Compatible read: newer llm_answering uses items, older formats may use questions.
    items = data.get("items") or data.get("questions") or []
    _log("INPUT", f"input_items={len(items)} schema={schema or '<empty>'}")

    bad = []
    for i, it in enumerate(items):
        if "dimension" not in it or "q_index" not in it or "question" not in it or "candidates" not in it:
            bad.append(i)
    if bad:
        raise ValueError(f"Input items are missing required fields at indexes {bad[:10]} ... Check llm_answering output structure.")

    aggregate_start = time.perf_counter()
    per_question, per_dimension, overall = aggregate_dimensions(items, cfg, qs_cfg)
    aggregate_elapsed = time.perf_counter() - aggregate_start
    _log("TIMING", f"aggregate_dimensions elapsed_sec={aggregate_elapsed:.3f}")
    _log(
        "SUMMARY",
        f"questions={len(per_question)} overall_score={overall['overall_score']:.3f} "
        f"overall_confidence={overall['overall_confidence']:.3f}"
    )
    for dim in DIM_ORDER:
        info = per_dimension.get(dim, {})
        _log(
            "DIM",
            f"{dim}: n={info.get('n', 0)} avg={info.get('avg', 0.0):.3f} "
            f"avg_alignment={info.get('avg_alignment', 0.0):.3f} avg_drift={info.get('avg_drift', 0.0):.3f} "
            f"dropped={info.get('dropped_reasons', {})}"
        )

    out_dir = REFINED_ROOT / pid / "postproc"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log("OUTPUT", f"postproc_output_dir={out_dir}")

    metrics = {
        "meta": {
            "pid": pid,
            "generated_at": now_str(),
            "source_file": str(refined_path),
            "schema": schema or "",
            "args": meta.get("args", {}),
            "stage": "stage_5_post_processing",
            "stage_elapsed_sec": None,
            "aggregate_elapsed_sec": round(aggregate_elapsed, 3),
        },
        "config_used": cfg,
        "overall": overall,
        "dimensions": per_dimension,
        "questions": per_question
    }
    # metrics.json is written once at the end (after stage_elapsed_sec is computed).

    # Per-question selected result for manual inspection.
    write_json(out_dir / "selected_by_question.json", per_question)

    # final_payload is the unified downstream entry point.
    final_payload = {
        "meta": {"pid": pid, "generated_at": now_str()},
        "dimensions": {}
    }
    _write_progress(0, len(DIM_ORDER), pid)
    for dim_idx, dim in enumerate(DIM_ORDER):
        qs = [q for q in per_question if q["dimension"] == dim and q.get("selected")]
        final_payload["dimensions"][dim] = {
            "score": round(float(per_dimension.get(dim, {}).get("avg", 0.0)) * 100, 1),
            "qas": [
                {
                    "q": q["question"],
                    "answer": q["selected"]["answer"],
                    "claims": _dedupe_preserve_order(q["selected"]["claims"]),
                    "evidence_hints": _dedupe_preserve_order(q["selected"]["evidence_hints"]),
                    "topic_tags": _dedupe_preserve_order(q["selected"].get("topic_tags", [])),
                    "provider": q["selected"].get("provider"),
                    "model": q["selected"].get("model"),
                    "alignment": q["selected"].get("alignment_ratio", 0.0),
                    "dimension_drift": q["selected"].get("dimension_drift", 0.0),
                    "confidence": q["selected"]["confidence"],
                    "caveats": "",
                    "general_insights": _uniq_general_insights(q["selected"].get("general_insights", []), topk=10),
                    "selection_mode": q["selected"].get("selection_mode", "filtered"),
                    "fallback_used": q["selected"].get("fallback_used", False)
                } for q in qs
            ],
            "rationales": (per_dimension.get(dim, {}).get("strengths") or [])[:3],
            "general_insights": per_dimension.get(dim, {}).get("general_insights", [])
        }
        _write_progress(dim_idx + 1, len(DIM_ORDER), pid)
    write_json(out_dir / "final_payload.json", final_payload)

    report_md = build_report_md(pid, meta, per_dimension, overall, cfg)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    # Drop-reason debug file, only for questions with drop stats.
    write_json(
        out_dir / "drops_debug.json",
        [
            {
                "dimension": q.get("dimension"),
                "q_index": q.get("q_index"),
                "question": q.get("question"),
                "drop_stats": q.get("drop_stats", {})
            }
            for q in metrics["questions"]
            if q.get("drop_stats")
        ]
    )

    elapsed = time.perf_counter() - stage_start
    audit = {
        "stage": "stage_5_post_processing",
        "purpose": "Score, filter, select, and aggregate Stage 4 answer candidates.",
        "proposal_id": pid,
        "source_file": str(refined_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "elapsed_sec": round(elapsed, 3),
        "aggregate_elapsed_sec": round(aggregate_elapsed, 3),
        "input_items": len(items),
        "selected_questions": len(per_question),
        "overall": overall,
        "dimension_summary": {
            dim: {
                "n": per_dimension.get(dim, {}).get("n", 0),
                "avg": per_dimension.get(dim, {}).get("avg", 0.0),
                "avg_alignment": per_dimension.get(dim, {}).get("avg_alignment", 0.0),
                "avg_drift": per_dimension.get(dim, {}).get("avg_drift", 0.0),
                "dropped_reasons": per_dimension.get(dim, {}).get("dropped_reasons", {}),
            }
            for dim in list(DIM_ORDER) + [d for d in per_dimension.keys() if d not in DIM_ORDER]
        },
        "outputs": {
            "metrics": str((out_dir / "metrics.json").resolve()),
            "selected_by_question": str((out_dir / "selected_by_question.json").resolve()),
            "final_payload": str((out_dir / "final_payload.json").resolve()),
            "report": str((out_dir / "report.md").resolve()),
            "drops_debug": str((out_dir / "drops_debug.json").resolve()),
        },
    }
    metrics["meta"]["stage_elapsed_sec"] = round(elapsed, 3)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "stage5_post_processing_audit.json", audit)

    _log("OK", f"metrics.json written path={out_dir/'metrics.json'}")
    _log("OK", f"selected_by_question.json written path={out_dir/'selected_by_question.json'}")
    _log("OK", f"final_payload.json written path={out_dir/'final_payload.json'}")
    _log("OK", f"report.md written path={out_dir/'report.md'}")
    _log("OK", f"drops_debug.json written path={out_dir/'drops_debug.json'}")
    _log("OK", f"stage5_post_processing_audit.json written path={out_dir/'stage5_post_processing_audit.json'}")

    total_q = len(metrics["questions"])
    dim_brief = ", ".join(f"{d}:{info['n']}" for d, info in metrics["dimensions"].items())
    ov = metrics["overall"]
    _log("SUMMARY", f"question_count={total_q} dimension_counts={dim_brief}")
    _log(
        "SUMMARY",
        f"overall_score={ov['overall_score']:.3f} overall_confidence={ov['overall_confidence']:.3f} "
        f"jaccard={ov['mean_pairwise_jaccard']:.3f} contradiction={ov['mean_pairwise_contradiction']:.3f}"
    )
    if "unknown" in per_dimension and per_dimension["unknown"]["n"] > 0:
        unk_ratio = ov.get("unknown_ratio", 0.0)
        _log("WARN", f"{per_dimension['unknown']['n']} question(s) landed in unknown dimension ({unk_ratio:.1%}); check question set and dimension extraction.")

    drg = ov.get("drop_reasons_global", {})
    if drg:
        disp = ", ".join([f"{k}:{v}" for k, v in sorted(drg.items(), key=lambda x: (-x[1], x[0]))])
        _log("SUMMARY", f"global_drop_reasons={disp}")

    pvstats = ov.get("provider_stats", {})
    if pvstats:
        parts = []
        for k, v in pvstats.items():
            parts.append(f"{k}: selected={v['selected_count']} | avg_alpha={v['avg_alpha']:.3f} | avg_final={v['avg_final']:.3f}")
        _log("SUMMARY", f"provider_stats={'; '.join(parts)}")

    _log("STAGE_5", f"Completed Stage 5 elapsed_sec={elapsed:.3f}")

if __name__ == "__main__":
    main()
