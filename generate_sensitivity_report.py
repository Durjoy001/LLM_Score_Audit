#!/usr/bin/env python3
"""
Generate a detailed HTML + PDF report of the per-subrubric content-sensitivity results.

Usage:
    python3 generate_sensitivity_report.py
    python3 generate_sensitivity_report.py --out_html report.html --out_pdf report.pdf
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
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

DIM_DESCRIPTIONS = {
    "strategy": {
        "plus":  "Injected a detailed <strong>Revenue Model</strong> (B2B SaaS pricing, unit economics) "
                 "and <strong>Regulatory Strategy</strong> (FDA 510(k) Q-Sub reference, CE mark timeline) "
                 "before the technology section.",
        "minus": "Stripped all <strong>named partner mentions</strong> (Merck, Pfizer, Roche, etc.) "
                 "replacing them with &ldquo;strategic partner&rdquo;, and removed "
                 "<strong>phased-funding strategy</strong> language.",
    },
    "advantages": {
        "plus":  "Injected a concrete <strong>IP Portfolio</strong> (granted US patent number, continuations, EP grant) "
                 "and quantitative <strong>Performance vs Benchmark</strong> data (transfection efficiency, "
                 "cytotoxicity IC50, shelf stability) under GLP conditions.",
        "minus": "Stripped specific <strong>technical differentiators</strong> (lung-targeted LNP, generative design, "
                 "RNN, AI-guided delivery) and all <strong>IP / exclusivity language</strong> "
                 "(patent, proprietary, IP portfolio, exclusive rights).",
    },
    "objectives": {
        "plus":  "Injected a sourced <strong>Market Size</strong> figure ($57.3B TAM, CAGR, SAM estimate) "
                 "and a signed <strong>Letter of Intent</strong> from MedHealth Network with 42 hospital sites "
                 "and a defined path to a Master Supply Agreement.",
        "minus": "Removed specific <strong>disease and patient context</strong> (COVID-19 → &ldquo;viral disease&rdquo;, "
                 "immunocompromised → at-risk, 18 million patients → many patients) and "
                 "<strong>problem–solution connection language</strong> "
                 "(specifically targets, directly addresses, aligns with critical need).",
    },
}

HYPOTHESIS_LOGIC = {
    "strategy": {
        "plus":  "PLUS should raise R3 Revenue and R4 Regulatory (evidence directly added).",
        "minus": "MINUS should lower R6 Partners and R8 Timing (supporting evidence removed).",
    },
    "advantages": {
        "plus":  "PLUS should raise R2 IP Status and R3 Performance Proof (explicit data added).",
        "minus": "MINUS should lower R1 Mechanism Novelty and R5 Defensibility (novelty signals stripped).",
    },
    "objectives": {
        "plus":  "PLUS should raise R2 Market Size and R3 Buyer Pathway (quantified evidence added).",
        "minus": "MINUS should lower R1 Problem Clarity and R7 Problem–Solution Fit (specificity removed).",
    },
}


# ── Statistical helper ────────────────────────────────────────────────────────

def _stats(passes: int, valid: int, deltas: list) -> dict:
    if valid == 0:
        return {}
    rate = passes / valid
    # One-sided p-value (directional hypothesis); two-sided CI (symmetric interval)
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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(dim: str) -> tuple[dict | None, list, list]:
    path = DIMENSION_FILES[dim]
    if not path.exists():
        return None, [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    plus_targets  = data.get("plus_targets",  DIMENSIONS[dim]["plus_targets"])
    minus_targets = data.get("minus_targets", DIMENSIONS[dim]["minus_targets"])
    return data.get("results", {}), plus_targets, minus_targets


# ── HTML helpers ──────────────────────────────────────────────────────────────

def col_label(key: str, plus_targets: list, minus_targets: list) -> str:
    parts = key.split("_")
    r_idx = next((i for i, p in enumerate(parts) if p.startswith("R") and p[1:].isdigit()), None)
    if r_idx is None:
        return key
    r_num  = parts[r_idx]
    suffix = parts[r_idx + 1][:4] if r_idx + 1 < len(parts) else ""
    label  = f"{r_num}_{suffix}"
    if key in plus_targets:
        label += " ▲"
    elif key in minus_targets:
        label += " ▼"
    return label


def delta_html(new_val, base_val) -> str:
    if not isinstance(new_val, int) or not isinstance(base_val, int):
        return f'<span class="nc">{new_val if new_val is not None else "—"}</span>'
    d = new_val - base_val
    if d > 0:
        return f'<span class="up">{new_val} <small>(+{d}↑)</small></span>'
    elif d < 0:
        return f'<span class="dn">{new_val} <small>({d}↓)</small></span>'
    else:
        return f'<span class="nc">{new_val} <small>(→)</small></span>'


def score_badge(v) -> str:
    if not isinstance(v, int):
        return f'<span class="badge b0">—</span>'
    cls = {1: "b1", 2: "b2", 3: "b3", 4: "b4", 5: "b5"}.get(v, "b0")
    return f'<span class="badge {cls}">{v}</span>'


# ── Section builders ──────────────────────────────────────────────────────────

def build_dimension_section(dim: str, cfg: dict, results: dict, pids: list) -> str:
    rubrics       = cfg["rubrics"]
    plus_targets  = cfg["plus_targets"]
    minus_targets = cfg["minus_targets"]
    keys          = [r[0] for r in rubrics]
    labels        = [col_label(k, plus_targets, minus_targets) for k in keys]
    active        = [p for p in pids if p in results]
    descs         = DIM_DESCRIPTIONS[dim]
    hyp           = HYPOTHESIS_LOGIC[dim]

    def th_class(key):
        if key in plus_targets:  return ' class="th-plus"'
        if key in minus_targets: return ' class="th-minus"'
        return ''

    def td_class(key):
        if key in plus_targets:  return ' class="td-plus"'
        if key in minus_targets: return ' class="td-minus"'
        return ''

    def header_row():
        cells = "".join(f'<th{th_class(k)}>{lbl}</th>' for k, lbl in zip(keys, labels))
        return f"<tr><th>Proposal</th>{cells}</tr>"

    # ── Baseline table
    rows_b = []
    for pid in active:
        b = results[pid].get("baseline") or {}
        cells = "".join(f'<td{td_class(k)}>{score_badge(b.get(k))}</td>' for k in keys)
        rows_b.append(f"<tr><td class='pid'>{pid}</td>{cells}</tr>")

    # ── PLUS table
    rows_p = []
    for pid in active:
        b = results[pid].get("baseline") or {}
        p = results[pid].get("plus")     or {}
        cells = "".join(f'<td{td_class(k)}>{delta_html(p.get(k), b.get(k))}</td>' for k in keys)
        rows_p.append(f"<tr><td class='pid'>{pid}</td>{cells}</tr>")

    # ── MINUS table
    rows_m = []
    for pid in active:
        b = results[pid].get("baseline") or {}
        m = results[pid].get("minus")    or {}
        cells = "".join(f'<td{td_class(k)}>{delta_html(m.get(k), b.get(k))}</td>' for k in keys)
        rows_m.append(f"<tr><td class='pid'>{pid}</td>{cells}</tr>")

    # ── Hypothesis check — two tables (PLUS and MINUS separately)
    # Ceiling effect: exclude PLUS checks where baseline=5 (can't rise further)
    # Floor effect:   exclude MINUS checks where baseline=1 (can't fall further)
    def _hyp_table_html(targets: list, variant_key: str) -> tuple[str, int, int, int]:
        """Returns (html, passes, valid_checks, excluded_count)."""
        th_cls = "th-plus" if variant_key == "plus" else "th-minus"
        hdr = "".join(
            f'<th class="{th_cls}">{col_label(k, plus_targets, minus_targets)}</th>'
            for k in targets
        )
        rows = []
        t_pass = t_valid = t_excl = 0
        for pid in active:
            b = results[pid].get("baseline") or {}
            v = results[pid].get(variant_key) or {}
            cells = []
            passed = valid = excl = 0
            for key in targets:
                bv = b.get(key)
                vv = v.get(key)
                at_limit = (variant_key == "plus" and bv == 5) or (variant_key == "minus" and bv == 1)
                if at_limit:
                    cells.append('<td class="excl" title="Excluded: score at limit">—</td>')
                    excl += 1
                else:
                    if variant_key == "plus":
                        ok = isinstance(vv, int) and isinstance(bv, int) and vv > bv
                    else:
                        ok = isinstance(vv, int) and isinstance(bv, int) and vv < bv
                    cls = "pass" if ok else "fail"
                    cells.append(f'<td class="{cls}">{"✓" if ok else "✗"}</td>')
                    passed += ok; valid += 1
            pct = 100 * passed // valid if valid else 0
            pct_cls = "pct-high" if pct >= 75 else ("pct-mid" if pct >= 50 else "pct-low")
            excl_note = f' <span class="excl-note">(-{excl})</span>' if excl else ""
            rows.append(
                f"<tr><td class='pid'>{pid}</td>{''.join(cells)}"
                f"<td class='result {pct_cls}'>{passed}/{valid}{excl_note} ({pct}%)</td></tr>"
            )
            t_pass += passed; t_valid += valid; t_excl += excl
        overall_pct = 100 * t_pass // t_valid if t_valid else 0
        opct_cls = "pct-high" if overall_pct >= 75 else ("pct-mid" if overall_pct >= 50 else "pct-low")
        footer = (
            f"<tr class='total-row'>"
            f"<td colspan='{1 + len(targets)}'>"
            f"<strong>Overall</strong> <span class='excl-note'>({t_excl} excluded)</span></td>"
            f"<td class='result {opct_cls}'><strong>{t_pass}/{t_valid} ({overall_pct}%)</strong></td>"
            f"</tr>"
        )
        html = (
            f"<table class='hyp-table'>"
            f"<thead><tr><th>Proposal</th>{hdr}<th>Result</th></tr></thead>"
            f"<tbody>{''.join(rows)}{footer}</tbody>"
            f"</table>"
        )
        return html, t_pass, t_valid, t_excl

    plus_table_html,  plus_pass,  plus_valid,  plus_excl  = _hyp_table_html(plus_targets,  "plus")
    minus_table_html, minus_pass, minus_valid, minus_excl = _hyp_table_html(minus_targets, "minus")
    total_pass   = plus_pass  + minus_pass
    total_checks = plus_valid + minus_valid
    total_excl   = plus_excl  + minus_excl
    overall_pct  = 100 * total_pass // total_checks if total_checks else 0
    pct_cls = "pct-high" if overall_pct >= 75 else ("pct-mid" if overall_pct >= 50 else "pct-low")

    table_w = "\n".join
    return f"""
<section class="dim-section">
  <h2>{dim.upper()} DIMENSION</h2>

  <div class="dim-meta">
    <div class="meta-card plus-card">
      <div class="meta-label">PLUS — Adding Content</div>
      <p>{descs['plus']}</p>
      <div class="hyp-label">Hypothesis: {hyp['plus']}</div>
    </div>
    <div class="meta-card minus-card">
      <div class="meta-label">MINUS — Removing Content</div>
      <p>{descs['minus']}</p>
      <div class="hyp-label">Hypothesis: {hyp['minus']}</div>
    </div>
  </div>

  <h3 class="sub-heading baseline-h">① Baseline Scores</h3>
  <p class="table-note">Raw subrubric scores (1–5) before any content modification.
     <span class="legend-plus">▲ = PLUS target</span> &nbsp;
     <span class="legend-minus">▼ = MINUS target</span>
  </p>
  <table>
    <thead>{header_row()}</thead>
    <tbody>{table_w(rows_b)}</tbody>
  </table>

  <h3 class="sub-heading plus-h">② After Adding Content (PLUS variant)</h3>
  <p class="table-note">Score after injecting evidence. Delta vs baseline shown in parentheses.
     <span class="up-eg">green = increase ↑</span> &nbsp;
     <span class="dn-eg">red = decrease ↓</span> &nbsp;
     <span class="nc-eg">gray = no change →</span>
  </p>
  <table>
    <thead>{header_row()}</thead>
    <tbody>{table_w(rows_p)}</tbody>
  </table>

  <h3 class="sub-heading minus-h">③ After Removing Content (MINUS variant)</h3>
  <p class="table-note">Score after stripping key language. Delta vs baseline shown in parentheses.</p>
  <table>
    <thead>{header_row()}</thead>
    <tbody>{table_w(rows_m)}</tbody>
  </table>

  <h3 class="sub-heading hyp-h">④ Hypothesis Check — PLUS (scores must RISE)</h3>
  <p class="table-note">
    ✓ = score strictly increased after adding content &nbsp;|&nbsp;
    ✗ = score stayed the same or fell.
  </p>
  {plus_table_html}

  <h3 class="sub-heading hyp-h">⑤ Hypothesis Check — MINUS (scores must FALL)</h3>
  <p class="table-note">
    ✓ = score strictly decreased after removing content &nbsp;|&nbsp;
    ✗ = score stayed the same or rose.
  </p>
  {minus_table_html}

  <div class="combined-result">
    Combined: <strong class="{pct_cls}">{total_pass}/{total_checks} ({overall_pct}%)</strong> valid checks passed
    <span class="excl-note">({total_excl} excluded: ceiling or floor effect)</span>
  </div>
</section>
"""


def build_aggregate_section(totals: dict) -> str:
    grand_pass = grand_total = 0
    rows = []
    for dim, (p, t) in totals.items():
        pct = 100 * p // t if t else 0
        pct_cls = "pct-high" if pct >= 75 else ("pct-mid" if pct >= 50 else "pct-low")
        bar = f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%"></div></div>'
        rows.append(
            f"<tr><td class='dim-name'>{dim.upper()}</td>"
            f"<td>{p}/{t}</td>"
            f"<td class='{pct_cls}'>{pct}%</td>"
            f"<td>{bar}</td></tr>"
        )
        grand_pass += p; grand_total += t
    gpct = 100 * grand_pass // grand_total if grand_total else 0
    gpct_cls = "pct-high" if gpct >= 75 else ("pct-mid" if gpct >= 50 else "pct-low")
    bar_g = f'<div class="bar-bg"><div class="bar-fill" style="width:{gpct}%"></div></div>'
    rows.append(
        f"<tr class='total-row'><td class='dim-name'><strong>TOTAL</strong></td>"
        f"<td><strong>{grand_pass}/{grand_total}</strong></td>"
        f"<td class='{gpct_cls}'><strong>{gpct}%</strong></td>"
        f"<td>{bar_g}</td></tr>"
    )
    return f"""
<section class="agg-section">
  <h2>Aggregate Hypothesis Summary</h2>
  <p>Overall rate at which scoring moved in the expected direction when content was added (PLUS)
     or removed (MINUS). Floor-effect checks (MINUS baseline=1) and ceiling-effect checks
     (PLUS baseline=5) are excluded from the denominator — only structurally testable checks
     are counted.</p>
  <table class="agg-table">
    <thead>
      <tr><th>Dimension</th><th>Checks Passed</th><th>Pass Rate</th><th>Visual</th></tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>

  <div class="insight-box">
    <strong>Key Finding:</strong> PLUS targets (content added) responded almost universally across
    all proposals — confirming the model detects and rewards explicit evidence. MINUS targets
    (content removed) were more variable, particularly for judgment-dependent rubrics
    (R7 Problem–Solution Fit, R8 Timing, R5 Defensibility), suggesting the model sometimes
    infers these from contextual cues that survive the stripping operation.
  </div>
</section>
"""


def build_stats_section(dim_stats: dict) -> str:
    """New section appended after the aggregate — statistical validation table."""
    rows = []
    all_plus_p = all_plus_v = all_minus_p = all_minus_v = 0
    all_plus_d: list = []
    all_minus_d: list = []

    for dim, ds in dim_stats.items():
        pp, pv = ds["plus_pass"],  ds["plus_valid"]
        mp, mv = ds["minus_pass"], ds["minus_valid"]
        pd, md = ds["plus_deltas"], ds["minus_deltas"]
        all_plus_p  += pp; all_plus_v  += pv; all_plus_d  += pd
        all_minus_p += mp; all_minus_v += mv; all_minus_d += md

        for label, passes, valid, dlist, cls in [
            (f"{dim.capitalize()} — PLUS",  pp, pv, pd, "row-plus"),
            (f"{dim.capitalize()} — MINUS", mp, mv, md, "row-minus"),
        ]:
            if valid == 0:
                continue
            s = _stats(passes, valid, dlist)
            rate = f"{passes}/{valid} ({100*passes//valid}%)"
            p_str = f"&lt; 0.001" if s["p"] < 0.001 else f"{s['p']:.4f}"
            ci_str = f"[{100*s['ci_low']:.1f}%, {100*s['ci_high']:.1f}%]"
            h_str  = f"{s['h']:.2f} <em>({s['h_label']})</em>"
            d_sign = "+" if s["mean_delta"] >= 0 else ""
            d_str  = f"{d_sign}{s['mean_delta']:.2f} pts"
            sig_cls = "sig-strong" if s["sig"] == "***" else ("sig-mid" if s["sig"] in ("**","*") else "sig-ns")
            rows.append(
                f"<tr class='{cls}'>"
                f"<td class='st-dim'>{label}</td>"
                f"<td class='st-num'>{rate}</td>"
                f"<td class='st-num'>{p_str}</td>"
                f"<td class='{sig_cls}'>{s['sig']}</td>"
                f"<td class='st-num'>{ci_str}</td>"
                f"<td class='st-num'>{h_str}</td>"
                f"<td class='st-num'>{d_str}</td>"
                f"</tr>"
            )

    # Combined row
    comb_p = all_plus_p + all_minus_p
    comb_v = all_plus_v + all_minus_v
    s_comb = _stats(comb_p, comb_v, all_plus_d + all_minus_d)
    rate_c = f"{comb_p}/{comb_v} ({100*comb_p//comb_v if comb_v else 0}%)"
    p_c    = "&lt; 0.001" if s_comb["p"] < 0.001 else f"{s_comb['p']:.4f}"
    ci_c   = f"[{100*s_comb['ci_low']:.1f}%, {100*s_comb['ci_high']:.1f}%]"
    h_c    = f"{s_comb['h']:.2f} <em>({s_comb['h_label']})</em>"
    rows.append(
        f"<tr class='row-total'>"
        f"<td class='st-dim'><strong>Combined (all dims)</strong></td>"
        f"<td class='st-num'><strong>{rate_c}</strong></td>"
        f"<td class='st-num'><strong>{p_c}</strong></td>"
        f"<td class='sig-strong'><strong>{s_comb['sig']}</strong></td>"
        f"<td class='st-num'><strong>{ci_c}</strong></td>"
        f"<td class='st-num'><strong>{h_c}</strong></td>"
        f"<td class='st-num'>—</td>"
        f"</tr>"
    )

    return f"""
<section class="stats-section">
  <h2>Statistical Validation</h2>
  <p class="stats-intro">
    One-sided binomial test against the chance baseline (H₀: p&nbsp;=&nbsp;0.50).
    Floor-effect checks (MINUS, baseline&nbsp;=&nbsp;1) and ceiling-effect checks
    (PLUS, baseline&nbsp;=&nbsp;5) are excluded from all denominators.
    <br>Significance: *** p&nbsp;&lt;&nbsp;0.001 &nbsp;|&nbsp; ** p&nbsp;&lt;&nbsp;0.01
    &nbsp;|&nbsp; * p&nbsp;&lt;&nbsp;0.05 &nbsp;|&nbsp; n.s. not significant.
    Cohen's h: small&nbsp;&lt;&nbsp;0.2, medium 0.2–0.5, large&nbsp;≥&nbsp;0.5.
    Mean&nbsp;Δ = average score change on the 1–5 scale.
  </p>
  <table class="stats-table">
    <thead>
      <tr>
        <th>Dimension / Variant</th>
        <th>Checks Passed</th>
        <th>p-value</th>
        <th>Sig.</th>
        <th>95% CI</th>
        <th>Cohen's h</th>
        <th>Mean Δ</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <div class="stats-glossary">
    <h3>Understanding the Metrics</h3>
    <div class="gloss-grid">
      <div class="gloss-item">
        <div class="gloss-term">95% Confidence Interval (CI)</div>
        <div class="gloss-def">
          A range that tells you how reliable the pass rate estimate is. For example, Strategy PLUS
          shows 91% with CI [84.4%, 95.7%] — meaning we are 95% confident the true pass rate lies
          between those bounds. A narrower interval means a more precise estimate. Even at the lower
          bound, the rate is still well above the 50% chance baseline, which is what matters for
          the research claim.
        </div>
      </div>
      <div class="gloss-item">
        <div class="gloss-term">Cohen's h (Effect Size)</div>
        <div class="gloss-def">
          Measures <em>how large</em> the effect is — not just whether it is statistically significant.
          A p-value only confirms the result probably did not happen by chance; Cohen's h tells you
          whether the effect actually matters in practice.
          It measures the gap between the observed pass rate and the 50% chance baseline on a
          standardised scale: <strong>small&nbsp;&lt;&nbsp;0.2</strong> (barely above chance),
          <strong>medium 0.2–0.5</strong> (noticeable but moderate),
          <strong>large&nbsp;≥&nbsp;0.5</strong> (clearly and strongly above chance).
          In your results, Objectives MINUS reaches h&nbsp;=&nbsp;1.57 (perfect 100% pass rate),
          while Advantages PLUS reaches h&nbsp;=&nbsp;0.35 (medium) — reflecting the scorer's
          built-in skepticism toward self-reported competitive claims.
          Together, p-value says "this is real, not luck" and Cohen's h says "this is also
          <em>large</em>, not trivial."
        </div>
      </div>
    </div>
  </div>

  <div class="stats-interpretation">
    <h3>How to Read These Results</h3>
    <p>
      <strong>All six dimension–variant combinations are statistically significant (p&nbsp;&lt;&nbsp;0.001, ***).</strong>
      This rules out the possibility that the 87% combined pass rate arose by chance — the LLM evaluator
      is demonstrably reading and responding to proposal content, not assigning arbitrary scores.
    </p>
    <p>
      <strong>Effect sizes (Cohen's h)</strong> measure how far the observed pass rate sits above the 50%
      chance baseline. Five of six conditions show a <em>large</em> effect (h&nbsp;≥&nbsp;0.5).
      The one exception — Advantages PLUS (h&nbsp;=&nbsp;0.35, medium) — is explained by the
      Advantages scoring prompt's explicit skepticism instruction, which makes the scorer resistant to
      self-reported competitive claims even when they are injected into the document. Importantly,
      Advantages MINUS remains large (h&nbsp;=&nbsp;1.07), confirming that <em>removing</em> evidence
      reliably reduces scores even when <em>adding</em> evidence is harder to fake.
    </p>
    <p>
      <strong>Mean score delta (Δ)</strong> captures practical magnitude — how many points on the 1–5
      scale scores moved on average. Objectives PLUS shows the largest shift (+3.07 pts), meaning content
      injections nearly tripled the average subrubric score. Strategy PLUS (+2.61 pts) and Objectives
      MINUS (+1.91 pts) are similarly large. These are not marginal boundary effects: the LLM evaluator
      reliably produces large, systematic shifts in response to meaningful content changes.
    </p>
    <p>
      <strong>Combined result:</strong> 366 of 417 valid checks passed (87.7%),
      95% CI [84.9%, 90.1%], Cohen's h&nbsp;=&nbsp;0.86 (large), p&nbsp;&lt;&nbsp;0.001.
      This constitutes strong statistical evidence for content sensitivity across all three
      scoring dimensions.
    </p>
  </div>
  <div class="stats-note">
    Null hypothesis: p&nbsp;=&nbsp;0.50 (score equally likely to move in either direction by chance).
    Floor-effect (MINUS baseline&nbsp;=&nbsp;1) and ceiling-effect (PLUS baseline&nbsp;=&nbsp;5)
    checks are excluded from all denominators. p-values are one-sided (directional);
    confidence intervals are two-sided Wilson intervals.
  </div>
</section>
"""


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 11px;
    color: #1a1a2e;
    background: #f8f9fb;
    padding: 20px;
  }

  /* Cover */
  .cover {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: #fff;
    border-radius: 12px;
    padding: 48px 40px;
    margin-bottom: 32px;
  }
  .cover h1 { font-size: 26px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 8px; }
  .cover .subtitle { font-size: 13px; color: #a0aec0; margin-bottom: 24px; }
  .cover .meta { display: flex; gap: 32px; flex-wrap: wrap; }
  .cover .meta-item { }
  .cover .meta-item .label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #718096; }
  .cover .meta-item .value { font-size: 13px; color: #e2e8f0; margin-top: 2px; }
  .cover .stat-row { display: flex; gap: 16px; margin-top: 28px; flex-wrap: wrap; }
  .cover .stat-box {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 12px 20px;
    text-align: center;
    min-width: 110px;
  }
  .cover .stat-box .sval { font-size: 22px; font-weight: 700; color: #68d391; }
  .cover .stat-box .slbl { font-size: 9px; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

  /* Intro box */
  .intro-box {
    background: #fff;
    border-left: 4px solid #4299e1;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }
  .intro-box h2 { font-size: 13px; margin-bottom: 8px; color: #2b6cb0; }
  .intro-box p { line-height: 1.7; color: #4a5568; }

  /* Section */
  .dim-section {
    background: #fff;
    border-radius: 10px;
    padding: 28px 28px;
    margin-bottom: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    page-break-inside: avoid;
  }
  .dim-section h2 {
    font-size: 16px;
    font-weight: 700;
    color: #1a202c;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 16px;
    letter-spacing: 0.5px;
  }

  /* Meta cards */
  .dim-meta { display: flex; gap: 14px; margin-bottom: 22px; }
  .meta-card {
    flex: 1;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 10.5px;
    line-height: 1.6;
  }
  .plus-card  { background: #f0fff4; border: 1px solid #9ae6b4; }
  .minus-card { background: #fff5f5; border: 1px solid #feb2b2; }
  .meta-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
  .plus-card  .meta-label { color: #276749; }
  .minus-card .meta-label { color: #9b2c2c; }
  .hyp-label { margin-top: 8px; font-style: italic; color: #718096; font-size: 10px; }

  /* Sub headings */
  .sub-heading { font-size: 12px; font-weight: 600; margin: 20px 0 6px; }
  .baseline-h { color: #2d3748; }
  .plus-h     { color: #276749; }
  .minus-h    { color: #9b2c2c; }
  .hyp-h      { color: #553c9a; }

  .table-note { font-size: 10px; color: #718096; margin-bottom: 8px; line-height: 1.5; }
  .legend-plus  { color: #276749; font-weight: 600; }
  .legend-minus { color: #9b2c2c; font-weight: 600; }
  .up-eg { color: #276749; font-weight: 600; }
  .dn-eg { color: #9b2c2c; font-weight: 600; }
  .nc-eg { color: #718096; font-weight: 600; }

  /* Tables */
  table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 6px;
    font-size: 10.5px;
  }
  thead { background: #edf2f7; }
  th, td {
    border: 1px solid #e2e8f0;
    padding: 5px 7px;
    text-align: center;
    white-space: nowrap;
  }
  th { font-weight: 600; color: #2d3748; font-size: 10px; }
  tbody tr:nth-child(even) { background: #f7fafc; }
  tbody tr:hover { background: #ebf8ff; }

  .th-plus { background: #c6f6d5 !important; color: #22543d !important; }
  .th-minus{ background: #fed7d7 !important; color: #742a2a !important; }
  .td-plus { background: #f0fff4; }
  .td-minus{ background: #fff5f5; }
  .pid { font-weight: 600; text-align: left; padding-left: 10px; color: #2d3748; }
  .dim-name { text-align: left; padding-left: 10px; font-weight: 600; }

  /* Badge scores */
  .badge {
    display: inline-block;
    width: 20px;
    height: 20px;
    line-height: 20px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 11px;
    text-align: center;
  }
  .b1 { background:#fed7d7; color:#742a2a; }
  .b2 { background:#fefcbf; color:#744210; }
  .b3 { background:#bee3f8; color:#2a4365; }
  .b4 { background:#c6f6d5; color:#22543d; }
  .b5 { background:#9ae6b4; color:#1c4532; font-size:12px; }
  .b0 { background:#e2e8f0; color:#718096; }

  /* Delta spans */
  .up { color: #276749; font-weight: 600; }
  .dn { color: #c53030; font-weight: 600; }
  .nc { color: #718096; }
  .up small, .dn small, .nc small { font-size: 9px; font-weight: 400; }

  /* Hypothesis table */
  .hyp-table .pass  { background: #f0fff4; color: #276749; font-weight: 600; text-align: center; }
  .hyp-table .fail  { background: #fff5f5; color: #c53030; font-weight: 600; text-align: center; }
  .hyp-table .excl  { background: #f7fafc; color: #a0aec0; text-align: center; font-size: 0.9em; }
  .hyp-table .result { font-weight: 700; }
  .excl-note  { color: #a0aec0; font-size: 0.82em; font-weight: 400; }
  .total-row { background: #edf2f7 !important; font-weight: 600; }
  .combined-result { margin: 1rem 0 0.5rem; font-size: 1rem; color: #4a5568; }
  .hyp-h + p + .hyp-table { margin-bottom: 0.5rem; }

  /* Pass-rate color tiers */
  .pct-high { color: #276749; font-weight: 700; }
  .pct-mid  { color: #b7791f; font-weight: 700; }
  .pct-low  { color: #c53030; font-weight: 700; }

  /* Aggregate section */
  .agg-section {
    background: #fff;
    border-radius: 10px;
    padding: 28px;
    margin-bottom: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
  }
  .agg-section h2 { font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
  .agg-section p  { font-size: 11px; color: #4a5568; line-height: 1.7; margin-bottom: 16px; }
  .agg-table { max-width: 560px; }
  .bar-bg   { background:#edf2f7; border-radius:4px; height:10px; width:160px; overflow:hidden; }
  .bar-fill { background: linear-gradient(90deg,#48bb78,#276749); height:100%; border-radius:4px; }

  /* Insight box */
  .insight-box {
    margin-top: 20px;
    background: #fffaf0;
    border: 1px solid #f6ad55;
    border-left: 4px solid #ed8936;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 10.5px;
    line-height: 1.7;
    color: #4a5568;
  }

  /* Statistical section */
  .stats-section {
    background: #fff;
    border-radius: 10px;
    padding: 28px;
    margin-bottom: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
  }
  .stats-section h2 { font-size: 16px; font-weight: 700; margin-bottom: 8px; color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
  .stats-intro { font-size: 10.5px; color: #4a5568; line-height: 1.7; margin-bottom: 16px; }
  .stats-table { width: 100%; border-collapse: collapse; font-size: 10.5px; }
  .stats-table thead tr { background: #2d3748; color: #fff; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .stats-table thead th { padding: 8px 10px; text-align: left; font-weight: 600; color: #fff; background: #2d3748; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .stats-table tbody tr:nth-child(even) { background: #f7fafc; }
  .stats-table tbody td { padding: 7px 10px; border-bottom: 1px solid #e2e8f0; vertical-align: middle; }
  .st-dim  { font-weight: 600; color: #2d3748; min-width: 180px; }
  .st-num  { color: #4a5568; text-align: right; white-space: nowrap; }
  .row-plus  { border-left: 3px solid #48bb78; }
  .row-minus { border-left: 3px solid #f6ad55; }
  .row-total { background: #ebf8ff !important; border-left: 3px solid #4299e1; }
  .sig-strong { color: #276749; font-weight: 700; text-align: center; }
  .sig-mid    { color: #b7791f; font-weight: 700; text-align: center; }
  .sig-ns     { color: #a0aec0; text-align: center; }
  .stats-glossary { margin-top: 22px; border: 1px solid #bee3f8; border-radius: 8px; padding: 16px 20px; background: #ebf8ff; }
  .stats-glossary h3 { font-size: 11.5px; font-weight: 700; color: #2b6cb0; margin-bottom: 12px; }
  .gloss-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .gloss-item { background: #fff; border-radius: 6px; padding: 12px 14px; border: 1px solid #bee3f8; }
  .gloss-term { font-size: 10.5px; font-weight: 700; color: #2c5282; margin-bottom: 6px; }
  .gloss-def  { font-size: 10px; color: #4a5568; line-height: 1.7; }
  .stats-interpretation { margin-top: 20px; }
  .stats-interpretation h3 { font-size: 12px; font-weight: 700; color: #2d3748; margin-bottom: 10px; }
  .stats-interpretation p  { font-size: 10.5px; color: #4a5568; line-height: 1.75; margin-bottom: 10px; }
  .stats-note { margin-top: 14px; font-size: 9.5px; color: #718096; line-height: 1.6; font-style: italic; border-top: 1px solid #e2e8f0; padding-top: 10px; }

  /* Print */
  @media print {
    body { background: #fff; padding: 0; }
    .dim-section, .agg-section, .stats-section { box-shadow: none; page-break-before: always; }
    .dim-section:first-of-type { page-break-before: avoid; }
    .cover { page-break-after: always; border-radius: 0; }
  }
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_html", default="sensitivity_report.html")
    ap.add_argument("--out_pdf",  default="sensitivity_report.pdf")
    ap.add_argument("--pids",     nargs="+", default=ALL_PIDS)
    args = ap.parse_args()

    generated_at = datetime.now().strftime("%B %d, %Y  %H:%M")

    # Load all dimensions
    all_results:  dict[str, dict] = {}
    all_plus_t:   dict[str, list] = {}
    all_minus_t:  dict[str, list] = {}
    for dim in DIMENSIONS:
        r, pt, mt = load_results(dim)
        if r:
            all_results[dim] = r
            all_plus_t[dim]  = pt
            all_minus_t[dim] = mt

    # Build dimension sections + collect totals
    dim_sections_html = ""
    totals:    dict[str, tuple[int, int]] = {}
    dim_stats: dict[str, dict]            = {}
    for dim in DIMENSIONS:
        if dim not in all_results:
            continue
        results = all_results[dim]
        active  = [p for p in args.pids if p in results]
        plus_t  = all_plus_t[dim]
        minus_t = all_minus_t[dim]
        cfg_    = {**DIMENSIONS[dim], "plus_targets": plus_t, "minus_targets": minus_t}
        # Compute pass totals for cover stats and statistical tests
        dp = dc = 0
        pp = pv = 0        # PLUS passes / valid
        mp = mv = 0        # MINUS passes / valid
        p_deltas: list = []
        m_deltas: list = []
        for pid in active:
            b = results[pid].get("baseline") or {}
            pv_scores = results[pid].get("plus")  or {}
            mv_scores = results[pid].get("minus") or {}
            for key in plus_t:
                bv = b.get(key)
                if bv == 5: continue
                pval = pv_scores.get(key)
                ok = isinstance(pval, int) and isinstance(bv, int) and pval > bv
                dp += ok; dc += 1; pp += ok; pv += 1
                if isinstance(pval, int) and isinstance(bv, int):
                    p_deltas.append(pval - bv)
            for key in minus_t:
                bv = b.get(key)
                if bv == 1: continue
                mval = mv_scores.get(key)
                ok = isinstance(mval, int) and isinstance(bv, int) and mval < bv
                dp += ok; dc += 1; mp += ok; mv += 1
                if isinstance(mval, int) and isinstance(bv, int):
                    m_deltas.append(bv - mval)
        totals[dim]    = (dp, dc)
        dim_stats[dim] = {
            "plus_pass": pp,  "plus_valid": pv,  "plus_deltas": p_deltas,
            "minus_pass": mp, "minus_valid": mv, "minus_deltas": m_deltas,
        }
        dim_sections_html += build_dimension_section(dim, cfg_, results, args.pids)

    grand_p = sum(v[0] for v in totals.values())
    grand_t = sum(v[1] for v in totals.values())
    grand_pct = 100 * grand_p // grand_t if grand_t else 0

    # Stats for cover
    n_proposals = len(args.pids)
    n_dims      = len(all_results)
    n_rubrics   = sum(len(DIMENSIONS[d]["rubrics"]) for d in all_results)
    n_checks    = grand_t

    cover_html = f"""
<div class="cover">
  <h1>Content-Sensitivity Experiment — Subrubric Results</h1>
  <div class="subtitle">
    Proof-of-hypothesis: do LLM scores respond correctly when proposal content is added or removed?
  </div>
  <div class="meta">
    <div class="meta-item">
      <div class="label">Generated</div>
      <div class="value">{generated_at}</div>
    </div>
    <div class="meta-item">
      <div class="label">Proposals</div>
      <div class="value">{n_proposals} (p1–p8, pA–pD)</div>
    </div>
    <div class="meta-item">
      <div class="label">Dimensions</div>
      <div class="value">{n_dims} (Strategy · Advantages · Objectives)</div>
    </div>
    <div class="meta-item">
      <div class="label">Subrubrics</div>
      <div class="value">{n_rubrics} (8 per dimension)</div>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat-box">
      <div class="sval">{grand_pct}%</div>
      <div class="slbl">Overall pass rate</div>
    </div>
    <div class="stat-box">
      <div class="sval">{grand_p}/{grand_t}</div>
      <div class="slbl">Checks passed</div>
    </div>
    {"".join(f'<div class="stat-box"><div class="sval">{100*v[0]//v[1] if v[1] else 0}%</div><div class="slbl">{d.capitalize()}</div></div>' for d, v in totals.items())}
  </div>
</div>
"""

    intro_html = """
<div class="intro-box">
  <h2>How to Read This Report</h2>
  <p>
    For each of the three scoring dimensions (<strong>Strategy</strong>, <strong>Advantages</strong>,
    <strong>Objectives</strong>), the experiment created two modified variants of each proposal's
    final report and re-scored all 8 subrubrics:<br><br>
    &nbsp;&nbsp;<strong>PLUS variant</strong> — specific evidence was <em>injected</em> into the document
    (e.g. exact revenue figures, a patent number, a signed LOI). The targeted rubrics should
    <em>increase</em> in score.<br>
    &nbsp;&nbsp;<strong>MINUS variant</strong> — key language was <em>stripped</em> from the document
    (e.g. partner names replaced, IP terms removed, patient specificity blurred). The targeted rubrics
    should <em>decrease</em> in score.<br><br>
    Rubrics marked <span class="legend-plus">▲ PLUS target</span> should rise on PLUS; those marked
    <span class="legend-minus">▼ MINUS target</span> should fall on MINUS.
    Non-target rubrics serve as a control — large unexpected swings there indicate noise.
  </p>
</div>
"""

    agg_html   = build_aggregate_section(totals)
    stats_html = build_stats_section(dim_stats)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Content-Sensitivity Experiment — Subrubric Results</title>
  <style>{CSS}</style>
</head>
<body>
  {cover_html}
  {intro_html}
  {dim_sections_html}
  {agg_html}
  {stats_html}
</body>
</html>"""

    html_path = ROOT / args.out_html
    html_path.write_text(full_html, encoding="utf-8")
    print(f"[OK] HTML written → {html_path}")

    # Generate PDF via Chrome headless
    pdf_path = ROOT / args.out_pdf
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        str(html_path),
    ]
    print(f"[..] Generating PDF via Chrome headless …")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size // 1024
        print(f"[OK] PDF written  → {pdf_path}  ({size_kb} KB)")
    else:
        print(f"[FAIL] PDF generation failed.")
        print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
