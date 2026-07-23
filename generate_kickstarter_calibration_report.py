#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a detailed HTML + PDF report of the Kickstarter GO-threshold calibration
study: production threshold (transferred, uncalibrated) vs. a Kickstarter-specific
threshold fit via stratified 5-fold cross-validation.

Usage:
    python3 generate_kickstarter_calibration_report.py
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "results" / "KickstarterCalibrated" / "evaluation" / "calibration_results.json"
IMPROVE_PATH = ROOT / "results" / "KickstarterCalibrated" / "evaluation" / "score_improvement_results.json"
ROBUST_PATH = ROOT / "results" / "KickstarterCalibrated" / "evaluation" / "score_robustness_check.json"


# ── Chart / table builders ─────────────────────────────────────────────────────

def metric_row(label, m, highlight=False):
    def fmt_pct(v):
        return f"{v*100:.1f}%" if v is not None else "&mdash;"

    cls = ' class="row-total"' if highlight else ""
    return f"""
    <tr{cls}>
      <td class="pid">{label}</td>
      <td>{m.get('tau_r', '&mdash;')}</td>
      <td>{m.get('tau_c', '&mdash;')}</td>
      <td class="st-num">{m['tp']}</td>
      <td class="st-num">{m['fp']}</td>
      <td class="st-num">{m['fn']}</td>
      <td class="st-num">{m['tn']}</td>
      <td>{fmt_pct(m['accuracy'])}</td>
      <td>{fmt_pct(m['precision'])}</td>
      <td>{fmt_pct(m['recall'])}</td>
      <td>{m['f1'] if m['f1'] is not None else '&mdash;'}</td>
    </tr>"""


def build_comparison_table(data):
    before = data["before_calibration"]
    is_ = data["in_sample_calibration"]
    cv = data["cross_validation"]["aggregated_held_out"]
    cv_disp = {**cv, "tau_r": "varies/fold", "tau_c": "varies/fold"}
    rows = (
        metric_row("Before (production threshold, transferred)", before)
        + metric_row("In-sample calibration (overfitting reference)", is_)
        + metric_row("Cross-validated (held-out, honest result)", cv_disp, highlight=True)
    )
    return f"""
    <table>
      <thead><tr>
        <th>Scenario</th><th>&tau;<sub>r</sub> (score)</th><th>&tau;<sub>c</sub> (confidence)</th>
        <th>TP</th><th>FP</th><th>FN</th><th>TN</th>
        <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_fold_table(data):
    rows_html = []
    for f in data["cross_validation"]["per_fold"]:
        rows_html.append(f"""
        <tr>
          <td class="pid">Fold {f['fold']}</td>
          <td>{f['n_test']}</td>
          <td>{f['n_test_success']}</td>
          <td>{f['tau_r']}</td>
          <td>{f['tau_c']}</td>
          <td>{f['train_f1']}</td>
          <td>{f['f1'] if f['f1'] is not None else '&mdash;'}</td>
          <td>{f'{f["precision"]*100:.0f}%' if f['precision'] is not None else '&mdash;'}</td>
          <td>{f'{f["recall"]*100:.0f}%' if f['recall'] is not None else '&mdash;'}</td>
        </tr>""")
    return f"""
    <table>
      <thead><tr>
        <th>Fold</th><th>Test n</th><th>Test successes</th>
        <th>&tau;<sub>r</sub> fit on train</th><th>&tau;<sub>c</sub> fit on train</th>
        <th>Train F1</th><th>Held-out F1</th><th>Held-out precision</th><th>Held-out recall</th>
      </tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>"""


def build_roc_svg(rows):
    W, H = 420, 420
    pad = 46
    plot = W - 2 * pad

    sorted_rows = sorted(rows, key=lambda r: -r["score"])
    n_pos = sum(r["success"] for r in rows)
    n_neg = len(rows) - n_pos
    pts = [(0.0, 0.0)]
    tp = fp = 0
    for r in sorted_rows:
        if r["success"] == 1:
            tp += 1
        else:
            fp += 1
        pts.append((fp / n_neg, tp / n_pos))

    def X(v):
        return pad + v * plot

    def Y(v):
        return pad + (1 - v) * plot

    path = "M " + " L ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
    diag = f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}" class="roc-diag"/>'

    ticks = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        ticks.append(f'<line x1="{X(t):.1f}" y1="{pad}" x2="{X(t):.1f}" y2="{pad+plot}" class="grid-line"/>')
        ticks.append(f'<text x="{X(t):.1f}" y="{pad+plot+16}" class="axis-label" text-anchor="middle">{t:.2f}</text>')
        ticks.append(f'<line x1="{pad}" y1="{Y(t):.1f}" x2="{pad+plot}" y2="{Y(t):.1f}" class="grid-line"/>')
        ticks.append(f'<text x="{pad-8}" y="{Y(t)+3:.1f}" class="axis-label" text-anchor="end">{t:.2f}</text>')

    return f"""
    <svg viewBox="0 0 {W} {H}" class="roc-svg" role="img" aria-label="ROC curve of AI score discriminating funded vs unfunded campaigns">
      {''.join(ticks)}
      {diag}
      <path d="{path}" class="roc-path"/>
      <text x="{pad+plot/2:.1f}" y="{H-4}" class="axis-title" text-anchor="middle">False positive rate</text>
      <text x="14" y="{pad+plot/2:.1f}" class="axis-title" text-anchor="middle" transform="rotate(-90 14 {pad+plot/2:.1f})">True positive rate</text>
      <rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" class="plot-border"/>
    </svg>"""


def _roc_points(rows, scores):
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    n_pos = sum(r["success"] for r in rows)
    n_neg = len(rows) - n_pos
    pts = [(0.0, 0.0)]
    tp = fp = 0
    for i in order:
        if rows[i]["success"] == 1:
            tp += 1
        else:
            fp += 1
        pts.append((fp / n_neg, tp / n_pos))
    return pts


def build_roc_overlay_svg(rows, series):
    """series: list of (label, css_class, scores) tuples, drawn on one ROC chart."""
    W, H = 460, 460
    pad = 46
    plot = W - 2 * pad

    def X(v):
        return pad + v * plot

    def Y(v):
        return pad + (1 - v) * plot

    ticks = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        ticks.append(f'<line x1="{X(t):.1f}" y1="{pad}" x2="{X(t):.1f}" y2="{pad+plot}" class="grid-line"/>')
        ticks.append(f'<text x="{X(t):.1f}" y="{pad+plot+16}" class="axis-label" text-anchor="middle">{t:.2f}</text>')
        ticks.append(f'<line x1="{pad}" y1="{Y(t):.1f}" x2="{pad+plot}" y2="{Y(t):.1f}" class="grid-line"/>')
        ticks.append(f'<text x="{pad-8}" y="{Y(t)+3:.1f}" class="axis-label" text-anchor="end">{t:.2f}</text>')

    diag = f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}" class="roc-diag"/>'

    paths = []
    legend = []
    for label, cls, scores in series:
        pts = _roc_points(rows, scores)
        path = "M " + " L ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        paths.append(f'<path d="{path}" class="roc-path {cls}"/>')
        legend.append(f'<span class="leg-item"><span class="leg-swatch {cls}"></span>{label}</span>')

    return f"""
    <svg viewBox="0 0 {W} {H}" class="roc-svg roc-svg-lg" role="img" aria-label="ROC curves comparing three scoring methods">
      {''.join(ticks)}
      {diag}
      {''.join(paths)}
      <text x="{pad+plot/2:.1f}" y="{H-4}" class="axis-title" text-anchor="middle">False positive rate</text>
      <text x="14" y="{pad+plot/2:.1f}" class="axis-title" text-anchor="middle" transform="rotate(-90 14 {pad+plot/2:.1f})">True positive rate</text>
      <rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" class="plot-border"/>
    </svg>
    <div class="scatter-legend">{''.join(legend)}</div>"""


def build_score_method_table(data2, data3):
    a, b, c = data2["method_a_overall_score"], data2["method_b_simple_composite"], data2["method_c_logistic_regression"]
    nb = data3["nested_method_b"]

    def row(label, roc_auc, avg_precision, cv, highlight=False, muted=False):
        cls = ' class="row-total"' if highlight else (' style="color:#a0aec0"' if muted else "")
        ap_disp = avg_precision if avg_precision is not None else "&mdash;"
        return f"""
        <tr{cls}>
          <td class="pid">{label}</td>
          <td>{roc_auc}</td>
          <td>{ap_disp}</td>
          <td>{cv['precision']*100:.1f}%</td>
          <td>{cv['recall']*100:.1f}%</td>
          <td>{cv['f1']}</td>
        </tr>"""

    return f"""
    <table>
      <thead><tr>
        <th>Method</th><th>ROC-AUC</th><th>Avg. Precision</th>
        <th>CV Precision</th><th>CV Recall</th><th>CV F1</th>
      </tr></thead>
      <tbody>
        {row("A: overall_score (production, 5 dims)", a['roc_auc'], a['average_precision'], a['cv_aggregated_held_out'])}
        {row("B: simple composite -- naive (dims picked using full dataset)", b['roc_auc'], b['average_precision'], b['cv_aggregated_held_out'], muted=True)}
        {row("C: logistic regression (5 dims, CV-fit)", c['roc_auc'], c['average_precision'], c['cv_aggregated_held_out'])}
        {row("B-nested: simple composite -- honest (dims + threshold both fit per-fold)", nb['roc_auc'], None, nb['cv_aggregated_held_out'], highlight=True)}
      </tbody>
    </table>"""


def build_coef_table(data2):
    coef = data2["method_c_logistic_regression"]["full_dataset_fit_coefficients_for_interpretation_only"]["coef"]
    rows_html = "".join(
        f'<tr><td class="pid">{d.capitalize()}</td><td class="st-num">{v:+.4f}</td></tr>'
        for d, v in sorted(coef.items(), key=lambda kv: -kv[1])
    )
    return f"""
    <table style="max-width:340px">
      <thead><tr><th>Dimension</th><th>Learned weight</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# ── CSS (shared conventions with generate_kickstarter_report.py) ──────────────

CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; font-size: 11px; color: #0b0b0b; background: #f9f9f7; padding: 20px; }

  .cover { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%); color: #fff; border-radius: 12px; padding: 48px 40px; margin-bottom: 32px; }
  .cover h1 { font-size: 25px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 8px; }
  .cover .subtitle { font-size: 13px; color: #a0aec0; margin-bottom: 24px; line-height: 1.6; }
  .cover .meta { display: flex; gap: 32px; flex-wrap: wrap; }
  .cover .meta-item .label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #718096; }
  .cover .meta-item .value { font-size: 13px; color: #e2e8f0; margin-top: 2px; }
  .cover .stat-row { display: flex; gap: 16px; margin-top: 28px; flex-wrap: wrap; }
  .cover .stat-box { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 12px 20px; text-align: center; min-width: 118px; }
  .cover .stat-box .sval { font-size: 21px; font-weight: 700; color: #68d391; }
  .cover .stat-box.warn .sval { color: #fc8181; }
  .cover .stat-box .slbl { font-size: 9px; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

  .intro-box { background: #fff; border-left: 4px solid #2a78d6; border-radius: 8px; padding: 20px 24px; margin-bottom: 28px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
  .intro-box h2 { font-size: 13px; margin-bottom: 8px; color: #184f95; }
  .intro-box p { line-height: 1.7; color: #4a5568; margin-bottom: 10px; }
  .intro-box p:last-child { margin-bottom: 0; }

  .section { background: #fff; border-radius: 10px; padding: 28px; margin-bottom: 28px; box-shadow: 0 2px 8px rgba(0,0,0,.07); page-break-inside: avoid; }
  .section h2 { font-size: 16px; font-weight: 700; color: #1a202c; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; margin-bottom: 16px; }
  .section h3 { font-size: 12.5px; font-weight: 700; color: #2d3748; margin: 18px 0 8px; }
  .section p { line-height: 1.75; color: #4a5568; margin-bottom: 10px; }
  .section p:last-child { margin-bottom: 0; }
  .plain { background: #ebf8ff; border: 1px solid #bee3f8; border-radius: 8px; padding: 12px 16px; font-size: 10.5px; color: #2c5282; line-height: 1.7; margin: 10px 0 16px; page-break-inside: avoid; }
  .plain b { color: #184f95; }

  table { border-collapse: collapse; width: 100%; margin-bottom: 6px; font-size: 10px; page-break-inside: avoid; }
  thead { background: #edf2f7; }
  th, td { border: 1px solid #e2e8f0; padding: 6px 7px; text-align: center; }
  th { font-weight: 600; color: #2d3748; font-size: 9.5px; }
  tbody tr:nth-child(even) { background: #f7fafc; }
  .pid { font-weight: 600; text-align: left; padding-left: 10px; color: #2d3748; }
  .st-num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 12px; }
  .row-total { background: #ebf8ff !important; border-left: 3px solid #2a78d6; font-weight: 600; }

  .stat-cards { display: flex; gap: 14px; margin: 14px 0; }
  .stat-card { flex: 1; background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; text-align: center; }
  .stat-card .cval { font-size: 20px; font-weight: 700; color: #184f95; }
  .stat-card .clbl { font-size: 9.5px; color: #718096; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 4px; }
  .stat-card .csub { font-size: 9.5px; color: #a0aec0; margin-top: 3px; }

  .roc-svg { width: 320px; height: auto; display: block; margin: 10px auto; }
  .roc-svg-lg { width: 360px; }
  .grid-line { stroke: #e1e0d9; stroke-width: 1; }
  .axis-label { font-size: 9px; fill: #898781; }
  .axis-title { font-size: 10px; fill: #52514e; font-weight: 600; }
  .plot-border { fill: none; stroke: #c3c2b7; stroke-width: 1; }
  .roc-diag { stroke: #c3c2b7; stroke-width: 1.4; stroke-dasharray: 4 3; }
  .roc-path { fill: none; stroke: #2a78d6; stroke-width: 2.4; }
  .roc-path.m-a { stroke: #898781; stroke-dasharray: 3 2; }
  .roc-path.m-b { stroke: #cbd5e0; stroke-width: 2; stroke-dasharray: 2 2; }
  .roc-path.m-c { stroke: #eda100; }
  .roc-path.m-nb { stroke: #0ca30c; stroke-width: 3; }
  .scatter-legend { display: flex; gap: 20px; justify-content: center; margin-top: 8px; font-size: 10px; color: #4a5568; flex-wrap: wrap; }
  .leg-item { display: flex; align-items: center; gap: 6px; }
  .leg-swatch { width: 16px; height: 3px; display: inline-block; border-radius: 2px; }
  .leg-swatch.m-a { background: #898781; }
  .leg-swatch.m-b { background: #cbd5e0; }
  .leg-swatch.m-c { background: #eda100; }
  .leg-swatch.m-nb { background: #0ca30c; }
  .boot-sig { color: #a0aec0; font-weight: 600; }
  .boot-notsig { color: #b7791f; font-weight: 600; }

  .outcome-box { border-radius: 8px; padding: 16px 20px; margin-bottom: 14px; }
  .outcome-good { background: #f0fff4; border-left: 4px solid #0ca30c; }
  .outcome-bad { background: #fff5f5; border-left: 4px solid #d03b3b; }
  .outcome-box h4 { font-size: 11.5px; font-weight: 700; margin-bottom: 6px; }
  .outcome-good h4 { color: #006300; }
  .outcome-bad h4 { color: #9b2c2c; }
  .outcome-box ul, .findings-list ul { margin-left: 18px; line-height: 1.8; color: #4a5568; }
  .outcome-box li, .findings-list li { margin-bottom: 6px; }

  .findings-list { background: #fffaf0; border: 1px solid #f6ad55; border-left: 4px solid #ed8936; border-radius: 8px; padding: 16px 20px; }
  .findings-list li b { color: #1a202c; }

  .glossary { margin-top: 10px; border: 1px solid #bee3f8; border-radius: 8px; padding: 14px 18px; background: #ebf8ff; }
  .glossary h3 { font-size: 11.5px; font-weight: 700; color: #184f95; margin-bottom: 10px; }
  .gloss-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .gloss-item { background: #fff; border-radius: 6px; padding: 10px 12px; border: 1px solid #bee3f8; }
  .gloss-term { font-size: 10.5px; font-weight: 700; color: #184f95; margin-bottom: 4px; }
  .gloss-def { font-size: 9.5px; color: #4a5568; line-height: 1.6; }

  .footnote { font-size: 9.5px; color: #898781; font-style: italic; line-height: 1.6; margin-top: 10px; border-top: 1px solid #e2e8f0; padding-top: 8px; }

  .fold-diagram { display: flex; gap: 6px; margin: 14px 0; }
  .fold-cell { flex: 1; height: 34px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700; }
  .fold-train { background: #e2e8f0; color: #4a5568; }
  .fold-test { background: #2a78d6; color: #fff; }

  @media print {
    body { background: #fff; padding: 0; }
    .section { box-shadow: none; }
    .cover { border-radius: 0; }
  }
"""


# ── Section builders ────────────────────────────────────────────────────────

def build_cover(data, data2, data3, generated_at):
    before = data["before_calibration"]
    cv = data["cross_validation"]["aggregated_held_out"]
    nb = data3["nested_method_b"]
    return f"""
    <div class="cover">
      <h1>Kickstarter GO-Threshold Calibration Study</h1>
      <div class="subtitle">Recalibrating the pipeline's success-prediction threshold for the Kickstarter domain,
      validated with a proper train/test split (5-fold cross-validation) rather than in-sample tuning -- plus a
      bootstrap-checked test of whether a better composite score can beat the production score outright.</div>
      <div class="meta">
        <div class="meta-item"><div class="label">Generated</div><div class="value">{generated_at}</div></div>
        <div class="meta-item"><div class="label">Dataset</div><div class="value">141 Kickstarter proposals</div></div>
        <div class="meta-item"><div class="label">Validation</div><div class="value">Stratified 5-fold cross-validation</div></div>
      </div>
      <div class="stat-row">
        <div class="stat-box"><div class="sval">{data['discrimination']['roc_auc']:.3f}</div><div class="slbl">ROC-AUC, production score</div></div>
        <div class="stat-box warn"><div class="sval">{before['recall']*100:.0f}%</div><div class="slbl">Recall, before calibration</div></div>
        <div class="stat-box"><div class="sval">{cv['recall']*100:.0f}%</div><div class="slbl">Recall, after (held-out)</div></div>
        <div class="stat-box"><div class="sval">{cv['precision']*100:.0f}%</div><div class="slbl">Precision, after (held-out)</div></div>
        <div class="stat-box"><div class="sval">{nb['roc_auc']:.3f}</div><div class="slbl">ROC-AUC, honest composite (nested)</div></div>
      </div>
    </div>"""


def build_intro():
    return """
    <div class="intro-box">
      <h2>What is this report, in plain terms?</h2>
      <p>In the earlier Kickstarter validation, the pipeline's GO/HOLD/NO-GO decision was made using a threshold
      (score &ge; 0.725 OR confidence &ge; 0.811) that had been calibrated on a completely different dataset
      &mdash; 12 biotech/semiconductor proposals. That threshold never fired "GO" on any of the 141 Kickstarter
      campaigns, even ones that actually succeeded, because Kickstarter proposals structurally score lower on
      this rubric than VC-style proposals do (explained in the previous report).</p>
      <p>This report asks a different, more useful question: <b>if we pick a decision threshold that's actually
      appropriate for the Kickstarter score distribution, does the AI's score have real power to separate winners
      from losers?</b> To answer that honestly, we can't just search for the "best" threshold and then grade
      ourselves on the same data we used to find it &mdash; that guarantees a flattering but meaningless number.
      Instead we use <b>train/test evaluation</b>: fit the threshold on part of the data, and check it on a
      different part it never saw. This report walks through exactly how that was done, and what it found.</p>
    </div>"""


def build_methodology():
    return """
    <div class="section">
      <h2>1. Methodology: How the Threshold Was Calibrated and Tested</h2>
      <h3>The decision rule being calibrated</h3>
      <p>The pipeline's verdict rule has always had this shape: predict a successful proposal ("GO") if
      <b>score &ge; &tau;<sub>r</sub> OR confidence &ge; &tau;<sub>c</sub></b>. The production system uses
      &tau;<sub>r</sub>=0.725 and &tau;<sub>c</sub>=0.811, calibrated on the 12-proposal biotech dataset. This
      study searches for better values of &tau;<sub>r</sub> and &tau;<sub>c</sub> specifically for Kickstarter,
      keeping the same OR-logic shape so it's a direct, like-for-like recalibration rather than a new mechanism.</p>

      <h3>Why train/test split, not just "pick the best threshold and report it"</h3>
      <div class="plain">
        <b>In simple terms:</b> if you try every possible threshold on all 141 proposals and report whichever one
        scores best on those same 141 proposals, you are guaranteed to get a good-looking number &mdash; but it
        tells you nothing about whether that threshold will work on the <i>next</i> Kickstarter campaign it hasn't
        seen. That's called overfitting. The fix is to test the threshold on data it was not calibrated on.
      </div>
      <p>With only 141 proposals and just 24 real successes, a single 70/30 train/test split would leave as few
      as 7 successes in the test set &mdash; too few to trust a precision/recall estimate. Instead we use
      <b>stratified 5-fold cross-validation</b>: the 141 proposals are split into 5 roughly equal groups
      ("folds"), each containing a proportional share of successes (~4-5 per fold, matching the 17% overall
      success rate). We then repeat the following 5 times, once per fold:</p>
      <ol style="margin-left:18px; line-height:1.9; color:#4a5568;">
        <li>Hold one fold out as the <b>test set</b> (never touched during calibration).</li>
        <li>Use the remaining 4 folds (the <b>training set</b>) to grid-search for the &tau;<sub>r</sub>,
        &tau;<sub>c</sub> pair that maximizes F1 score &mdash; not accuracy (explained below).</li>
        <li>Apply that threshold to the held-out test fold and record how it did.</li>
      </ol>
      <p>Every one of the 141 proposals ends up in exactly one test fold, evaluated using a threshold it never
      influenced. The 5 test-fold results are then combined into one overall confusion matrix &mdash; this is the
      honest, out-of-sample "after calibration" result in this report.</p>

      <div class="fold-diagram">
        <div class="fold-cell fold-test">Fold 1 = test</div>
        <div class="fold-cell fold-train">2</div>
        <div class="fold-cell fold-train">3</div>
        <div class="fold-cell fold-train">4</div>
        <div class="fold-cell fold-train">5</div>
      </div>
      <p style="font-size:9.5px;color:#718096;margin-top:-6px">Illustration of round 1 of 5: fold 1 held out as
      test, folds 2-5 used to fit the threshold. This repeats 5 times, rotating which fold is held out, so every
      proposal is tested exactly once.</p>

      <h3>Why F1 instead of accuracy as the calibration objective</h3>
      <p>Only 17% of Kickstarter campaigns in this dataset actually succeeded. A model that predicts "will fail"
      for every single proposal already scores 83% accuracy without ever identifying a real success &mdash; which
      is exactly the failure mode of the original transferred threshold. <b>F1 score</b> (the balance of precision
      and recall) forces the calibration to actually find successes, not just avoid false alarms, so it was used
      as the objective for selecting &tau;<sub>r</sub> and &tau;<sub>c</sub> on each training fold.</p>

      <h3>An additional, threshold-independent check</h3>
      <p>Before even picking a threshold, we also computed <b>ROC-AUC</b> and <b>Average Precision</b> directly
      from the raw AI score, with no threshold involved at all. These answer a more fundamental question: does the
      score contain real information about funding success, regardless of where any cutoff is drawn?</p>
    </div>"""


def build_discrimination(data):
    auc = data["discrimination"]["roc_auc"]
    ap = data["discrimination"]["average_precision"]
    sr = data["success_rate"]
    return f"""
    <div class="section">
      <h2>2. Does the Score Have Real Signal? (Threshold-Free Check)</h2>
      <div class="stat-cards">
        <div class="stat-card"><div class="cval">{auc:.3f}</div><div class="clbl">ROC-AUC</div><div class="csub">0.5 = random, 1.0 = perfect</div></div>
        <div class="stat-card"><div class="cval">{ap:.3f}</div><div class="clbl">Average Precision</div><div class="csub">vs. {sr*100:.0f}% random baseline</div></div>
      </div>
      {build_roc_svg(data["per_proposal"])}
      <div class="plain">
        <b>In simple terms:</b> ROC-AUC of {auc:.3f} means that if you picked one successful campaign and one
        failed campaign at random, the AI would rank the successful one higher <b>{auc*100:.0f}% of the time</b>.
        0.5 would mean the score is no better than a coin flip; {auc:.3f} is a real, meaningful signal (a widely
        used rule of thumb treats 0.7&ndash;0.8 as "fair to good" discrimination). Average precision of {ap:.3f}
        is roughly <b>{ap/sr:.1f}&times;</b> better than randomly guessing, given that only {sr*100:.0f}% of
        campaigns actually succeed. This confirms the earlier Spearman correlation finding from a different angle:
        the score is informative &mdash; the earlier 0% recall result was a threshold-placement problem, not a
        sign that the score itself carries no information.
      </div>
    </div>"""


def build_before_after(data):
    return f"""
    <div class="section">
      <h2>3. Before vs. After Calibration</h2>
      <p>All three rows below use the exact same underlying AI scores and the exact same OR-logic decision rule
      &mdash; only the threshold values differ.</p>
      {build_comparison_table(data)}
      <div class="plain">
        <b>Before</b> (production threshold, unmodified): 0% recall &mdash; confirmed in the prior report.<br>
        <b>In-sample</b> (threshold fit and tested on the same 141 proposals): looks great (91.7% recall, F1=0.478)
        but this number is <b>not trustworthy</b> &mdash; it's graded on the data it was tuned to fit.<br>
        <b>Cross-validated / held-out</b> (the honest result): recall rises from 0% to <b>58.3%</b>, precision
        lands at <b>26.4%</b> (vs. a 17.0% random baseline &mdash; a real, if modest, lift), and F1 improves from
        undefined to <b>0.364</b>. This is the number that should be quoted as the pipeline's actual, generalizable
        performance on Kickstarter-style proposals.
      </div>
    </div>"""


def build_fold_stability(data):
    return f"""
    <div class="section">
      <h2>4. Per-Fold Detail: How Stable Is the Calibrated Threshold?</h2>
      <p>The table below shows what threshold was chosen in each of the 5 training rounds, and how it performed
      on that round's held-out test fold. This is useful transparency: it shows whether the "right" threshold is
      consistent across different subsets of the data, or whether it's jumping around (a sign of an unstable
      estimate given the small number of successes).</p>
      {build_fold_table(data)}
      <div class="plain">
        <b>In simple terms:</b> &tau;<sub>r</sub> (the score cutoff) stays tight across all 5 folds (0.55&ndash;0.61),
        which is reassuring &mdash; it means the "right" score cutoff isn't just an artifact of which proposals
        happened to be in the training set. Held-out F1 per fold is noisier (0.0 to 0.47), which is expected with
        only ~5 successes per test fold &mdash; a single missed or extra success swings the fold's F1 a lot. This
        is exactly why the <b>aggregated</b> result (pooling all 5 test folds into one confusion matrix, Section 3)
        is the number to trust, not any individual fold.
      </div>
    </div>"""


def build_bootstrap_table(data3):
    rows_html = []
    labels = {"B_vs_A": "B (naive) vs. A", "B_vs_C": "B (naive) vs. C", "C_vs_A": "C vs. A"}
    for key, label in labels.items():
        v = data3["bootstrap"][key]
        sig_cls = "boot-sig" if v["significant_at_0.05"] else "boot-notsig"
        sig_txt = "yes" if v["significant_at_0.05"] else "no"
        rows_html.append(f"""
        <tr>
          <td class="pid">{label}</td>
          <td class="st-num">{v['mean_diff']:+.4f}</td>
          <td>[{v['ci_95_lower']:+.4f}, {v['ci_95_upper']:+.4f}]</td>
          <td>{v['prob_method_2_better']*100:.1f}%</td>
          <td class="{sig_cls}">{sig_txt}</td>
        </tr>""")
    return f"""
    <table>
      <thead><tr>
        <th>Comparison (AUC difference)</th><th>Mean diff</th><th>95% CI</th>
        <th>P(2nd method better)</th><th>Statistically significant?</th>
      </tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>"""


def build_dim_stability_table(data3):
    rows_html = []
    for f in data3["dimension_stability"]["per_fold"]:
        match_cls = "boot-sig" if f["matches_fixed_selection"] else "boot-notsig"
        match_txt = "yes" if f["matches_fixed_selection"] else "no"
        rows_html.append(f"""
        <tr>
          <td class="pid">Fold {f['fold']}</td>
          <td>{', '.join(f['top3_selected'])}</td>
          <td class="{match_cls}">{match_txt}</td>
        </tr>""")
    return f"""
    <table style="max-width:520px">
      <thead><tr><th>Fold</th><th>Top-3 dimensions (by train-fold correlation)</th><th>Matches feasibility/strategy/advantages?</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>"""


def build_score_improvement(data2, data3):
    a = data2["method_a_overall_score"]
    b = data2["method_b_simple_composite"]
    c = data2["method_c_logistic_regression"]
    nb = data3["nested_method_b"]
    pp = data2["per_proposal"]
    nested_lookup = {r["pid"]: r["nested_score"] for r in nb["per_proposal_scores"]}
    series = [
        ("A: overall_score (production)", "m-a", [r["overall_score"] for r in pp]),
        ("B: simple composite -- naive", "m-b", [r["simple_composite"] for r in pp]),
        ("C: logistic regression (5 dims, CV-fit)", "m-c", [r["logreg_oof_proba"] for r in pp]),
        ("B-nested: simple composite -- honest", "m-nb", [nested_lookup[r["pid"]] for r in pp]),
    ]
    n_match = data3["dimension_stability"]["n_folds_matching_fixed_selection"]
    k_folds = data3["dimension_stability"]["k_folds"]
    return f"""
    <div class="section">
      <h2>5. Improving the Score: Can a Better Composite Beat the Production Score?</h2>
      <p>The earlier sections calibrate <i>where to draw the line</i> on the existing AI score. This section asks a
      different question: can the underlying score itself be improved? The production <code>overall_score</code>
      is a weighted average across all 5 rubric dimensions (team, objective, strategy, advantages, feasibility) --
      but the earlier correlation study found that <b>team and objective show no significant relationship</b> with
      real Kickstarter outcomes, while feasibility, strategy, and advantages do. Averaging in two noisy dimensions
      dilutes the signal in the other three.</p>
      <p>Three scoring methods were tested head-to-head, all evaluated with the same honest 5-fold cross-validation
      used throughout this report (same folds, same seed):</p>
      <ul style="margin-left:18px; line-height:1.9; color:#4a5568;">
        <li><b>Method A -- overall_score:</b> the unmodified production score (baseline).</li>
        <li><b>Method B -- simple composite (naive):</b> a plain, unweighted average of just feasibility + strategy
        + advantages. Team/objective were dropped using a correlation study computed on the <i>full</i> 141-proposal
        dataset -- fast to compute, but as shown below, that's a softer form of the same leakage problem this
        report exists to catch.</li>
        <li><b>Method C -- logistic regression:</b> an L2-regularized model that learns its own weights across all 5
        dimensions, fit strictly within each training fold (never seeing its own test fold), producing pooled
        out-of-fold predicted probabilities.</li>
        <li><b>Method B-nested -- the honest version:</b> the same idea as Method B, but with the "which 3
        dimensions?" decision <i>also</i> re-made independently within each training fold, exactly like the
        threshold and the logistic regression weights already were.</li>
      </ul>

      {build_score_method_table(data2, data3)}
      {build_roc_overlay_svg(pp, series)}

      <div class="plain">
        <b>The naive number was inflated.</b> Method B looked strong at first (ROC-AUC {b['roc_auc']:.3f}) because
        picking "feasibility, strategy, advantages" using the entire dataset gives the composite a small,
        undisclosed head start. Once dimension selection is done honestly -- fit per fold, like everything else in
        this report -- the real number is <b>ROC-AUC {nb['roc_auc']:.3f}</b>: still directionally better than the
        production score ({a['roc_auc']:.3f}), but a much smaller edge than first reported.
      </div>

      <h3>Is the improvement real, or just luck?</h3>
      <p>With only 24 successful campaigns to test against, a difference that looks real could still be sampling
      noise. To check, each pairwise AUC difference was bootstrapped 10,000 times (resample the 141 proposals with
      replacement, recompute both AUCs, repeat) to get a confidence interval on the gap.</p>
      {build_bootstrap_table(data3)}
      <div class="plain">
        <b>In simple terms:</b> none of the three comparisons reach conventional statistical significance -- every
        95% confidence interval includes zero, just barely. Method B does look better than A about 95% of the time
        across resamples, which is suggestive of a real effect, but it falls just short of the standard bar for
        "proven." The honest conclusion is <b>"a promising, directionally consistent improvement that has not yet
        been shown to be more than chance"</b> -- not a confirmed result.
      </div>

      <h3>Did picking those 3 dimensions "cheat" by looking at all the data?</h3>
      <p>Separately, the dimension-selection step itself was re-run using <i>only</i> each training fold's data, to
      check whether "feasibility, strategy, advantages" is a stable choice or an artifact of looking at the
      full dataset at once.</p>
      {build_dim_stability_table(data3)}
      <div class="plain">
        <b>In simple terms:</b> {n_match} of {k_folds} folds independently picked the exact same three dimensions
        using only their own training data -- reassuring stability, not a full-dataset fluke. The one exception
        (fold 4) swapped "advantages" for "team," showing the choice isn't perfectly rock-solid at the margin
        either.
      </div>

      <h3>Why the learned model didn't win</h3>
      <p>The logistic regression's own coefficients (fit on the full dataset, shown below for interpretation only
      -- not used in the evaluation above) explain why: it does lean more on feasibility and strategy, but L2
      regularization keeps every dimension's weight in a similar, modest range rather than sharply zeroing out
      team and objective.</p>
      {build_coef_table(data2)}
      <div class="plain">
        <b>In simple terms:</b> with only 24 real successes to learn from (about 19 per training fold), there
        isn't enough data for the model to become <i>confident</i> that team and objective are irrelevant, so it
        hedges and keeps a little weight on everything. The nested version of Method B avoids that hedging problem
        differently -- not by using outside knowledge for free, but by re-deriving "which dimensions matter" from
        scratch on each training fold and checking it's stable (Section 5 above: {n_match}/{k_folds} folds agree).
        That's a fair comparison to Method C's per-fold-fit weights, and it still comes out slightly ahead
        ({nb['roc_auc']:.3f} vs {c['roc_auc']:.3f}) -- a small, honestly-earned edge, worth stating plainly as
        <b>a modest, not-yet-statistically-confirmed sign that domain-informed feature selection may generalize
        better than unconstrained learning in this small-sample regime.</b>
      </div>
    </div>"""


def build_outcomes(data, data2, data3):
    cv = data["cross_validation"]["aggregated_held_out"]
    before = data["before_calibration"]
    auc = data["discrimination"]["roc_auc"]
    nb = data3["nested_method_b"]
    boot = data3["bootstrap"]["B_vs_A"]
    return f"""
    <div class="section">
      <h2>6. Positive and Negative Outcomes (Updated)</h2>
      <p>Framed for a research paper, incorporating this calibration study alongside the correlation results from
      the earlier Kickstarter report.</p>

      <div class="outcome-box outcome-good">
        <h4>Positive outcomes</h4>
        <ul>
          <li>The AI score has genuine, threshold-independent discriminative power: <b>ROC-AUC = {auc:.3f}</b> on
          141 real campaigns &mdash; well above the 0.5 random baseline.</li>
          <li>A properly cross-validated, out-of-sample recalibration lifts recall on real successes from
          <b>0% to 58.3%</b>, and precision from undefined to <b>26.4%</b> (a ~1.55&times; enrichment over the
          17.0% base rate) &mdash; using held-out test folds only, never in-sample.</li>
          <li>The recalibrated score threshold (&tau;<sub>r</sub> &asymp; 0.55&ndash;0.61) is stable across all 5
          cross-validation folds, suggesting it reflects a real property of the score distribution rather than
          noise in one particular data split.</li>
          <li>This directly demonstrates that the earlier "0% recall" finding was a <b>threshold transfer problem</b>,
          not evidence the pipeline's scoring is uninformative for this domain &mdash; an important, correctable
          distinction for the paper's narrative.</li>
          <li>Dropping the two dimensions independently shown to lack predictive validity (team, objective) and
          simply averaging the other three raises ROC-AUC from {auc:.3f} to <b>{nb['roc_auc']:.3f}</b>, evaluated
          fully out-of-sample with dimension selection re-derived per fold (not the inflated {data2['method_b_simple_composite']['roc_auc']:.3f}
          a naive full-dataset selection would suggest) &mdash; a real, if modest, directional edge over both the
          production score and a learned logistic-regression alternative (Section 5).</li>
          <li>The dimension selection behind that improvement is stable: {data3['dimension_stability']['n_folds_matching_fixed_selection']}
          of {data3['dimension_stability']['k_folds']} independent training folds picked the same three dimensions
          on their own, without seeing the full dataset.</li>
        </ul>
      </div>

      <div class="outcome-box outcome-bad">
        <h4>Negative outcomes / limitations</h4>
        <ul>
          <li>Precision remains modest (26.4%) &mdash; roughly 3 in 4 campaigns flagged as likely-successful still
          fail. This is not a strong enough signal to automate a go/no-go decision; it is, at best, a triage aid.</li>
          <li>The gap between in-sample (F1=0.478) and held-out (F1=0.364) performance is a clean, quotable
          illustration of overfitting risk &mdash; and a reason to be skeptical of any threshold reported without
          a train/test split, including the biotech dataset's own OR-logic threshold, which was calibrated
          in-sample on all 12 proposals with no held-out check.</li>
          <li>Fold-level F1 is noisy (0.0&ndash;0.47) because each held-out fold contains only ~5 real successes
          &mdash; the aggregated result is more trustworthy than any individual fold, but the overall sample is
          still small for the positive class, and confidence intervals on precision/recall would be wide.</li>
          <li>This calibration is specific to Kickstarter's score distribution; it should not be assumed to
          transfer to a third domain any more than the original biotech threshold transferred to Kickstarter.</li>
          <li>The composite-score improvement (Section 5) does <b>not</b> reach statistical significance: a
          10,000-sample bootstrap on the ROC-AUC gap gives a 95% CI of [{boot['ci_95_lower']:+.4f}, {boot['ci_95_upper']:+.4f}],
          which includes zero. It should be reported as a promising, directionally consistent signal (~95%
          bootstrap probability of being real) &mdash; not a confirmed improvement &mdash; until tested on a larger
          sample.</li>
        </ul>
      </div>
    </div>"""


def build_findings(data3):
    nb = data3["nested_method_b"]
    n_match = data3["dimension_stability"]["n_folds_matching_fixed_selection"]
    k_folds = data3["dimension_stability"]["k_folds"]
    return f"""
    <div class="section">
      <h2>7. Interesting Findings Worth Highlighting in a Paper</h2>
      <div class="findings-list">
        <ul>
          <li><b>Domain-shift threshold failure, precisely diagnosed and fixed:</b> a threshold transferred
          unmodified across domains produced 0% recall; recalibrating the same decision rule (same OR-logic shape,
          new cutoffs) on the target domain, validated out-of-sample, restored 58.3% recall &mdash; a clean
          before/after story about covariate shift in LLM-based scoring rubrics.</li>
          <li><b>Threshold-free vs. threshold-based evidence tell a consistent story:</b> ROC-AUC (0.764,
          threshold-independent) and the earlier Spearman correlation (0.366, rank-based) both independently
          support that the score carries real signal &mdash; while the naive classification accuracy (83%, driven
          by base rate) and the original 0% recall both independently misled in the opposite direction. Worth
          discussing as a methods lesson: which metrics are trustworthy under class imbalance and domain shift,
          and which are not.</li>
          <li><b>An honest overfitting demonstration, in-repo:</b> the in-sample calibration (F1=0.478) vs.
          held-out cross-validation (F1=0.364) gap is a small, self-contained illustration of exactly the failure
          mode the paper should be careful to avoid elsewhere &mdash; including, arguably, in its own existing
          12-proposal biotech threshold calibration, which was never validated out-of-sample.</li>
          <li><b>Small positive-class sizes make single train/test splits unreliable:</b> with only 24 successes
          in 141 proposals, stratified k-fold cross-validation was necessary to get a stable estimate at all &mdash;
          a methodological point that generalizes to any future outcome-based validation this pipeline undergoes
          on similarly rare-success datasets.</li>
          <li><b>A quiet leak, caught by nesting the whole pipeline:</b> a "zero-fitting" composite score that
          simply dropped two known-irrelevant dimensions still turned out to be quietly leaking information -- its
          feature-selection step used the full dataset's correlations, inflating ROC-AUC from an honest {nb['roc_auc']:.3f}
          to an overstated 0.805. Re-deriving the dimension choice independently within each fold ({n_match}/{k_folds}
          folds agreed) recovered a defensible, smaller number. A concrete illustration that leakage doesn't
          require model fitting to sneak in -- even a one-time, seemingly "free" data-driven design choice can
          leak if it touches the evaluation set.</li>
          <li><b>A modest, not-yet-significant improvement, reported honestly:</b> the fully-nested composite score
          beats the production score on point estimate (ROC-AUC {nb['roc_auc']:.3f} vs 0.764) with a ~95% bootstrap
          probability of being a real effect, but the 95% confidence interval on the difference still includes
          zero at this sample size. Presenting a result at exactly this level of confidence, rather than rounding
          up to "proven," is itself a useful methodological example for the paper's broader trustworthiness
          argument.</li>
        </ul>
      </div>
    </div>"""


def build_glossary():
    return """
    <div class="section">
      <h2>Glossary: Train/Test and Related Terms (Plain-Language)</h2>
      <div class="glossary">
        <div class="gloss-grid">
          <div class="gloss-item">
            <div class="gloss-term">Train set / test set</div>
            <div class="gloss-def">The train set is the data used to pick the threshold. The test set is
            different data, held back, used only to check how well that threshold actually performs. Using the
            same data for both makes results look better than they really are.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">K-fold cross-validation</div>
            <div class="gloss-def">Splits the data into k equal groups (folds). Repeats the train/test process k
            times, each time using a different fold as the test set and the rest as training. Every data point
            gets tested exactly once, using a threshold it never influenced.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">Stratified</div>
            <div class="gloss-def">When splitting into folds, we made sure each fold has roughly the same
            proportion of successes as the full dataset (~17%), so no fold ends up with zero or almost-zero
            successes to test against.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">Overfitting</div>
            <div class="gloss-def">When a threshold (or model) is tuned so precisely to one specific dataset that
            it stops generalizing to new data. The in-sample vs. held-out gap in this report (F1 0.478 vs 0.364)
            is a direct, measured example of it.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">ROC-AUC</div>
            <div class="gloss-def">The probability that a randomly chosen success scores higher than a randomly
            chosen failure. 0.5 = no better than random guessing; 1.0 = perfect separation. Does not depend on
            any specific threshold.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">Precision vs. recall</div>
            <div class="gloss-def">Precision: of the campaigns flagged as "likely to succeed," what fraction
            actually did? Recall: of the campaigns that actually succeeded, what fraction did we flag? There's
            usually a trade-off between the two.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">F1 score</div>
            <div class="gloss-def">A single number combining precision and recall (their harmonic mean). Used
            instead of accuracy here because accuracy is misleading when one outcome (failure) is much more
            common than the other (success).</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">Base rate</div>
            <div class="gloss-def">How often the positive outcome happens by default &mdash; here, 17.0% of
            Kickstarter campaigns succeed. Any useful signal has to beat this baseline, not just look impressive
            in isolation.</div>
          </div>
        </div>
      </div>
      <div class="footnote">
        Source data: results/KickstarterCalibrated/evaluation/calibration_results.json (generated by
        calibrate_kickstarter_threshold.py). Report generated by generate_kickstarter_calibration_report.py.
        Random seed 42, 5-fold stratified cross-validation.
      </div>
    </div>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_html", default="results/KickstarterCalibrated/kickstarter_calibration_report.html")
    ap.add_argument("--out_pdf", default="results/KickstarterCalibrated/kickstarter_calibration_report.pdf")
    args = ap.parse_args()

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    data2 = json.loads(IMPROVE_PATH.read_text(encoding="utf-8"))
    data3 = json.loads(ROBUST_PATH.read_text(encoding="utf-8"))
    generated_at = datetime.now().strftime("%B %d, %Y  %H:%M")

    body = "".join([
        build_cover(data, data2, data3, generated_at),
        build_intro(),
        build_methodology(),
        build_discrimination(data),
        build_before_after(data),
        build_fold_stability(data),
        build_score_improvement(data2, data3),
        build_outcomes(data, data2, data3),
        build_findings(data3),
        build_glossary(),
    ])

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kickstarter GO-Threshold Calibration Study</title>
  <style>{CSS}</style>
</head>
<body>
  {body}
</body>
</html>"""

    html_path = ROOT / args.out_html
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(full_html, encoding="utf-8")
    print(f"[OK] HTML written -> {html_path}")

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
    print("[..] Generating PDF via Chrome headless ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size // 1024
        print(f"[OK] PDF written  -> {pdf_path}  ({size_kb} KB)")
    else:
        print("[FAIL] PDF generation failed.")
        print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
