#!/usr/bin/env python3
"""
Generate a detailed HTML + PDF report of the Kickstarter real-world validation
results: AI proposal scores vs. actual campaign funding outcomes.

Usage:
    python3 generate_kickstarter_report.py
    python3 generate_kickstarter_report.py --out_html report.html --out_pdf report.pdf
"""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "tools"))

from kickstarter_dataset import load_funded_pct  # noqa: E402
from src.backend.utils.score_extraction import extract_report_scores  # noqa: E402

REPORTS_DIR = ROOT / "src" / "data" / "reports"
SUMMARY_PATH = ROOT / "results" / "Kickstarter" / "evaluation" / "evaluation_summary.json"

DIM_LABELS = {
    "team": "Team",
    "objective": "Objective",
    "strategy": "Strategy",
    "advantages": "Advantages / Innovation",
    "feasibility": "Feasibility",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_rows():
    funded = load_funded_pct()
    rows = []
    for pid, pct in funded.items():
        p = REPORTS_DIR / f"{pid}__openai_final_report.md"
        if not p.exists():
            p = REPORTS_DIR / f"{pid}_final_report.md"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        s = extract_report_scores(text)
        city = "Toronto" if pid.startswith("kt") else "Montreal"
        rows.append({
            "pid": pid, "city": city, "funded_pct": pct,
            "overall": s["overall_ranking"], "verdict": s["verdict"],
            **{d: s[d] for d in DIM_LABELS},
        })
    return rows


def verdict_text_counts():
    counts = {"HOLD": 0, "NO-GO": 0, "GO": 0}
    for p in REPORTS_DIR.glob("k[tm]*__openai_final_report.md"):
        text = p.read_text(encoding="utf-8")
        for v in ("GO", "HOLD", "NO-GO"):
            if f"**Overall verdict:** {v}" in text:
                counts[v] += 1
                break
    return counts


# ── Small chart builders (CSS bar charts + inline SVG scatter) ────────────────

def sig_badge(holm_p):
    if holm_p is None:
        return '<span class="sig-ns">n/a</span>'
    if holm_p < 0.01:
        return '<span class="sig-strong">p &lt; 0.01</span>'
    if holm_p < 0.05:
        return '<span class="sig-strong">p &lt; 0.05</span>'
    if holm_p < 0.10:
        return '<span class="sig-mid">borderline</span>'
    return '<span class="sig-ns">not sig.</span>'


def rho_bar(rho, max_abs=0.6):
    pct = min(100, abs(rho) / max_abs * 100)
    color = "pos" if rho >= 0 else "neg"
    return (f'<div class="bar-bg"><div class="bar-fill {color}" style="width:{pct:.1f}%"></div></div>')


def build_overall_table(summary):
    rows_html = []
    for key, label in (("toronto", "Toronto"), ("montreal", "Montreal"), ("combined", "Combined")):
        g = summary[key]
        os_ = g["overall_spearman"]
        rows_html.append(f"""
        <tr class="{'row-total' if key == 'combined' else ''}">
          <td class="pid">{label}</td>
          <td>{os_['n']}</td>
          <td class="st-num">{os_['rho']:+.3f}</td>
          <td>{rho_bar(os_['rho'])}</td>
          <td>{sig_badge(os_.get('holm_p_value'))}</td>
        </tr>""")
    return f"""
    <table>
      <thead><tr><th>Group</th><th>n</th><th>Spearman &rho;</th><th>Strength</th><th>Significance</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>"""


def build_dimension_table(summary):
    combined = summary["combined"]["dimension_spearman"]
    order = sorted(DIM_LABELS.keys(), key=lambda d: -combined[d]["rho"])
    rows_html = []
    for d in order:
        c = combined[d]
        rows_html.append(f"""
        <tr>
          <td class="dim-name">{DIM_LABELS[d]}</td>
          <td class="st-num">{c['rho']:+.3f}</td>
          <td>{rho_bar(c['rho'])}</td>
          <td>{sig_badge(c.get('holm_p_value'))}</td>
        </tr>""")
    return f"""
    <table>
      <thead><tr><th>Dimension</th><th>Spearman &rho; (combined, n=141)</th><th>Strength</th><th>Significance</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>"""


def build_classification_table(summary):
    rows_html = []
    for key, label in (("toronto", "Toronto"), ("montreal", "Montreal"), ("combined", "Combined")):
        c = summary[key]["classification"]
        acc = f"{c['accuracy']*100:.1f}%" if c["accuracy"] is not None else "—"
        rec = f"{c['recall']*100:.1f}%" if c["recall"] is not None else "—"
        rows_html.append(f"""
        <tr class="{'row-total' if key == 'combined' else ''}">
          <td class="pid">{label}</td>
          <td>{acc}</td>
          <td>{rec}</td>
          <td>{c['tp']}</td>
          <td>{c['fp']}</td>
          <td>{c['fn']}</td>
          <td>{c['tn']}</td>
        </tr>""")
    return f"""
    <table>
      <thead><tr><th>Group</th><th>Accuracy</th><th>Recall (success)</th>
      <th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>"""


def build_scatter_svg(rows):
    W, H = 720, 360
    pad_l, pad_r, pad_t, pad_b = 56, 24, 24, 44
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b

    xs = [math.log10(r["funded_pct"] + 1) for r in rows]
    ys = [r["overall"] for r in rows]
    x_max = max(xs) * 1.05
    y_min, y_max = 0.15, 0.78

    def X(v):
        return pad_l + (v / x_max) * plot_w

    def Y(v):
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    # gridlines at 0%, 10%, 100%, 1000% funded
    grid_vals = [0, 10, 100, 1000]
    grid_lines = []
    for gv in grid_vals:
        gx = X(math.log10(gv + 1))
        grid_lines.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t+plot_h}" class="grid-line"/>')
        grid_lines.append(f'<text x="{gx:.1f}" y="{pad_t+plot_h+16}" class="axis-label" text-anchor="middle">{gv}%</text>')

    y_grid_vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for gv in y_grid_vals:
        gy = Y(gv)
        grid_lines.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l+plot_w}" y2="{gy:.1f}" class="grid-line"/>')
        grid_lines.append(f'<text x="{pad_l-8}" y="{gy+3:.1f}" class="axis-label" text-anchor="end">{gv:.2f}</text>')

    # success threshold (funded=100%) vertical reference
    x100 = X(math.log10(101))
    ref_v = f'<line x1="{x100:.1f}" y1="{pad_t}" x2="{x100:.1f}" y2="{pad_t+plot_h}" class="ref-line"/>'
    ref_v_label = f'<text x="{x100+4:.1f}" y="{pad_t+12}" class="ref-label">funded &ge; 100%</text>'

    # GO threshold: production verdict_rule() is score >= 0.725 OR confidence >= 0.811.
    # Confidence isn't plotted on this score-vs-funding% chart, but no Kickstarter proposal
    # reaches confidence >= 0.811 either (see calibrate_kickstarter_threshold.py), so the
    # score line alone correctly shows why zero GO verdicts fired.
    y_go = Y(0.725)
    ref_h = f'<line x1="{pad_l}" y1="{y_go:.1f}" x2="{pad_l+plot_w}" y2="{y_go:.1f}" class="ref-line"/>'
    ref_h_label = f'<text x="{pad_l+plot_w-4:.1f}" y="{y_go-6:.1f}" class="ref-label" text-anchor="end">GO threshold (score &ge; 0.725)</text>'

    points = []
    for r in rows:
        cx = X(math.log10(r["funded_pct"] + 1))
        cy = Y(r["overall"])
        cls = "pt-success" if r["funded_pct"] >= 100 else "pt-fail"
        title = f"{r['pid']} ({r['city']}): funded {r['funded_pct']:.0f}%, AI score {r['overall']:.2f}"
        points.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.2" class="{cls}"><title>{title}</title></circle>')

    axis_x_title = f'<text x="{pad_l+plot_w/2:.1f}" y="{H-4}" class="axis-title" text-anchor="middle">Actual Kickstarter funding (% of goal, log scale)</text>'
    axis_y_title = f'<text x="16" y="{pad_t+plot_h/2:.1f}" class="axis-title" text-anchor="middle" transform="rotate(-90 16 {pad_t+plot_h/2:.1f})">AI overall score (0&ndash;1)</text>'

    return f"""
    <svg viewBox="0 0 {W} {H}" class="scatter-svg" role="img" aria-label="Scatter plot of AI overall score versus actual Kickstarter funding percentage">
      {''.join(grid_lines)}
      {ref_v}{ref_v_label}
      {ref_h}{ref_h_label}
      {''.join(points)}
      {axis_x_title}
      {axis_y_title}
      <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" class="plot-border"/>
    </svg>
    <div class="scatter-legend">
      <span class="leg-item"><span class="leg-dot pt-success"></span>Actually funded (&ge; 100% of goal), n=24</span>
      <span class="leg-item"><span class="leg-dot pt-fail"></span>Not funded (&lt; 100% of goal), n=117</span>
    </div>"""


def build_highlight_table(rows):
    top = sorted(rows, key=lambda r: -r["overall"])[:5]
    bottom = sorted(rows, key=lambda r: r["overall"])[:5]

    def row_html(r):
        funded_ok = r["funded_pct"] >= 100
        badge = '<span class="tag-good">funded</span>' if funded_ok else '<span class="tag-bad">not funded</span>'
        return f"""
        <tr>
          <td class="pid">{r['pid']}</td>
          <td>{r['city']}</td>
          <td class="st-num">{r['overall']:.3f}</td>
          <td class="st-num">{r['funded_pct']:.0f}%</td>
          <td>{badge}</td>
          <td>{r['verdict']}</td>
        </tr>"""

    return f"""
    <div class="dim-meta">
      <div class="meta-card plus-card" style="flex:1">
        <div class="meta-label">Highest AI Scores</div>
        <table class="mini-table">
          <thead><tr><th>PID</th><th>City</th><th>AI score</th><th>Funded</th><th>Outcome</th><th>Verdict</th></tr></thead>
          <tbody>{''.join(row_html(r) for r in top)}</tbody>
        </table>
      </div>
      <div class="meta-card minus-card" style="flex:1">
        <div class="meta-label">Lowest AI Scores</div>
        <table class="mini-table">
          <thead><tr><th>PID</th><th>City</th><th>AI score</th><th>Funded</th><th>Outcome</th><th>Verdict</th></tr></thead>
          <tbody>{''.join(row_html(r) for r in bottom)}</tbody>
        </table>
      </div>
    </div>"""


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 11px;
    color: #0b0b0b;
    background: #f9f9f7;
    padding: 20px;
  }

  .cover {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    color: #fff;
    border-radius: 12px;
    padding: 48px 40px;
    margin-bottom: 32px;
  }
  .cover h1 { font-size: 26px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 8px; }
  .cover .subtitle { font-size: 13px; color: #a0aec0; margin-bottom: 24px; line-height: 1.6; }
  .cover .meta { display: flex; gap: 32px; flex-wrap: wrap; }
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
  .cover .stat-box.warn .sval { color: #fc8181; }
  .cover .stat-box .slbl { font-size: 9px; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

  .intro-box {
    background: #fff;
    border-left: 4px solid #2a78d6;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }
  .intro-box h2 { font-size: 13px; margin-bottom: 8px; color: #184f95; }
  .intro-box p { line-height: 1.7; color: #4a5568; margin-bottom: 10px; }
  .intro-box p:last-child { margin-bottom: 0; }

  .section {
    background: #fff;
    border-radius: 10px;
    padding: 28px;
    margin-bottom: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    page-break-inside: avoid;
  }
  .section h2 {
    font-size: 16px;
    font-weight: 700;
    color: #1a202c;
    padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 16px;
  }
  .section h3 { font-size: 12.5px; font-weight: 700; color: #2d3748; margin: 18px 0 8px; }
  .section p { line-height: 1.75; color: #4a5568; margin-bottom: 10px; }
  .section p:last-child { margin-bottom: 0; }
  .plain { background: #ebf8ff; border: 1px solid #bee3f8; border-radius: 8px; padding: 12px 16px; font-size: 10.5px; color: #2c5282; line-height: 1.7; margin: 10px 0 16px; }
  .plain b { color: #184f95; }

  table { border-collapse: collapse; width: 100%; margin-bottom: 6px; font-size: 10.5px; }
  thead { background: #edf2f7; }
  th, td { border: 1px solid #e2e8f0; padding: 6px 8px; text-align: center; }
  th { font-weight: 600; color: #2d3748; font-size: 10px; }
  tbody tr:nth-child(even) { background: #f7fafc; }
  .pid { font-weight: 600; text-align: left; padding-left: 10px; color: #2d3748; }
  .dim-name { text-align: left; padding-left: 10px; font-weight: 600; }
  .st-num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 12px; }
  .row-total { background: #ebf8ff !important; border-left: 3px solid #2a78d6; font-weight: 600; }

  .bar-bg { background:#e1e0d9; border-radius:4px; height:10px; width:150px; overflow:hidden; display:inline-block; vertical-align:middle; }
  .bar-fill { height:100%; border-radius:4px; }
  .bar-fill.pos { background: linear-gradient(90deg,#6da7ec,#184f95); }
  .bar-fill.neg { background: linear-gradient(90deg,#e87ba4,#9b2c2c); }

  .sig-strong { color: #006300; font-weight: 700; }
  .sig-mid { color: #b7791f; font-weight: 700; }
  .sig-ns { color: #898781; }

  .mini-table { font-size: 9.5px; margin-top: 6px; }
  .mini-table th, .mini-table td { padding: 4px 6px; }
  .meta-card { flex: 1; border-radius: 8px; padding: 14px 16px; }
  .plus-card { background: #f0fff4; border: 1px solid #9ae6b4; }
  .minus-card { background: #fff5f5; border: 1px solid #feb2b2; }
  .meta-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
  .plus-card .meta-label { color: #006300; }
  .minus-card .meta-label { color: #9b2c2c; }
  .dim-meta { display: flex; gap: 14px; margin: 12px 0; }
  .tag-good { background:#c6f6d5; color:#22543d; border-radius:4px; padding:1px 6px; font-size:9px; font-weight:700; }
  .tag-bad { background:#fed7d7; color:#742a2a; border-radius:4px; padding:1px 6px; font-size:9px; font-weight:700; }

  .scatter-svg { width: 100%; height: auto; margin-top: 10px; }
  .grid-line { stroke: #e1e0d9; stroke-width: 1; }
  .axis-label { font-size: 9px; fill: #898781; }
  .axis-title { font-size: 10px; fill: #52514e; font-weight: 600; }
  .plot-border { fill: none; stroke: #c3c2b7; stroke-width: 1; }
  .ref-line { stroke: #eda100; stroke-width: 1.2; stroke-dasharray: 4 3; }
  .ref-label { font-size: 8.5px; fill: #b7791f; font-weight: 600; }
  .pt-success { fill: #0ca30c; fill-opacity: 0.75; stroke: #006300; stroke-width: 0.6; }
  .pt-fail { fill: #d03b3b; fill-opacity: 0.55; stroke: #9b2c2c; stroke-width: 0.6; }
  .scatter-legend { display: flex; gap: 24px; justify-content: center; margin-top: 8px; font-size: 10px; color: #4a5568; }
  .leg-item { display: flex; align-items: center; gap: 6px; }
  .leg-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

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

  @media print {
    body { background: #fff; padding: 0; }
    .section { box-shadow: none; }
    .cover { border-radius: 0; }
  }
"""


# ── Section builders ────────────────────────────────────────────────────────

def build_cover(summary, generated_at, verdict_counts):
    combined = summary["combined"]
    return f"""
    <div class="cover">
      <h1>Kickstarter Real-World Validation Study</h1>
      <div class="subtitle">AI proposal-review pipeline scores vs. actual crowdfunding outcomes &mdash; a real-world,
      out-of-domain test of the same 9-stage evaluation system validated earlier on VC/biotech proposals.</div>
      <div class="meta">
        <div class="meta-item"><div class="label">Generated</div><div class="value">{generated_at}</div></div>
        <div class="meta-item"><div class="label">Dataset</div><div class="value">Kickstarter (Toronto + Montreal)</div></div>
        <div class="meta-item"><div class="label">Ground truth</div><div class="value">Actual campaign funded (%)</div></div>
        <div class="meta-item"><div class="label">Success rule</div><div class="value">Funded % &ge; 100 (all-or-nothing)</div></div>
      </div>
      <div class="stat-row">
        <div class="stat-box"><div class="sval">141</div><div class="slbl">Proposals scored</div></div>
        <div class="stat-box"><div class="sval">{combined['overall_spearman']['rho']:+.3f}</div><div class="slbl">Combined Spearman &rho;</div></div>
        <div class="stat-box"><div class="sval">{combined['classification']['accuracy']*100:.0f}%</div><div class="slbl">Verdict accuracy</div></div>
        <div class="stat-box warn"><div class="sval">{verdict_counts['GO']}</div><div class="slbl">GO verdicts issued</div></div>
      </div>
    </div>"""


def build_intro():
    return """
    <div class="intro-box">
      <h2>What is this report, in plain terms?</h2>
      <p>Earlier validation of this AI proposal-review pipeline compared its scores to opinions from human experts
      on 12 biotech/semiconductor investment proposals. That is a useful but "soft" test &mdash; human experts can
      disagree with each other, and there is no single objectively correct answer.</p>
      <p>This report is a harder, "hard-money" test. We ran the exact same pipeline, unmodified, on 141 real
      Kickstarter crowdfunding campaigns (104 from Toronto, 37 from Montreal) and compared its scores against
      what actually happened: whether the campaign was fully funded by real backers. Kickstarter is "all-or-nothing"
      &mdash; a campaign that reaches 99% of its goal collects $0 &mdash; so "funded % &ge; 100" is an unambiguous,
      objective pass/fail line with no room for interpretation.</p>
      <p>The question this answers: <b>does the AI's sense of "this looks fundable" line up with what real people
      actually funded with real money?</b></p>
    </div>"""


def build_methodology():
    return """
    <div class="section">
      <h2>1. Methodology</h2>
      <p>141 Kickstarter campaign PDFs were run through the unmodified 9-stage review pipeline (the same one used
      for the biotech/semiconductor validation), producing an AI overall score (0&ndash;1), five dimension scores
      (Team, Objective, Strategy, Advantages, Feasibility), and a GO / HOLD / NO-GO verdict for each proposal.</p>
      <div class="plain">
        <b>In simple terms:</b> the AI never saw the funding outcome. It only read the campaign pitch (like a
        prospective backer would) and produced a score. We then, separately, looked up what actually happened on
        Kickstarter and checked whether the two lined up.
      </div>
      <h3>Metrics used</h3>
      <p><b>Spearman rank correlation (&rho;):</b> do proposals the AI ranked higher also tend to be the ones that
      raised more money, relative to their goal? Ranges from &minus;1 (perfectly backwards) to +1 (perfectly
      aligned); 0 means no relationship. Used instead of a straight-line correlation because funding % is extremely
      skewed (most campaigns raise 0&ndash;150% of goal, a few raise over 1000%).</p>
      <p><b>Classification (GO vs. actual success):</b> treating the AI's verdict as a prediction ("GO" = predicted
      success) and checking it against whether the campaign actually hit 100% funding.</p>
      <p><b>Holm correction:</b> when testing many correlations at once (5 dimensions x 3 groups), some will look
      "significant" by chance alone. The Holm correction tightens the significance bar to account for that, so a
      result that survives it is safer to trust.</p>
    </div>"""


def build_overall_results(summary):
    return f"""
    <div class="section">
      <h2>2. Overall Result: Does the AI Score Predict Real Funding Success?</h2>
      <div class="plain">
        <b>Short answer:</b> yes, weakly-to-moderately, and the relationship is statistically real (not chance).
        Proposals the AI scored higher were, on average, more likely to have actually been funded &mdash; but the
        relationship is far from perfect.
      </div>
      {build_overall_table(summary)}
      <p style="margin-top:10px">Montreal shows a stronger relationship (&rho;=0.506) than Toronto (&rho;=0.314)
      despite having a third of the sample size &mdash; worth flagging as a city-level difference rather than
      treating the combined number as uniform across cities.</p>
    </div>"""


def build_dimension_results(summary):
    return f"""
    <div class="section">
      <h2>3. Which Scoring Dimension Predicts Real Success Best?</h2>
      {build_dimension_table(summary)}
      <div class="plain">
        <b>In simple terms:</b> <b>Feasibility</b> (can this actually be built/delivered) and <b>Strategy</b>
        (is the plan coherent) are the AI's dimensions that most closely track what real backers rewarded.
        <b>Team</b> and <b>Objective</b> show little to no relationship with real funding outcomes &mdash; the
        opposite pattern from the biotech dataset, where Team was one of the stronger predictors of what human
        experts liked.
      </div>
    </div>"""


def build_classification_results(summary, verdict_counts):
    return f"""
    <div class="section">
      <h2>4. Can the Pipeline's GO / HOLD / NO-GO Verdict Predict a Successful Campaign?</h2>
      {build_classification_table(summary)}
      <div class="plain">
        <b>The headline problem:</b> across all 141 campaigns, the pipeline issued
        <b>{verdict_counts['HOLD']} HOLD</b> and <b>{verdict_counts['NO-GO']} NO-GO</b> verdicts, and
        <b>{verdict_counts['GO']} GO</b> verdicts. It never once said "GO" &mdash; not even for the campaign that
        raised 151% of its goal. That means <b>recall on real successes is 0%</b>: as currently calibrated, the
        pipeline's decision layer cannot flag a winning Kickstarter campaign as such, no matter how well it actually
        performs. The 83% "accuracy" figure above is not meaningful on its own &mdash; it is inflated by the fact
        that 83% of Kickstarter campaigns in this dataset failed anyway, so "always predict failure" scores well by
        default.
      </div>
    </div>"""


def build_scatter_section(rows):
    return f"""
    <div class="section">
      <h2>5. AI Score vs. Actual Funding Outcome (Every Proposal)</h2>
      <p>Each dot is one Kickstarter campaign. The horizontal axis is the actual money raised (as a percent of
      the funding goal, log scale, so both small and huge outcomes are visible). The vertical axis is the AI's
      overall score. The dashed lines mark the two thresholds that matter: 100% funding (real-world success) and
      the pipeline's actual "GO" decision rule, <b>score &ge; 0.725</b> (the rule also allows confidence &ge; 0.811
      as an alternative trigger; no Kickstarter proposal reaches that either, so the score line alone explains the
      outcome here).</p>
      {build_scatter_svg(rows)}
      <div class="plain">
        <b>What to look for:</b> green (funded) dots sit slightly higher on average than red (not funded) dots
        &mdash; that is the positive correlation. But <b>no dot of either color gets close to the 0.725 GO line</b>
        &mdash; the highest score in the entire dataset (0.640) still falls nearly 0.09 short of it &mdash; so the
        top-right "confidently predicted success" quadrant is empty. The signal exists in the ranking; it just
        never gets strong enough, on the AI's own scoring scale, to trigger a GO.
      </div>
    </div>"""


def build_highlights(rows):
    return f"""
    <div class="section">
      <h2>6. Highest- and Lowest-Scored Proposals</h2>
      <p>The AI's own top pick, <b>kt49</b>, scored 0.640 &mdash; the highest of any Kickstarter proposal, but still
      well short of the 0.725 GO threshold &mdash; and was correctly verdicted HOLD; the campaign itself only raised
      4% of its goal (a real failure). Meanwhile <b>kt10</b> scored almost identically (0.639), was also correctly
      verdicted HOLD, yet was actually funded at 151%. Both examples illustrate the same underlying point: overall
      score has a real but noisy relationship to outcome, and near-identical scores can sit on either side of the
      real funding outcome &mdash; a ceiling on what proposal text alone can predict, not a bug in the verdict
      logic itself.</p>
      {build_highlight_table(rows)}
    </div>"""


def build_outcomes():
    return """
    <div class="section">
      <h2>7. Positive and Negative Outcomes</h2>
      <p>Framed for a research paper: what this experiment supports, and what it exposes as a limitation.</p>

      <div class="outcome-box outcome-good">
        <h4>Positive outcomes (supports the pipeline's validity)</h4>
        <ul>
          <li>The AI's ranking of proposals correlates with <b>real, independently-verifiable, financial outcomes</b>
          &mdash; not just human opinion &mdash; and the correlation (&rho;=0.366 combined, n=141) is statistically
          significant even after correcting for multiple comparisons (Holm p &lt; 0.001).</li>
          <li>The result replicates across <b>two independent cities</b> (Toronto &rho;=0.314, Montreal &rho;=0.506)
          scored with the same unmodified pipeline, which is evidence the signal is not an artifact of one dataset.</li>
          <li>Funded campaigns scored 0.573 on average vs. 0.499 for unfunded ones &mdash; a consistent, directionally
          correct gap that shows up even though it never crosses the pipeline's own GO threshold.</li>
          <li><b>Feasibility</b> and <b>Strategy</b> are meaningful, transferable predictors of real-world funding
          success, generalizing beyond the biotech domain the rubric was originally designed for.</li>
          <li>This is a genuine <b>out-of-domain generalization test</b>: the pipeline was built and tuned for
          VC/biotech proposals, never touched for Kickstarter, and still produced a significant signal on a
          completely different proposal genre (consumer/creative crowdfunding) and outcome type (crowd funding vs.
          expert judgment).</li>
        </ul>
      </div>

      <div class="outcome-box outcome-bad">
        <h4>Negative outcomes / limitations (should be reported honestly)</h4>
        <ul>
          <li>The pipeline's <b>decision layer (GO/HOLD/NO-GO) is miscalibrated for this dataset</b>: its GO rule
          (score &ge; 0.725 OR confidence &ge; 0.811, transferred unmodified from the 12-proposal biotech dataset
          it was tuned on) issued zero GO verdicts across 141 campaigns, including one that raised 151% of its goal
          &mdash; recall on real successes is 0%. The highest AI score in the whole dataset (0.640) still falls
          short of the 0.725 cutoff by a wide margin, so this isn't a borderline miscalibration &mdash; the
          threshold is structurally out of reach for this domain.</li>
          <li>The 83% classification "accuracy" is misleading in isolation: it mainly reflects the base rate
          (83% of campaigns failed anyway), not the model's discriminative power on the positive class.</li>
          <li><b>Team</b> and <b>Objective</b> dimension scores show no significant relationship with real funding
          outcomes (Team: Holm p=0.115, Objective: Holm p=0.302) &mdash; these dimensions may be measuring signals
          that matter to expert reviewers but not to crowd backers, or may not transfer well outside the biotech
          domain.</li>
          <li>Correlation strength (&rho;&asymp;0.37) is moderate, not strong &mdash; roughly 13% of the variance in
          funding rank is explained by AI score rank (&rho;&sup2;), leaving most of what determines a campaign's
          success unexplained by the pipeline.</li>
          <li>Toronto and Montreal show meaningfully different correlation strengths (0.314 vs. 0.506) on the same
          pipeline, suggesting some sensitivity to sample composition or city-specific campaign characteristics
          that has not been investigated.</li>
        </ul>
      </div>
    </div>"""


def build_findings():
    return """
    <div class="section">
      <h2>8. Interesting Findings Worth Highlighting in a Paper</h2>
      <div class="findings-list">
        <ul>
          <li><b>Cross-domain generalization without retuning:</b> a pipeline calibrated entirely on biotech/hardware
          VC proposals produces a statistically significant, positive correlation with real consumer-crowdfunding
          outcomes it was never tuned on &mdash; a natural "transfer" or "external validity" result.</li>
          <li><b>Dimension importance flips across domains:</b> Team dominates in the expert-judged biotech dataset,
          but is one of the weakest, non-significant predictors in the real-money Kickstarter dataset, where
          Feasibility and Strategy dominate instead. This is a clean, quotable contrast between "what experts value"
          and "what crowds actually fund."</li>
          <li><b>The score ceiling sits structurally below the GO threshold:</b> the single highest-scoring proposal
          in the entire 141-campaign set (0.640) still falls nearly 0.09 short of the 0.725 GO cutoff &mdash; not a
          borderline miss, but a domain-wide ceiling effect (see the earlier Team-dimension analysis for why).
          A nearly identically-scored proposal pair, <b>kt49</b> (0.640, correctly HOLD, only raised 4%) and
          <b>kt10</b> (0.639, correctly HOLD, actually funded at 151%), also shows that even where scores agree
          almost exactly, real outcomes can diverge completely &mdash; a clean illustration of the ceiling on what
          proposal text alone can predict, independent of the threshold question.</li>
          <li><b>A real-world validation without expert raters:</b> unlike the biotech dataset (limited to 12
          proposals because human expert time is the bottleneck), the Kickstarter ground truth is free, objective,
          and scales to any sample size &mdash; a methodological point worth making about how this kind of pipeline
          can be validated cheaply going forward.</li>
          <li><b>All-or-nothing funding as a clean binary label:</b> Kickstarter's own funding mechanic
          (0% payout below 100% of goal) removes the ambiguity that usually plagues "success" labels in this kind of
          study &mdash; there is no partial-credit judgment call being smuggled into the ground truth.</li>
        </ul>
      </div>
    </div>"""


def build_glossary():
    return """
    <div class="section">
      <h2>Glossary (Plain-Language)</h2>
      <div class="glossary">
        <div class="gloss-grid">
          <div class="gloss-item">
            <div class="gloss-term">Spearman correlation (&rho;)</div>
            <div class="gloss-def">Measures whether two rankings move together, ignoring exact values. +1 = perfectly
            aligned rankings, 0 = no relationship, &minus;1 = perfectly reversed.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">p-value</div>
            <div class="gloss-def">The probability of seeing a correlation this strong purely by chance if there were
            actually no relationship. Smaller = more confident the relationship is real.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">Holm correction</div>
            <div class="gloss-def">A stricter significance bar applied when running many tests at once, to avoid
            false positives from testing many things and reporting only the ones that happened to look significant.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">Recall (on successes)</div>
            <div class="gloss-def">Of all campaigns that actually succeeded, what fraction did the AI correctly flag
            as GO? 0% means it caught none of them.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">Precision</div>
            <div class="gloss-def">Of all campaigns the AI flagged as GO, what fraction actually succeeded? Undefined
            here because the AI never flagged any campaign as GO.</div>
          </div>
          <div class="gloss-item">
            <div class="gloss-term">All-or-nothing funding</div>
            <div class="gloss-def">Kickstarter's rule: campaigns that don't reach 100% of their goal collect $0.
            There is no partial success, which is what makes "funded % &ge; 100" an unambiguous label.</div>
          </div>
        </div>
      </div>
      <div class="footnote">
        Source data: results/Kickstarter/evaluation/evaluation_summary.json (generated 2026-07-15).
        Report generated by generate_kickstarter_report.py.
      </div>
    </div>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_html", default="results/Kickstarter/kickstarter_results_report.html")
    ap.add_argument("--out_pdf", default="results/Kickstarter/kickstarter_results_report.pdf")
    args = ap.parse_args()

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = load_rows()
    vcounts = verdict_text_counts()
    generated_at = datetime.now().strftime("%B %d, %Y  %H:%M")

    body = "".join([
        build_cover(summary, generated_at, vcounts),
        build_intro(),
        build_methodology(),
        build_overall_results(summary),
        build_dimension_results(summary),
        build_classification_results(summary, vcounts),
        build_scatter_section(rows),
        build_highlights(rows),
        build_outcomes(),
        build_findings(),
        build_glossary(),
    ])

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kickstarter Real-World Validation Study — Detailed Results</title>
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
