#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a single comprehensive, research-paper-ready HTML + PDF document
covering the entire project: full 9-stage pipeline architecture, every
dataset, every evaluation methodology, and every result produced across the
biotech/semiconductor human-expert study and the three-part Kickstarter
real-outcome study (correlation, threshold calibration, score improvement,
robustness checks).

Numeric values for Dataset1/2 are loaded live from
results/Dataset1/evaluation/evaluation_summary.json. Kickstarter numbers are
loaded live from the four JSON files under
results/KickstarterCalibrated/evaluation/ and results/Kickstarter/evaluation/.
Pipeline architecture, rubric text, and existing-paper numbers were verified
against source files in this session and are embedded as structured constants
below with file:line provenance in comments.

Usage:
    python3 generate_full_project_report.py
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent

DATASET1_PATH = ROOT / "results" / "Dataset1" / "evaluation" / "evaluation_summary.json"
DATASET1_XLSX_PATH = ROOT / "results" / "Dataset1" / "evaluation" / "evaluation_report.xlsx"
DATASET1_SWEEP_PATH = ROOT / "results" / "Dataset1" / "calibration_sweep.json"
KS_EVAL_PATH = ROOT / "results" / "Kickstarter" / "evaluation" / "evaluation_summary.json"
KS_CALIB_PATH = ROOT / "results" / "KickstarterCalibrated" / "evaluation" / "calibration_results.json"
KS_IMPROVE_PATH = ROOT / "results" / "KickstarterCalibrated" / "evaluation" / "score_improvement_results.json"
KS_ROBUST_PATH = ROOT / "results" / "KickstarterCalibrated" / "evaluation" / "score_robustness_check.json"


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Heiti SC', sans-serif; font-size: 11px; color: #0b0b0b; background: #f9f9f7; padding: 20px; }

  .cover { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%); color: #fff; border-radius: 12px; padding: 52px 44px; margin-bottom: 32px; }
  .cover h1 { font-size: 27px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 10px; line-height: 1.25; }
  .cover .subtitle { font-size: 13px; color: #a0aec0; margin-bottom: 24px; line-height: 1.6; max-width: 640px; }
  .cover .meta { display: flex; gap: 32px; flex-wrap: wrap; }
  .cover .meta-item .label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #718096; }
  .cover .meta-item .value { font-size: 13px; color: #e2e8f0; margin-top: 2px; }
  .cover .stat-row { display: flex; gap: 14px; margin-top: 28px; flex-wrap: wrap; }
  .cover .stat-box { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; padding: 12px 18px; text-align: center; min-width: 104px; }
  .cover .stat-box .sval { font-size: 19px; font-weight: 700; color: #68d391; }
  .cover .stat-box .slbl { font-size: 8.5px; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 2px; }

  .toc { background: #fff; border-radius: 10px; padding: 24px 28px; margin-bottom: 28px; box-shadow: 0 2px 8px rgba(0,0,0,.07); }
  .toc h2 { font-size: 14px; font-weight: 700; margin-bottom: 12px; color: #1a202c; }
  .toc-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 32px; }
  .toc-part { font-weight: 700; color: #184f95; margin-top: 8px; font-size: 10.5px; }
  .toc-item { font-size: 10px; color: #4a5568; padding-left: 10px; line-height: 1.9; }

  .part-divider { background: linear-gradient(135deg, #184f95 0%, #2a78d6 100%); color: #fff; border-radius: 10px; padding: 22px 28px; margin-bottom: 24px; page-break-inside: avoid; break-inside: avoid; page-break-before: always; break-before: page; }
  .part-divider h2 { font-size: 18px; font-weight: 700; }
  .part-divider p { font-size: 11px; color: #cbd5e0; margin-top: 6px; }

  .intro-box { background: #fff; border-left: 4px solid #2a78d6; border-radius: 8px; padding: 20px 24px; margin-bottom: 28px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
  .intro-box h2 { font-size: 13px; margin-bottom: 8px; color: #184f95; }
  .intro-box p { line-height: 1.7; color: #4a5568; margin-bottom: 10px; }
  .intro-box p:last-child { margin-bottom: 0; }

  .section { background: #fff; border-radius: 10px; padding: 26px 28px; margin-bottom: 22px; box-shadow: 0 2px 8px rgba(0,0,0,.07); page-break-inside: avoid; }
  .section h2 { font-size: 15px; font-weight: 700; color: #1a202c; padding-bottom: 9px; border-bottom: 2px solid #e2e8f0; margin-bottom: 14px; }
  .section h3 { font-size: 12px; font-weight: 700; color: #2d3748; margin: 16px 0 7px; page-break-after: avoid; break-after: avoid; }
  .section h4 { font-size: 10.5px; font-weight: 700; color: #4a5568; margin: 12px 0 5px; text-transform: uppercase; letter-spacing: 0.4px; page-break-after: avoid; break-after: avoid; }
  .section p { line-height: 1.7; color: #4a5568; margin-bottom: 9px; font-size: 10.5px; }
  .section p:last-child { margin-bottom: 0; }
  .section ul, .section ol { margin: 6px 0 12px 18px; line-height: 1.8; color: #4a5568; font-size: 10.5px; }
  .section li { margin-bottom: 4px; }
  .section code { background: #edf2f7; padding: 1px 5px; border-radius: 3px; font-size: 9.5px; color: #2d3748; }
  .cite { font-size: 8.5px; color: #a0aec0; font-family: monospace; }

  .plain { background: #ebf8ff; border: 1px solid #bee3f8; border-radius: 8px; padding: 12px 16px; font-size: 10.5px; color: #2c5282; line-height: 1.7; margin: 10px 0 16px; page-break-inside: avoid; }
  .plain b { color: #184f95; }
  .formula-box { background: #f7fafc; border: 1px solid #cbd5e0; border-radius: 8px; padding: 14px 18px; font-family: 'Courier New', monospace; font-size: 10px; color: #1a202c; margin: 10px 0 14px; line-height: 1.7; white-space: pre-wrap; page-break-inside: avoid; }

  table { border-collapse: collapse; width: 100%; margin-bottom: 6px; font-size: 9.8px; page-break-inside: avoid; }
  thead { background: #edf2f7; }
  th, td { border: 1px solid #e2e8f0; padding: 5px 7px; text-align: center; }
  th { font-weight: 600; color: #2d3748; font-size: 9.3px; }
  tbody tr:nth-child(even) { background: #f7fafc; }
  .pid { font-weight: 600; text-align: left; padding-left: 9px; color: #2d3748; }
  .dim-name { text-align: left; padding-left: 9px; font-weight: 600; }
  .st-num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 10px; }
  .row-total { background: #ebf8ff !important; border-left: 3px solid #2a78d6; font-weight: 600; }
  .row-old { color: #a0aec0; }
  .neg { color: #c53030; }
  .pos { color: #006300; }

  .stage-flow { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 16px; }
  .stage-box { flex: 1; min-width: 88px; background: #edf2f7; border: 1px solid #cbd5e0; border-radius: 6px; padding: 8px 6px; text-align: center; }
  .stage-box .snum { font-size: 15px; font-weight: 700; color: #184f95; }
  .stage-box .sname { font-size: 8px; color: #4a5568; margin-top: 2px; line-height: 1.3; }

  .rubric-band { border-radius: 6px; padding: 8px 12px; margin-bottom: 5px; font-size: 9.5px; line-height: 1.6; }
  .band-a { background: #f0fff4; border-left: 3px solid #0ca30c; }
  .band-b { background: #ebf8ff; border-left: 3px solid #2a78d6; }
  .band-c { background: #fffaf0; border-left: 3px solid #eda100; }
  .band-d { background: #fff5f5; border-left: 3px solid #d03b3b; }
  .band-label { font-weight: 700; }

  .outcome-box { border-radius: 8px; padding: 15px 18px; margin-bottom: 12px; }
  .outcome-good { background: #f0fff4; border-left: 4px solid #0ca30c; }
  .outcome-bad { background: #fff5f5; border-left: 4px solid #d03b3b; }
  .outcome-box h4 { font-size: 11px; font-weight: 700; margin-bottom: 6px; text-transform: none; letter-spacing: 0; }
  .outcome-good h4 { color: #006300; }
  .outcome-bad h4 { color: #9b2c2c; }
  .outcome-box ul { margin-left: 16px; }

  .findings-list { background: #fffaf0; border: 1px solid #f6ad55; border-left: 4px solid #ed8936; border-radius: 8px; padding: 15px 18px; margin-bottom: 14px; }
  .findings-list li b { color: #1a202c; }

  .glossary { margin-top: 8px; border: 1px solid #bee3f8; border-radius: 8px; padding: 12px 16px; background: #ebf8ff; }
  .gloss-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .gloss-item { background: #fff; border-radius: 6px; padding: 9px 11px; border: 1px solid #bee3f8; }
  .gloss-term { font-size: 10px; font-weight: 700; color: #184f95; margin-bottom: 3px; }
  .gloss-def { font-size: 9px; color: #4a5568; line-height: 1.55; }

  .footnote { font-size: 9px; color: #898781; font-style: italic; line-height: 1.6; margin-top: 8px; border-top: 1px solid #e2e8f0; padding-top: 7px; }

  .roc-svg { width: 340px; height: auto; display: block; margin: 10px auto; }
  .grid-line { stroke: #e1e0d9; stroke-width: 1; }
  .axis-label { font-size: 9px; fill: #898781; }
  .axis-title { font-size: 10px; fill: #52514e; font-weight: 600; }
  .plot-border { fill: none; stroke: #c3c2b7; stroke-width: 1; }
  .roc-diag { stroke: #c3c2b7; stroke-width: 1.4; stroke-dasharray: 4 3; }
  .roc-path.m-a { stroke: #898781; stroke-dasharray: 3 2; stroke-width: 2; }
  .roc-path.m-nb { stroke: #0ca30c; stroke-width: 3; fill: none; }
  .scatter-legend { display: flex; gap: 20px; justify-content: center; margin-top: 8px; font-size: 10px; color: #4a5568; flex-wrap: wrap; }
  .leg-item { display: flex; align-items: center; gap: 6px; }
  .leg-swatch { width: 16px; height: 3px; display: inline-block; border-radius: 2px; }
  .leg-swatch.m-a { background: #898781; }
  .leg-swatch.m-nb { background: #0ca30c; }

  .script-table td.pid { font-family: monospace; font-size: 9px; }

  @media print {
    body { background: #fff; padding: 0; }
    .section, .toc { box-shadow: none; }
    .cover { border-radius: 0; }
  }
"""


# ── Small helpers ────────────────────────────────────────────────────────────

def sec(num, title, body):
    return f"""<div class="section"><h2>{num}. {title}</h2>{body}</div>"""


def plain(html):
    return f'<div class="plain">{html}</div>'


def part_divider(title, sub):
    return f"""<div class="part-divider"><h2>{title}</h2><p>{sub}</p></div>"""


def stage_flow():
    stages = [
        ("0", "Prepare Proposal Text"), ("1", "Extract Facts by Chunk"),
        ("2", "Build Dimensions"), ("3", "Generate Questions"),
        ("4", "LLM Answering"), ("5", "Post-Processing"),
        ("6", "AI Expert Opinion"), ("7", "Generate Final Report"),
        ("8", "Generate LLM Scores"),
    ]
    boxes = "".join(f'<div class="stage-box"><div class="snum">{n}</div><div class="sname">{name}</div></div>' for n, name in stages)
    return f'<div class="stage-flow">{boxes}</div>'


def roc_points(rows, scores):
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    n_pos = sum(rows[i]["success"] for i in range(len(rows)))
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


def build_roc_svg(rows, series):
    W, H = 420, 420
    pad = 44
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
    paths, legend = [], []
    for label, cls, scores in series:
        pts = roc_points(rows, scores)
        path = "M " + " L ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        paths.append(f'<path d="{path}" class="roc-path {cls}"/>')
        legend.append(f'<span class="leg-item"><span class="leg-swatch {cls}"></span>{label}</span>')
    return f"""
    <svg viewBox="0 0 {W} {H}" class="roc-svg">
      {''.join(ticks)}{diag}{''.join(paths)}
      <text x="{pad+plot/2:.1f}" y="{H-4}" class="axis-title" text-anchor="middle">False positive rate</text>
      <text x="14" y="{pad+plot/2:.1f}" class="axis-title" text-anchor="middle" transform="rotate(-90 14 {pad+plot/2:.1f})">True positive rate</text>
      <rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" class="plot-border"/>
    </svg>
    <div class="scatter-legend">{''.join(legend)}</div>"""


# ── Part I: Pipeline sections ───────────────────────────────────────────────

def build_cover(generated_at):
    return f"""
    <div class="cover">
      <h1>The YangtzeDelta Proposal Analyser: Full System Architecture,<br>Datasets, and Evaluation Results</h1>
      <div class="subtitle">A complete technical reference covering the 9-stage LLM-assisted venture-screening pipeline
      end to end (document upload through final report), every dataset it has been evaluated against, every
      evaluation methodology used, and the complete numeric results of each study — compiled as source material
      for academic writing.</div>
      <div class="meta">
        <div class="meta-item"><div class="label">Generated</div><div class="value">{generated_at}</div></div>
        <div class="meta-item"><div class="label">Pipeline</div><div class="value">9-stage LLM proposal review system</div></div>
        <div class="meta-item"><div class="label">Datasets</div><div class="value">Dataset1/2 (biotech/semiconductor) + Kickstarter</div></div>
      </div>
      <div class="stat-row">
        <div class="stat-box"><div class="sval">9</div><div class="slbl">Pipeline stages</div></div>
        <div class="stat-box"><div class="sval">12</div><div class="slbl">Expert-scored proposals</div></div>
        <div class="stat-box"><div class="sval">141</div><div class="slbl">Kickstarter proposals</div></div>
        <div class="stat-box"><div class="sval">5</div><div class="slbl">Studies conducted</div></div>
        <div class="stat-box"><div class="sval">153</div><div class="slbl">Total proposals scored</div></div>
      </div>
    </div>"""


def build_toc():
    return """
    <div class="toc">
      <h2>Contents</h2>
      <div class="toc-cols">
        <div>
          <div class="toc-part">Part I &mdash; System Architecture</div>
          <div class="toc-item">1. Design Philosophy</div>
          <div class="toc-item">2. Formal Task Definition &amp; Validity Framework</div>
          <div class="toc-item">3. The Nine-Stage Pipeline: Overview</div>
          <div class="toc-item">4. Stage 0 &mdash; Document Ingestion &amp; OCR</div>
          <div class="toc-item">5. Stage 1 &mdash; Atomic Fact Extraction</div>
          <div class="toc-item">6. Stage 2 &mdash; Dimension Construction</div>
          <div class="toc-item">7. Stage 3 &mdash; Question Generation</div>
          <div class="toc-item">8. Stage 4 &mdash; LLM Answering</div>
          <div class="toc-item">9. Stage 5 &mdash; Post-Processing &amp; Scoring</div>
          <div class="toc-item">10. Stage 6 &mdash; AI Expert Opinion &amp; Rubric</div>
          <div class="toc-item">11. Stage 7 &mdash; Final Report Generation</div>
          <div class="toc-item">12. Stage 8 &mdash; LLM Rubric Scoring</div>
          <div class="toc-item">13. The Verdict Rule &amp; Score Formulas</div>
          <div class="toc-item">14. Operational Footprint</div>
          <div class="toc-item">15. Related Work &amp; Positioning</div>
        </div>
        <div>
          <div class="toc-part">Part II &mdash; Datasets</div>
          <div class="toc-item">16. Dataset 1 &amp; 2 (Biotech/Semiconductor)</div>
          <div class="toc-item">17. The Kickstarter Dataset</div>
          <div class="toc-part">Part III &mdash; Evaluation Methodology</div>
          <div class="toc-item">18. Human-Expert Agreement Methodology</div>
          <div class="toc-item">19. Real-World Outcome Methodology</div>
          <div class="toc-item">20. Threshold Calibration Methodology</div>
          <div class="toc-item">21. Score Improvement Methodology</div>
          <div class="toc-item">22. Robustness Check Methodology</div>
          <div class="toc-item">23. Groundedness &amp; Content-Sensitivity Audits</div>
          <div class="toc-part">Part IV &mdash; Full Results</div>
          <div class="toc-item">24. Human-Expert Study Results</div>
          <div class="toc-item">25. Kickstarter Correlation Results</div>
          <div class="toc-item">26. Kickstarter Highlight Examples</div>
          <div class="toc-item">27. Threshold Calibration Results</div>
          <div class="toc-item">28. Score Improvement Results</div>
          <div class="toc-item">29. Robustness Check Results</div>
          <div class="toc-item">30. Outcome-Free Audit Coverage Matrix</div>
          <div class="toc-part">Part V &mdash; Synthesis</div>
          <div class="toc-item">31. Cross-Dataset Comparison</div>
          <div class="toc-item">32. Key Findings for the Paper</div>
          <div class="toc-item">33. Limitations</div>
          <div class="toc-item">Appendix A &mdash; Glossary</div>
          <div class="toc-item">Appendix B &mdash; Script Inventory</div>
        </div>
      </div>
    </div>"""


def s1_philosophy():
    body = """
    <p>The system is framed not as a venture-success predictor but as a <b>measurement instrument</b>: it produces
    evidence-grounded, provenance-linked scores intended to support (not replace) human reviewer judgment. The
    project's own research paper states this explicitly: the central research question is <i>"Can an
    evidence-grounded LLM evaluation system serve as a credible outcome-free measurement instrument for venture
    proposal review?"</i> &mdash; deliberately narrower than "can an LLM predict which ventures will succeed."</p>
    <p><b>Provenance-preserving design:</b> rather than asking an LLM to read a proposal and output a score in one
    shot, the pipeline decomposes the task into nine auditable stages, each storing an intermediate artifact on
    disk (raw facts, dimension summaries, generated questions, candidate answers, selected answers, expert
    commentary, rubric scores) rather than passing only free-form text between stages. This means every score can
    be traced back to the specific extracted facts and generated evidence that produced it &mdash; a design
    explicitly modeled on FActScore's atomic-fact decomposition and RAGAS/ARES's separation of context quality from
    answer faithfulness.</p>
    <p><b>Two validation philosophies used in this project:</b> (1) <i>Outcome-free auditing</i> &mdash; comparing
    AI scores against human expert judgments on the same 12 proposals, used because real venture outcomes
    (acquisition, failure, sustained revenue) take years to materialize and are confounded by post-investment
    events. (2) <i>Outcome-based validation</i> &mdash; comparing AI scores against real, already-known outcomes
    (Kickstarter campaign funding results), which this project added specifically because it provides an objective,
    scalable ground truth the outcome-free approach cannot.</p>
    """
    return sec(1, "Design Philosophy", body)


def s2_task_definition():
    body = """
    <p>The project's research paper formalizes the screening task with explicit notation, useful for framing any
    academic writeup precisely rather than describing the system only in prose.</p>
    <div class="formula-box">D_i   = proposal i (pages/slides + extracted evidence)
H_i   = expert assessment of proposal i

Five ordinal dimensions, each normalized to [0,1]:
    Team, Objectives, Strategy, Advantages, Feasibility

System output for proposal i:
    H&#770;_i = { h&#770;_team, h&#770;_objective, h&#770;_strategy, h&#770;_advantages, h&#770;_feasibility,
             r&#770;_i,  c&#770;_i,  y&#770;_i }

    where  r&#770;_i &isin; [0,1]        overall screening score
           c&#770;_i &isin; [0,1]        confidence
           y&#770;_i &isin; {GO, NO-GO}  binary recommendation</div>
    <h3>Four score-validity targets</h3>
    <p>Rather than asking one blanket question ("is the AI accurate?"), the evaluation framework decomposes
    validity into four independently testable targets &mdash; each with its own dedicated metric and its own
    Part-IV results section in this document:</p>
    <table>
      <thead><tr><th>Validity target</th><th>Question it answers</th><th>Primary metric(s)</th><th>Where addressed</th></tr></thead>
      <tbody>
        <tr><td class="pid">Ordinal validity</td><td style="text-align:left">Does the system rank proposals the way experts would?</td><td style="text-align:left">Spearman &rho;, pairwise concordance</td><td>Section 24</td></tr>
        <tr><td class="pid">Scale validity</td><td style="text-align:left">Are absolute score values unbiased and correctly spread, not compressed or offset?</td><td style="text-align:left">ICC, mean-bias paired test</td><td>Section 24</td></tr>
        <tr><td class="pid">Evidence validity</td><td style="text-align:left">Are the claims underlying a score actually supported by the source proposal?</td><td style="text-align:left">Strict/lenient groundedness</td><td>Section 24</td></tr>
        <tr><td class="pid">Responsiveness validity</td><td style="text-align:left">Do scores move in the expected direction when evidence is added or removed?</td><td style="text-align:left">Prompt-coupled PLUS/MINUS pass rate</td><td>Section 24</td></tr>
      </tbody>
    </table>
    <p>This four-way decomposition is itself a reusable methodological contribution independent of this specific
    pipeline's numeric results &mdash; it gives a vocabulary for saying precisely <i>which kind</i> of validity a
    given metric speaks to, rather than treating "is the AI good" as a single monolithic question.</p>
    <p class="cite">Source: research_paper.pdf Section 3 (Task Definition).</p>
    """
    return sec(2, "Formal Task Definition &amp; Validity Framework", body)


def s3_stage_overview():
    body = f"""
    <p>The pipeline runs <b>nine numbered stages (0&ndash;8)</b>, orchestrated by
    <code>src/tools/run_multi_provider_lifecycle.py</code>, whose <code>STAGES</code> list (lines 41-51) is the
    canonical definition:</p>
    {stage_flow()}
    <table>
      <thead><tr><th>#</th><th>Script</th><th>Purpose</th></tr></thead>
      <tbody>
        <tr><td>0</td><td class="pid">prepare_proposal_text</td><td style="text-align:left">Extract text from the uploaded document (PDF/DOCX/PPTX/TXT/MD), with OCR fallback and optional vision-model description of charts/figures</td></tr>
        <tr><td>1</td><td class="pid">extract_facts_by_chunk</td><td style="text-align:left">Extract atomic, source-linked facts from chunked proposal text</td></tr>
        <tr><td>2</td><td class="pid">build_dimensions_from_facts</td><td style="text-align:left">Bucket facts into the five review dimensions and synthesize per-dimension evidence packets</td></tr>
        <tr><td>3</td><td class="pid">generate_questions</td><td style="text-align:left">Generate dimension-anchored evaluation questions linked to specific facts/risks</td></tr>
        <tr><td>4</td><td class="pid">llm_answering</td><td style="text-align:left">Answer every generated question, per provider, using only proposal facts</td></tr>
        <tr><td>5</td><td class="pid">post_processing</td><td style="text-align:left">Score and select the best candidate answer per question; compute per-dimension and overall QA scores</td></tr>
        <tr><td>6</td><td class="pid">ai_expert_opinion</td><td style="text-align:left">Write expert-style commentary and compute the investment-rubric scores (the core scoring stage)</td></tr>
        <tr><td>7</td><td class="pid">generate_final_report</td><td style="text-align:left">Compile executive summary, expert review, and evidence into one Markdown report</td></tr>
        <tr><td>8</td><td class="pid">generate_llm_scores</td><td style="text-align:left">Re-score the compiled report directly on the human 1&ndash;5 / 1&ndash;20 rubric scale, for evaluation comparability</td></tr>
      </tbody>
    </table>
    <p class="cite">Source: run_multi_provider_lifecycle.py:41-51; README.md:140-152 (stage table cross-verified, identical).</p>
    <div class="plain"><b>Two execution modes exist</b> for stages 6 and 8, selected by the
    <code>USE_PROVIDER_LIFECYCLE</code> environment variable (default: true). In the default mode,
    <code>ai_expert_opinion.py</code> and <code>generate_llm_scores.py</code> run once per configured LLM
    provider, each in its own independent thread, producing separate <code>&lt;pid&gt;__openai</code>,
    <code>&lt;pid&gt;__gemini</code>, etc. artifact sets for inter-rater comparison. In the legacy mode, a fan-out
    wrapper (<code>generate_multi_provider_expert_opinions.py</code> / <code>generate_multi_provider_scores.py</code>)
    runs all providers under one shared proposal id and copies the preferred successful provider's output back to
    the canonical path. Both modes produce compatible output shapes; the results in this document all use the
    OpenAI provider (<code>gpt-4o-mini</code>).</div>
    """
    return sec(3, "The Nine-Stage Pipeline: Overview", body)


def s3_stage0():
    body = """
    <p><b>Script:</b> <code>src/tools/prepare_proposal_text.py</code> (667 lines). <b>Input:</b> the uploaded
    proposal file (PDF, DOCX, PPTX, TXT, or Markdown). <b>Output:</b>
    <code>src/data/prepared/&lt;pid&gt;/full_text.txt</code>, <code>pages.json</code> (per-page text, source tag,
    global character offsets), and <code>stage0_audit.json</code> (per-page character counts and low-text page
    flags).</p>
    <h3>PDF extraction and OCR fallback</h3>
    <p>Text is first extracted natively via <code>pdfplumber</code>. If a page's extracted text falls below
    <code>MIN_TEXT_CHARS_PER_PAGE = 30</code> characters (a proxy for "this page is a scanned image, not real
    text"), the page is rendered to an image and passed through OCR via <code>pdf2image</code> +
    <code>pytesseract</code>. The OCR language is set to <b><code>chi_sim+eng</code></b> by default (mixed
    Simplified-Chinese and English), reflecting that the proposal corpus is bilingual. When both native and OCR
    text are short, a helper keeps whichever is longer to avoid discarding useful labels.</p>
    <h3>Vision-model pass for charts, tables, and figures</h3>
    <p>Any page containing embedded images is optionally (default: enabled) rendered at 150 dpi and sent to a
    vision-capable LLM with a prompt instructing it to describe "ALL visual content present: charts, graphs,
    tables, figures, diagrams, flowcharts," including specific numbers, axis labels, and trends. The resulting
    description is appended into the page's text as a tagged <code>[VISUAL CONTENT: ...]</code> block, so
    downstream stages can reason about a chart's content even though they only ever see text. Vision calls
    auto-disable after 3 consecutive failures to avoid wasting the rest of a run retrying a broken call. DOCX and
    PPTX documents get analogous treatment: paragraphs and tables are extracted as text; inline pictures go through
    the same vision pipeline.</p>
    <p class="cite">Source: prepare_proposal_text.py:48,65,68,70-72,97-113,222-392,395-529,578-622.</p>
    """
    return sec(4, "Stage 0 &mdash; Document Ingestion &amp; OCR", body)


def s4_stage1():
    body = """
    <p><b>Script:</b> <code>src/tools/extract_facts_by_chunk.py</code> (1074 lines). <b>Input:</b> Stage 0's
    <code>full_text.txt</code>. <b>Output:</b> <code>src/data/extracted/&lt;pid&gt;/raw_facts.jsonl</code> and
    <code>stage1_audit.json</code>.</p>
    <p>The proposal text is split into overlapping chunks, and the LLM is asked to extract <b>atomic facts</b> from
    each chunk &mdash; single, source-checkable propositions rather than interpretive statements. Each fact is
    tagged with: (a) a subset of the five review dimensions it's relevant to (<code>team, objectives, strategy,
    innovation, feasibility</code>), (b) a <code>type</code> from a 19-value enumeration (<code>team_member,
    milestone, tech_route, ip_asset, risk, mitigation,</code> etc.), and (c) source metadata (chunk index,
    character range) linking it back to the exact source text. This atomic decomposition is the mechanism that
    later lets the pipeline separate "the proposal reports three patents" (a checkable claim) from "the
    intellectual-property position is defensible" (an evaluator judgment) &mdash; the same distinction the
    project's paper models on FActScore. Chunks yielding suspiciously few facts (below a low-fact warning
    threshold) are retried, and missing dimension labels are inferred from keywords when the LLM under-labels a
    fact.</p>
    <p class="cite">Source: extract_facts_by_chunk.py:1-24,73,79-100.</p>
    """
    return sec(5, "Stage 1 &mdash; Atomic Fact Extraction", body)


def s5_stage2():
    body = """
    <p><b>Script:</b> <code>src/tools/build_dimensions_from_facts.py</code> (746 lines). <b>Input:</b> Stage 1's
    <code>raw_facts.jsonl</code>. <b>Output:</b> <code>dimensions_v2.json</code>, <code>dimension_facts.json</code>,
    and a copy at <code>src/data/parsed/parsed_dimensions.clean.llm.json</code>.</p>
    <p>Stage-1 facts are bucketed into the five review dimensions, then the LLM is called <b>once per dimension</b>
    with a strict grounding prompt that explicitly forbids inventing organizations, numbers, or technologies, and
    forbids speculative language ("likely," "probably") unless it is present in the source facts. This produces a
    <code>summary</code>, <code>key_points</code>, <code>risks</code>, and <code>mitigations</code> per dimension
    &mdash; the evidence packet that all downstream scoring is ultimately based on. <code>dimensions_v2.json</code>
    is the file the Stage 6 investment-scoring call reads directly (bypassing the QA-answer pipeline), so its
    quality has an outsized effect on the final rubric scores.</p>
    <p class="cite">Source: build_dimensions_from_facts.py:1-19,141-200.</p>
    """
    return sec(6, "Stage 2 &mdash; Dimension Construction", body)


def s6_stage3():
    body = """
    <p><b>Script:</b> <code>src/tools/generate_questions.py</code> (1461 lines). <b>Input:</b>
    <code>dimensions_v2.json</code>. <b>Output:</b> <code>generated_questions.json</code> (simplified list) and
    <code>generated_questions_detail.json</code> (full metadata: <code>qid</code>, <code>aspect</code>,
    <code>answer_type</code>, <code>links_to</code>).</p>
    <p>Questions are generated per dimension and must explicitly link back to specific <code>key_points</code>,
    <code>risks</code>, or <code>mitigations</code> via index references (<code>links_to</code>) &mdash; another
    provenance-preservation mechanism, ensuring every question has a traceable reason for existing rather than
    being a generic template question.</p>
    <p class="cite">Source: generate_questions.py:32-36.</p>
    """
    return sec(7, "Stage 3 &mdash; Question Generation", body)


def s7_stage4():
    body = """
    <p><b>Script:</b> <code>src/tools/llm_answering.py</code> (2274 lines, typically the slowest stage).
    <b>Input:</b> generated questions + dimension evidence. <b>Output:</b> per-provider raw answer files
    (<code>chatgpt_raw.json</code>, <code>gemini_raw.json</code>, <code>deepseek_raw.json</code>) merged into
    <code>all_refined_items.json</code>.</p>
    <p>Every generated question is answered using <b>only proposal facts</b>, plus general domain knowledge for
    explanatory context &mdash; explicitly forbidden from inventing new experiments, numbers, organizations,
    registrations, or evidence, and with no web search access. All configured providers run concurrently; their
    candidate answers are merged for the selection step in Stage 5. This structured-answer schema (separating
    direct response, factual claims, evidence pointers, confidence, and caveats) mirrors RAGAS/ARES's separation of
    context quality from answer faithfulness.</p>
    <p class="cite">Source: llm_answering.py:1-19.</p>
    """
    return sec(8, "Stage 4 &mdash; LLM Answering", body)


def s8_stage5():
    body = """
    <p><b>Script:</b> <code>src/tools/post_processing.py</code> (1727 lines). <b>Input:</b> Stage 4's merged
    candidate answers. <b>Output:</b> <code>metrics.json</code>, <code>selected_by_question.json</code>,
    <code>final_payload.json</code>, <code>report.md</code>, <code>drops_debug.json</code>.</p>
    <p>This is where per-answer quality scoring and dimension/overall QA-score aggregation happen &mdash; the
    numbers that feed into (but are ultimately dominated by) Stage 6's investment-rubric score.</p>
    <h3>Per-answer scoring formula</h3>
    <p>Each candidate answer receives sub-scores for length norm (<code>s_len</code>), claim count norm
    (<code>s_clm</code>), evidence count/authority/coverage (<code>s_evc</code>/<code>s_eva</code>/<code>s_evg</code>),
    structure (<code>s_str</code>), question/claim/evidence alignment (<code>s_aln</code>), and calibrated
    confidence (<code>s_cal</code>), combined as a weighted sum:</p>
    <div class="formula-box">fields_part = 0.16&middot;length + 0.20&middot;claims + 0.10&middot;evidence_count
+ 0.02&middot;evidence_authority + 0.02&middot;evidence_coverage
+ 0.17&middot;structure + 0.28&middot;alignment + 0.05&middot;calibrated_confidence</div>
    <p>This is then blended with a peer-consistency score (Jaccard similarity vs. other candidate answers to the
    same question, weight 0.10):</p>
    <div class="formula-box">total = 0.90 &times; fields_part + 0.10 &times; consistency_score</div>
    <p>Penalties are then subtracted: contradiction &minus;0.05, overclaim &minus;0.03, understructure &minus;0.03,
    redline_residual &minus;0.06, dimension_drift &minus;0.04 (clipped to [0,1]). This <code>total</code> is used to
    select the best candidate answer per question; each dimension's <code>avg</code> QA score is the mean of its
    selected answers' totals.</p>
    <h3>Dimension weights (used at both Stage 5 and Stage 6)</h3>
    <div class="formula-box">team = 1.00   objectives = 1.00   strategy = 1.00
innovation = 1.10   feasibility = 1.20</div>
    <h3>Stage 5's overall_score (QA-based, pre-Stage-6-blend)</h3>
    <div class="formula-box">overall_score = &Sigma;(dimension_avg &times; dimension_weight) / &Sigma;(dimension_weight)</div>
    <p><code>overall_confidence</code> is computed separately as a blend of mean per-answer confidence (weight
    0.5), mean pairwise Jaccard cross-provider agreement (weight 0.3), and one minus the mean pairwise
    contradiction rate (weight 0.2).</p>
    <p class="cite">Source: post_processing.py:195-211,700-768,1275-1301.</p>
    """
    return sec(9, "Stage 5 &mdash; Post-Processing &amp; QA Scoring", body)


def s9_stage6():
    body = """
    <p><b>Script:</b> <code>src/tools/ai_expert_opinion.py</code> (1239 lines, internally versioned "Stage 6 - AI
    Expert Opinion v4.1"). This is the core rubric-scoring stage and the single most important piece of the
    pipeline for understanding how final scores are produced. It makes <b>two separate LLM calls</b>.</p>

    <h3>Call A &mdash; Dimension commentary (qualitative, non-scoring)</h3>
    <p>Produces the expert-style narrative (strengths, concerns, recommendations) at <code>temperature=0.25</code>
    (non-zero &mdash; the source of the pipeline's known run-to-run non-determinism). This call is explicitly
    forbidden from outputting "question IDs, numeric scores, percentages, or internal metric names," so that its
    qualitative narrative cannot leak score information into itself. On failure it falls back to a deterministic
    local rule-based generator.</p>

    <h3>Call B &mdash; Investment-grade scoring (<code>INVEST_SCORE_SYSTEM</code>)</h3>
    <p>A fully separate call at <code>temperature=0.0, seed=42</code> (i.e., deterministic), reading
    <code>dimensions_v2.json</code> facts directly &mdash; bypassing the QA-answer pipeline entirely, to avoid
    document-quality/verbosity bias contaminating the investment judgment. This call contains the full scoring
    rubric, reproduced in complete detail below.</p>

    <div class="plain">The rubric's own preamble instructs the model: <i>"Score each dimension 0.0&ndash;1.0 based
    ONLY on the proposal facts provided. Be discriminating: most proposals should land in 0.40&ndash;0.70. Scores
    above 0.80 require strong, specific evidence. Do NOT be generous... Resist the urge to cluster scores near
    0.65-0.75."</i> This single instruction is the reason overall scores across every dataset in this project
    cluster in the 0.4&ndash;0.7 range rather than approaching 1.0 &mdash; it is a deliberate design choice in the
    rubric, not a limitation of the scoring model.</div>

    <h4>Team</h4>
    <div class="rubric-band band-a"><span class="band-label">T-A (0.83&ndash;0.95):</span> ANY of three qualifying sets &mdash; (1) academic research leader: CAS/national-academy/MIT-Harvard-Stanford-caliber PI + 50+ publications or major patents; (2) deep industry domain leader: 25+ years expertise + demonstrable commercial success at scale; (3) world-class operational excellence: DPPM&lt;10, &gt;1B units shipped, &gt;20 tier-1 customers, cross-functional leadership.</div>
    <div class="rubric-band band-b"><span class="band-label">T-B (0.65&ndash;0.80):</span> major-university professor (985/211 tier), OR senior executive (10+ yrs CXO/VP/Director), OR founder with a prior exit, OR company with 15+ years domain history.</div>
    <div class="rubric-band band-c"><span class="band-label">T-C (0.45&ndash;0.62):</span> credible practitioners without exceptional-performance proof &mdash; e.g. a clinical expert with a limited publication record, or an industry team without a large-scale execution track record.</div>
    <div class="rubric-band band-d"><span class="band-label">T-D (0.20&ndash;0.42):</span> vague or unverifiable credentials, or AI/software applied to a domain without deep domain expertise.</div>

    <h4>Innovation (mapped to "Advantages" at Stage 8 / in the human rubric)</h4>
    <div class="plain"><b>Two preliminary rules apply before categorization.</b> RULE 1 &mdash; the "AI-powered
    auto-cap": if the primary commercialized product is an AI <i>software system</i> (clinical decision support,
    hospital management, industrial process optimization &mdash; i.e. "AI IS the product"), the dimension is
    <b>automatically capped at I-C (0.42&ndash;0.56)</b> regardless of engineering sophistication. RULE 2 &mdash; the
    exception: if AI/ML is merely a computational <i>design tool</i> used to engineer a physical product (an enzyme,
    protein, drug molecule, material, or piece of hardware) and that physical output is what's being commercialized,
    the I-C cap does not apply &mdash; the physical/biological/chemical innovation is scored on its own merits.</div>
    <div class="rubric-band band-a"><span class="band-label">I-A (0.83&ndash;0.95):</span> requires ALL three: (1) a genuinely new mechanistic class (the "HOW," not just the "WHERE"), (2) an explicit IP claim (filed/granted patents or documented independent IP rights), (3) quantitative proof (specific numeric performance data). Missing any one disqualifies I-A.</div>
    <div class="rubric-band band-b"><span class="band-label">I-B (0.63&ndash;0.80):</span> novel but missing one I-A condition.</div>
    <div class="rubric-band band-c"><span class="band-label">I-C (0.40&ndash;0.60):</span> application-layer innovation, including most AI proposals. Explicit hard cap: <i>"Any proposal described as 'AI-powered', 'intelligent platform', 'data-driven', 'multimodal data integration', or 'machine learning for [clinical/industrial task]' &rarr; score 0.42&ndash;0.56, NO EXCEPTIONS."</i></div>
    <div class="rubric-band band-d"><span class="band-label">I-D (0.20&ndash;0.38):</span> incremental or commodity, no novel mechanism.</div>

    <h4>Objectives (three bands only, no D)</h4>
    <div class="rubric-band band-a"><span class="band-label">O-A (0.78&ndash;0.92):</span> quantified market gap, named buyer/payer pathway, documented unmet need with regulatory/clinical evidence.</div>
    <div class="rubric-band band-b"><span class="band-label">O-B (0.58&ndash;0.75):</span> clear problem + market-size estimate, but vague/aspirational buyer pathway.</div>
    <div class="rubric-band band-c"><span class="band-label">O-C (0.38&ndash;0.55):</span> real problem but undefined/speculative market size, no buyer evidence.</div>

    <h4>Strategy (three bands; scored on soundness/defensibility, not documentation completeness)</h4>
    <div class="rubric-band band-a"><span class="band-label">S-A (0.78&ndash;0.92):</span> strong on ALL of: defensibility (moat &mdash; first-mover IP, in-progress regulatory approval, exclusive channel, demonstrated network effects), execution realism (strategy matches team capability), and at least one market-validation signal (named production customer, filed regulatory submission, or existing paying customers).</div>
    <div class="rubric-band band-b"><span class="band-label">S-B (0.58&ndash;0.75):</span> coherent strategy with partial moat or partial execution evidence.</div>
    <div class="rubric-band band-c"><span class="band-label">S-C (0.38&ndash;0.55):</span> aspirational/generic strategy ("partner with hospitals," "sell to enterprises") without differentiation evidence.</div>

    <h4>Feasibility (four bands)</h4>
    <div class="rubric-band band-a"><span class="band-label">F-A (0.78&ndash;0.92):</span> itemized budget, realistic multi-year milestones, named regulatory pathway, existing infrastructure or confirmed manufacturing partner.</div>
    <div class="rubric-band band-b"><span class="band-label">F-B (0.58&ndash;0.75):</span> reasonable budget with partial detail, milestones exist.</div>
    <div class="rubric-band band-c"><span class="band-label">F-C (0.38&ndash;0.55):</span> budget stated without breakdown, vague timeline, risks acknowledged but not mitigated.</div>
    <div class="rubric-band band-d"><span class="band-label">F-D (0.20&ndash;0.35):</span> unrealistic or missing budget, unresolved critical dependencies.</div>

    <p>The model must return, per dimension: a numeric <code>score</code> (0&ndash;1), a <code>category</code>
    letter code, and a <code>rationale</code> citing specific evidence.</p>

    <h3>Blending investment score with QA score</h3>
    <div class="formula-box">INVEST_WEIGHT = 0.90
blended_dim_score = 0.90 &times; invest_score[dim] + 0.10 &times; qa_score[dim]</div>
    <p>i.e. the deterministic, fact-grounded investment rubric dominates (90%) over the QA-answer-quality score
    (10%). All three values &mdash; <code>score_echo</code>, <code>qa_score_echo</code>, and
    <code>invest_score_echo</code> &mdash; are stored per dimension for audit and downstream calibration (this is
    what <code>calibration_sweep.py</code> reads back without making any new LLM calls).</p>
    <p>The same dimension weights from Stage 5 (team=1.0, objectives=1.0, strategy=1.0, innovation=1.10,
    feasibility=1.20) are re-applied to the blended per-dimension scores to produce the final
    <code>blended_overall</code> score, which is what feeds the verdict rule below.
    <code>overall_confidence</code> is carried over unchanged from Stage 5.</p>
    <p class="cite">Source: ai_expert_opinion.py:102-228 (full rubric),231-306,519,644-718,1018-1131,1056.</p>
    """
    return sec(10, "Stage 6 &mdash; AI Expert Opinion &amp; the Investment Rubric", body)


def s10_stage7():
    body = """
    <p><b>Script:</b> <code>src/tools/generate_final_report.py</code> (566 lines). <b>Input:</b> Stage 6's
    expert-opinion JSON + Stage 5's final Q&amp;A payload. <b>Output:</b>
    <code>src/data/reports/&lt;pid&gt;_final_report.md</code>.</p>
    <p>The compiled Markdown report has three top-level sections:</p>
    <ol>
      <li><b>Executive Summary</b> &mdash; overall verdict (GO/HOLD/NO-GO with a plain-language action note),
      one-line decision basis, overall score and confidence, a <b>reader-facing score-band guide</b>
      (&ge;0.62 = "conditions look relatively strong"; 0.45&ndash;0.62 = "evidence incomplete/mixed";
      &lt;0.45 = "clear weaknesses" &mdash; note these are informational thresholds for non-technical readers and
      are <i>different numbers</i> from the actual decision-rule thresholds described in Section 13 below), top 3
      strengths, top 3 risks/gaps, 3&ndash;5 recommended next actions, and a deterministic "Scale-Up Diligence
      Focus" checklist.</li>
      <li><b>AI Expert Review (Detailed)</b> &mdash; the full Stage 6 narrative commentary, with heading levels
      demoted and a redundant compact-score echo list stripped out.</li>
      <li><b>Dimension Q&amp;A and Scoring Evidence</b> &mdash; per-dimension aggregate score plus selected
      question/answer traceability rows (capped at 10 rows per dimension in the report; the full uncapped set
      remains in <code>final_payload.json</code>).</li>
    </ol>
    <p class="cite">Source: generate_final_report.py:139-193,197-272,276+,407-566.</p>
    """
    return sec(11, "Stage 7 &mdash; Final Report Generation", body)


def s11_stage8():
    body = """
    <p><b>Script:</b> <code>src/tools/generate_llm_scores.py</code> (312 lines standalone; fan-out version
    <code>generate_multi_provider_scores.py</code> in the legacy execution mode). <b>Input:</b> the compiled Stage 7
    report Markdown (trimmed to 12,000 characters) plus the expert-opinion summary/verdict/risks/strengths.
    <b>Output:</b> <code>src/data/llm_scores/&lt;pid&gt;/llm_scores.json</code>.</p>
    <p>This stage asks the LLM to score the <i>report</i> (not the raw proposal facts) directly on the same scale
    used by human experts: five dimensions on a 1&ndash;5 scale, an overall ranking on a 1&ndash;20 scale, and a
    binary GO/NO-GO verdict. The system prompt gives explicit calibration anchors (1&ndash;2 = critical gap, 3 =
    common/screened-proposal quality, 4 = meaningfully strong, 5 = exceptional) and instructs a "Y" verdict when
    at least 3 of 5 dimensions score 4 or above. On failure, it falls back to a deterministic linear remap of
    Stage 6's blended scores (<code>1 + score&times;4</code> for the 1&ndash;5 scale,
    <code>1 + score&times;19</code> for the 1&ndash;20 scale).</p>
    <p><b>Dimension name mapping</b> used at this stage (and in all evaluation code that compares against the
    human rubric): <code>team&rarr;team, objectives&rarr;objective, strategy&rarr;strategy,
    innovation&rarr;advantages, feasibility&rarr;feasibility</code>. This is the single place the internal name
    "innovation" becomes the externally-reported name "advantages" &mdash; important to note consistently when
    writing about the pipeline, since the two names refer to the same dimension throughout.</p>
    <p class="cite">Source: generate_llm_scores.py:1-17,104-134,175-196; README.md:22-40 (mapping table).</p>
    """
    return sec(12, "Stage 8 &mdash; LLM Rubric Scoring", body)


def s12_verdict(sweep):
    seen = {}
    for r in sweep:
        seen.setdefault(r["invest_weight"], r["spearman"])
    peak_val = max(seen.values())
    peak_iws = sorted(iw for iw, sp in seen.items() if abs(sp - peak_val) < 1e-9)
    sweep_rows = "".join(
        f"""<tr{' class="row-total"' if iw in peak_iws else ''}><td class="pid">{iw:.2f}</td><td>{sp:+.4f}</td></tr>"""
        for iw, sp in sorted(seen.items())
    )
    sweep_table = f"""
    <table style="max-width:360px">
      <thead><tr><th>invest_weight</th><th>Spearman &rho; vs. human scores</th></tr></thead>
      <tbody>{sweep_rows}</tbody>
    </table>"""
    sweep_peak_iw = " and ".join(f"{iw:.2f}" for iw in peak_iws)
    sweep_peak_val = f"{peak_val:+.4f}"

    body = f"""
    <p>The pipeline's final decision layer &mdash; <code>verdict_rule()</code>, at
    <code>ai_expert_opinion.py:732-749</code> &mdash; converts the blended overall score and confidence into one of
    three verdicts:</p>
    <div class="formula-box">def verdict_rule(score, conf, dims):
    inv = dims["innovation"]["avg"]     # blended (post Stage-6) value
    fea = dims["feasibility"]["avg"]    # blended (post Stage-6) value

    if score &gt;= 0.725 or conf &gt;= 0.811:
        return "GO"

    if score &lt; 0.40 or inv &lt; 0.30 or fea &lt; 0.30:
        return "NO-GO"

    return "HOLD"</div>
    <p><b>GO</b> fires on an OR-condition: either the overall score clears 0.725, <i>or</i> confidence alone clears
    0.811 &mdash; either is sufficient. <b>NO-GO</b> fires if the score is very low, or if either innovation or
    feasibility individually falls below 0.30, regardless of the overall score. Everything else is <b>HOLD</b>.</p>
    <h3>Where these exact numbers came from</h3>
    <p>The 0.725/0.811 pair was <b>not hand-picked</b> &mdash; it was produced by
    <code>calibration_sweep.py</code>'s <code>sweep_verdict_thresholds()</code> function, a 2-D grid search over
    candidate score/confidence thresholds combined with OR logic, run against the 12-proposal human-expert dataset
    to maximize verdict accuracy. This zero-LLM-call sweep reads cached <code>invest_score_echo</code> /
    <code>qa_score_echo</code> values, so re-running it costs nothing. It replaced an earlier AND-rule
    (<code>score &ge; 0.65 AND confidence &ge; 0.65</code>), improving verdict accuracy from 58.3% to 83.3% on the
    12-proposal set (full comparison in Section 24).</p>
    <div class="plain"><b>Important nuance for accuracy when writing about this system:</b> this OR-logic
    threshold pair was calibrated exclusively on the 12-proposal biotech/semiconductor dataset, <i>in-sample</i>
    (fit and evaluated on the same 12 proposals, with no held-out check). When this exact threshold was later
    applied unmodified to the 141-proposal Kickstarter dataset, it produced <b>zero GO verdicts</b>, because
    Kickstarter proposals structurally never reach scores anywhere near 0.725 (see Sections 25 and 27). This is
    the central empirical finding motivating the entire Kickstarter calibration study in Part IV.</div>

    <h3>The empirical sweep behind INVEST_WEIGHT=0.90 (previously undocumented in this report)</h3>
    <p><code>results/Dataset1/calibration_sweep.json</code> contains the actual 168-point grid search
    (<code>invest_weight</code> &times; <code>verdict_threshold</code>) that justifies the production
    <code>INVEST_WEIGHT=0.90</code> value. Spearman correlation with the 12 human expert scores, as
    <code>invest_weight</code> is swept from 0 (QA-answer-quality score only) to 1 (investment rubric score
    only):</p>
    {sweep_table}
    <div class="plain">Two things worth noting: (1) at <code>invest_weight=0.0</code>, Spearman is
    <b>&minus;0.325</b> &mdash; this is the exact number behind the research paper's separately-reported finding
    that "QA-only scoring reverses expert ranking" (Section 24) &mdash; the sweep is the source of that claim, not
    an independent confirmation of it. (2) Spearman rises essentially monotonically as more weight shifts to the
    investment rubric, peaking (tied) at <b>invest_weight = {sweep_peak_iw}</b> (Spearman {sweep_peak_val}) &mdash;
    the production value of 0.90 is one of the two tied peak points, a real empirical justification rather than an
    arbitrary round number. (3) This
    sweep's peak Spearman ({sweep_peak_val}) is noticeably lower than the <b>0.589</b> reported in the live
    evaluation (Section 24) &mdash; the sweep computes correlation directly from the blended
    <code>invest_score_echo</code>/<code>qa_score_echo</code> values (Stage 6 output), while the live evaluation
    compares against Stage 8's separate LLM re-scoring of the compiled report on the human 1-5/1-20 scale &mdash;
    two different, non-interchangeable "AI score" pipelines that happen to correlate with each other but are not
    the same measurement.</div>
    <p class="cite">Source: ai_expert_opinion.py:732-749,1122; calibration_sweep.py:137,163-217; README.md:786-793;
    results/Dataset1/calibration_sweep.json (live).</p>
    """
    return sec(13, "The Verdict Rule &amp; Score Formulas", body)


def s13_operational():
    body = """
    <p>Per the project's own research paper (Section 4.7), the 12-proposal OpenAI evaluation run made
    <b>1,868 logged LLM calls</b> in total, averaging <b>~156 calls per proposal</b>, with a mean per-call latency
    of <b>8.74 seconds</b> (median 7.46 seconds); calls are partially parallelized across stages. No aggregate
    token or dollar cost is reported for the batch (a documented gap).</p>
    <p>Separately, a performance engineering pass (documented in <code>docs/pipeline_time_sink_analysis.md</code>)
    root-caused two major bottlenecks &mdash; a stalled Stage 1 OpenAI call (fixed via a 120-second
    <code>OPENAI_TIMEOUT_SECONDS</code>) and a Stage 4 batch-call schema mismatch (fixed directly in
    <code>llm_answering.py</code>) &mdash; and introduced concurrency controls
    (<code>STAGE1_CHUNK_WORKERS=4</code>, <code>STAGE4_DIM_WORKERS=2</code>, <code>STAGE4_VARIANT_WORKERS=3</code>,
    <code>STAGE4_REFINE_WORKERS=3</code>). Total wall-clock time for a full proposal run dropped from
    <b>~10h 4m</b> to <b>~28m 32s</b> (a ~21&times; speedup) between two benchmarked runs.</p>
    <p class="cite">Source: research_paper.pdf Section 4.7; docs/pipeline_time_sink_analysis.md.</p>
    """
    return sec(14, "Operational Footprint", body)


def s15_related_work():
    body = """
    <p>The project's research paper positions this work against three literatures. Reproducing that positioning
    here is useful context for framing related-work sections of any derived paper.</p>

    <h3>Financial LLMs and venture screening</h3>
    <p>Domain-specific financial LLMs (BloombergGPT, FinGPT, PIXIU) and finance benchmarks (FinEval, CFBenchmark,
    FinBen) establish that finance is an active LLM-specialization area, but target general financial NLP rather
    than proposal screening specifically. Grounded financial QA work (FinanceRAG, Multi-Reranker) targets retrieval
    correctness, not investment scoring. <b>FAITH</b> is directly relevant: it shows LLMs produce unsupported
    numerical answers even with source tables in context, worse as reasoning moves from direct lookup to
    compositional use &mdash; the same hallucination risk this project's groundedness audit (Section 24) checks
    for directly in the venture-screening setting. A separate investment-bias study found LLMs tend to preserve an
    initial investment preference despite counter-evidence, arguing investment agents should be evaluated on
    intermediate behavior, robustness, and human collaboration &mdash; not just final-task accuracy, a framing this
    project's whole "outcome-free audit" design follows.</p>
    <p><b>DIALECTIC</b> is the closest prior startup-evaluation system: it structures startup facts, question
    trees, pro/con reasoning, critique, and numeric investment scores &mdash; architecturally similar to this
    pipeline's fact&rarr;dimension&rarr;question&rarr;answer&rarr;score decomposition. The paper's stated point of
    novelty relative to DIALECTIC and similar systems: <i>"Our novelty is not another multi-agent scoring
    architecture. We instead treat the produced score as the object of validation."</i> VCBench and
    founder-success-prediction work are noted as adjacent but explicitly out of scope &mdash; this project audits
    measurement validity, it does not attempt outcome prediction (with the partial exception of the Kickstarter
    real-outcome study added in this session, which is itself framed as a validity check, not a launched
    predictor).</p>

    <h3>Grounded and trustworthy financial decision support</h3>
    <p>RAGAS and ARES's separation of context quality from answer faithfulness underlies the Stage 4 answering
    schema (Section 8). FActScore's atomic-fact decomposition underlies Stage 1 (Section 5) and the claim-grounding
    taxonomy (Section 24). SelfCheckGPT (claim-level hallucination detection without external references) is cited
    as a related but distinct approach the groundedness audit does not directly implement. Prior ICAIF-adjacent
    work on investment-risk models varying by country/gender, balanced arguments revealing latent investment
    preferences under counter-evidence, and transparency/prompt-design effects on human-AI investment decisions all
    motivate treating this system's outputs as decision-support signals requiring audit, not authoritative
    judgments.</p>

    <h3>Human-AI agreement and the measurement gap</h3>
    <p>A cited systematic review warns that task formulation, metric choice, benchmark construction, and reporting
    practices can produce contradictory conclusions across LLM-evaluation studies &mdash; part of the paper's
    motivation for reporting agreement, ranking, calibration, and dimension reliability as <i>separate</i> numbers
    rather than collapsing them into one "the AI is good/bad" verdict (see the four-target validity framework,
    Section 2). LLM-as-judge literature (G-Eval, MT-Bench, Chatbot Arena, Prometheus-style rubric evaluation) is
    cited as the closest methodological neighbor for the sensitivity/content-injection checks, alongside the
    caveat that using human expert scores as the operational reference (rather than another LLM judge) avoids
    compounding one LLM's biases with another's.</p>
    <p class="cite">Source: research_paper.pdf Section 2 (Related Work), 41-reference bibliography.</p>
    """
    return sec(15, "Related Work &amp; Positioning", body)


# ── Part II: Datasets ────────────────────────────────────────────────────────

def s14_dataset12():
    proposals = [
        ("1", "心衰专病大模型BP", "Heart-failure disease-specific large-model project plan", "Biotech / medical AI"),
        ("2", "新型高端酶制剂的智能化创制与应用", "Novel high-end enzyme-preparation intelligent creation &amp; application", "Biotech"),
        ("3", "基于非天然氨基酸调控的高效抗体结合蛋白", "Antibody-binding protein regulated by non-natural amino acids", "Biotech"),
        ("4", "第二代肿瘤治疗电场（TTF）解决方案", "China's first 2nd-generation Tumor Treating Fields (TTF) solution", "Medtech"),
        ("5-bp1", "苏州纳飞卫星动力有限公司", "Satellite propulsion company business plan", "Aerospace"),
        ("6-bp2", "特种车辆自动驾驶智能计算单元", "Autonomous-driving compute unit for special vehicles", "Automotive / AI hardware"),
        ("7", "LPNP-mRNA免疫调节因子项目", "LPNP-mRNA immunomodulatory-factor project", "Biotech"),
        ("8", "Ebovir LNP拨投结合项目", "Lung-targeted LNP gene-drug delivery platform", "Biotech"),
        ("A", "安海半导体产业化方案", "Semiconductor industrialization plan", "Semiconductor"),
        ("B", "BMT技术（基石资本）", "\"Dashun\" technology, Series-B, Cornerstone Capital", "Hardware"),
        ("C", "ZX学院商业计划书", "Education/academy venture business plan", "Education"),
        ("D", "大瞬科技投决报告", "Same company as B, Series-C investment decision report", "Hardware"),
    ]
    rows = "".join(f'<tr><td class="pid">{pid}</td><td style="text-align:left">{cn}</td><td style="text-align:left">{en}</td><td>{dom}</td></tr>' for pid, cn, en, dom in proposals)
    body = f"""
    <p>The human-expert-compared dataset consists of <b>12 real Chinese-language venture/investment proposals</b>
    (PDF/PPTX), spanning two source folders (<code>DeltaProjectDatasets/Dataset1</code>, 8 proposals p1&ndash;p8;
    <code>Dataset2</code>, 4 proposals pA&ndash;pD). Despite the folder names, the domain mix is broader than
    "biotech" alone &mdash; it includes aerospace, automotive AI hardware, semiconductor, and education ventures
    alongside biotech/medtech, which is worth stating precisely rather than calling it a pure biotech dataset.</p>
    <table>
      <thead><tr><th>PID</th><th>Project (Chinese)</th><th>Project (English gloss)</th><th>Domain</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <h3>Ground truth: <code>ExpertsEvaluations.xlsx</code></h3>
    <p>A single-sheet spreadsheet with 12 rows (one per proposal) and columns: <code>ID, file name, Project Name,
    Team, Objective, Strategy, Advantages, Feasibility</code> (each 1&ndash;5), <code>Overall Ranking</code>
    (1&ndash;20), and <code>Funded Y/N</code>. This is the human-expert score set every AI-vs-human comparison in
    this project is measured against. The project's paper explicitly describes this file as <i>"an operational
    reference rather than universal ground truth"</i> &mdash; there is no second human rater and no held-out test
    set, a limitation carried through every result derived from it.</p>
    <p class="cite">Source: DeltaProjectDatasets/Dataset1, Dataset2; ExpertsEvaluations.xlsx.</p>
    """
    return sec(16, "Datasets 1 &amp; 2 &mdash; Biotech / Semiconductor Venture Proposals", body)


def s15_kickstarter():
    body = """
    <p>To obtain an evaluation with objective, real-world ground truth at a scale the 12-proposal expert study
    cannot reach, this project separately sourced <b>141 real Kickstarter crowdfunding campaigns</b>: 104 from
    Toronto and 37 from Montreal (pid prefixes <code>kt</code> and <code>km</code> respectively), each with its own
    campaign PDF and a real <code>funded (%)</code> outcome recorded in <code>Kickstarter/dataset_info.xlsx</code>
    (a 3-sheet workbook: <code>Combined</code>, <code>Toronto, Canada</code>, <code>Montreal, Canada</code>).</p>
    <p><b>Ground truth definition:</b> Kickstarter uses all-or-nothing funding &mdash; a campaign that reaches 99%
    of its goal still collects $0. This makes <code>funded % &ge; 100</code> an unambiguous, objective binary
    success label with no partial-credit judgment call, unlike a subjective "did this venture succeed" label.</p>
    <p><b>Coverage:</b> all 141 campaigns were run through the unmodified 9-stage pipeline (same code, same rubric,
    same weights, same verdict rule as the biotech/semiconductor dataset) and scored &mdash; a genuine
    out-of-domain generalization test, since the pipeline and its rubric were designed and tuned entirely on VC-style
    biotech/hardware proposals, never touched for consumer crowdfunding.</p>
    <p class="cite">Source: Kickstarter/dataset_info.xlsx; src/tools/kickstarter_dataset.py.</p>
    """
    return sec(17, "The Kickstarter Dataset &mdash; A Real-World Outcome Benchmark", body)


# ── Part III: Methodology ────────────────────────────────────────────────────

def s16_methodology_human():
    body = """
    <p>All comparisons between AI and human-expert scores use the following metrics, computed by
    <code>evaluate_cohens_kappa.py</code> (despite its name, it computes more than kappa):</p>
    <ul>
      <li><b>Quadratic-weighted Cohen's &kappa;</b> &mdash; both AI and human scores are discretized onto a 5-point
      grid (0, 0.25, 0.50, 0.75, 1.00) before computing agreement beyond chance, with bootstrap 95% confidence
      intervals. The paper explicitly flags &kappa; as noisy here because AI scores are heavily concentrated in
      one or two bins (sparse marginal occupancy).</li>
      <li><b>ICC(2,1)</b> (Intraclass Correlation) &mdash; used specifically for the continuous overall-ranking
      score, with a 95% CI, as a measure of absolute-value agreement rather than just ranking agreement.</li>
      <li><b>Spearman rank correlation</b> (&rho;) with p-value &mdash; the primary ranking-agreement metric,
      computed on raw 0&ndash;1 floats (not binned), for each of the 5 dimensions plus the overall ranking.</li>
      <li><b>Paired mean-difference bias check</b> &mdash; mean(AI&minus;human), tested with a paired t-test (and a
      Wilcoxon signed-rank sensitivity check), to detect systematic score-scale offset.</li>
      <li><b>Pairwise rank concordance</b> &mdash; of all pairs of proposals the human ranked in a definite order
      (ties excluded), what fraction does the AI rank in the same relative order? Directly measures whether
      screening priority ordering is preserved.</li>
      <li><b>Verdict classification metrics</b> &mdash; accuracy, precision, recall, F1, and the full confusion
      matrix, comparing the AI's binary GO/NO-GO verdict against the human "Funded Y/N" label.</li>
    </ul>
    <p><b>Multiple-testing correction:</b> the six Spearman tests (5 dimensions + overall) are treated as one
    family and corrected using the <b>Holm method</b> &mdash; chosen as more powerful than Bonferroni while still
    conservative. The six paired mean-difference tests form a second Holm-corrected family.</p>
    <p><b>Statistical power context:</b> at n=12, a two-sided correlation test needs |&rho;| &asymp; 0.73 for 80%
    statistical power at &alpha;=0.05 (Fisher-transformation calculation) &mdash; explicitly noted in the paper as
    a reason to treat single-dataset correlation results here as exploratory, not confirmatory.</p>
    <p class="cite">Source: research_paper.pdf Section 5; evaluate_cohens_kappa.py:1-27.</p>
    """
    return sec(18, "Methodology A &mdash; Human-Expert Agreement", body)


def s17_methodology_outcome():
    body = """
    <p><code>evaluate_kickstarter_correlation.py</code> compares AI scores against real funding outcomes, computed
    separately for Toronto, Montreal, and combined:</p>
    <ul>
      <li><b>Overall Spearman correlation</b> between the AI's <code>overall_score</code> (0&ndash;1) and actual
      <code>funded (%)</code>, with p-value and Holm-corrected p-value. Spearman (rank-based) is used instead of a
      linear correlation because funded % is extremely right-skewed &mdash; most campaigns raise 0&ndash;150% of
      goal, a handful raise over 1000%.</li>
      <li><b>Per-dimension Spearman</b> &mdash; each of the 5 dimension scores correlated individually against
      funded %, to see which dimensions carry real predictive signal in this domain.</li>
      <li><b>Binary classification</b> &mdash; treating the AI's GO verdict as a prediction of
      <code>funded % &ge; 100</code>, with the full accuracy/precision/recall/F1 confusion-matrix breakdown.</li>
    </ul>
    <p class="cite">Source: evaluate_kickstarter_correlation.py:1-25.</p>
    """
    return sec(19, "Methodology B &mdash; Real-World Outcome Validation (Kickstarter)", body)


def s18_methodology_calib():
    body = """
    <p><code>calibrate_kickstarter_threshold.py</code> re-derives a Kickstarter-appropriate GO decision threshold,
    keeping the exact same OR-logic shape as production (<code>score &ge; &tau;<sub>r</sub> OR confidence &ge;
    &tau;<sub>c</sub></code>) but searching for new values of &tau;<sub>r</sub>, &tau;<sub>c</sub> instead of
    reusing the biotech-derived 0.725/0.811 pair.</p>
    <p><b>Why cross-validation instead of a single train/test split:</b> only 24 of 141 Kickstarter proposals
    actually succeeded. A single 70/30 split would leave as few as 7 successes in the test set &mdash; too few for
    a stable precision/recall estimate. Instead, <b>stratified 5-fold cross-validation</b> (seed=42) is used: the
    141 proposals are split into 5 folds, each with a proportional (~17%) share of successes. For each of the 5
    rounds, one fold is held out as test data, the other 4 are used to grid-search the F1-maximizing
    (&tau;<sub>r</sub>, &tau;<sub>c</sub>) pair (candidate thresholds are midpoints between consecutive sorted
    training-fold values, so every real decision boundary in that fold is tested), and the chosen threshold is
    applied only to the held-out fold. Every proposal is evaluated exactly once, using a threshold that never saw
    it. The 5 test-fold confusion matrices are summed into one aggregated result.</p>
    <p><b>Why F1, not accuracy, is the calibration objective:</b> only 17% of campaigns succeed, so "always predict
    failure" already scores 83% accuracy without ever identifying a real success &mdash; exactly the failure mode
    of the original transferred threshold. F1 (the harmonic mean of precision and recall) forces the calibration
    to actually find successes.</p>
    <p>Three scenarios are reported side by side: (1) <b>before</b> &mdash; the unmodified production threshold;
    (2) <b>in-sample</b> &mdash; threshold grid-searched and evaluated on the same 141 proposals, reported only as
    an overfitting-risk reference; (3) <b>cross-validated / held-out</b> &mdash; the honest, generalizable result.</p>
    <p class="cite">Source: calibrate_kickstarter_threshold.py (full docstring and implementation).</p>
    """
    return sec(20, "Methodology C &mdash; Threshold Calibration via Cross-Validation", body)


def s19_methodology_improve():
    body = """
    <p><code>improve_kickstarter_score.py</code> tests whether the underlying score itself (not just the
    threshold) can be improved, given that the earlier correlation study (Section 25) found team and objective
    carry no significant relationship with real Kickstarter outcomes while feasibility, strategy, and advantages
    do. Three scoring methods are compared head-to-head, all evaluated via the same 5-fold CV (same folds/seed),
    using a single score threshold (no confidence OR-gate, to isolate score quality alone):</p>
    <ul>
      <li><b>Method A &mdash; overall_score:</b> the unmodified production score (baseline).</li>
      <li><b>Method B &mdash; simple composite:</b> a plain, unweighted average of feasibility + strategy +
      advantages only, dropping team and objective. Initially computed using dimension-selection derived from the
      full 141-proposal correlation study (a design choice later shown to be a soft form of leakage &mdash; see
      Section 22).</li>
      <li><b>Method C &mdash; logistic regression:</b> an L2-regularized model (regularization strength C=1.0,
      fit via <code>scipy.optimize.minimize</code>) that learns its own weights across all 5 standardized
      dimension scores, fit strictly within each training fold (never seeing its own test fold), producing pooled
      out-of-fold predicted probabilities for evaluation.</li>
    </ul>
    <p>Discrimination is additionally measured in a threshold-free way via <b>ROC-AUC</b> (the probability a
    randomly chosen success scores higher than a randomly chosen failure &mdash; computed via the
    rank-based Mann-Whitney formula) and <b>Average Precision</b>, both independent of any specific decision
    threshold.</p>
    <p class="cite">Source: improve_kickstarter_score.py (full docstring and implementation).</p>
    """
    return sec(21, "Methodology D &mdash; Score Improvement Testing", body)


def s20_methodology_robust():
    body = """
    <p><code>kickstarter_score_robustness_check.py</code> performs two rigor checks specifically on the Method B
    result, after recognizing that its "no fitting" claim was incomplete &mdash; the choice of <i>which</i> three
    dimensions to keep was itself made using full-dataset correlations, a softer form of the same leakage problem
    the threshold-calibration study exists to prevent.</p>
    <h3>Check 1 &mdash; Bootstrap confidence intervals</h3>
    <p>Each pairwise ROC-AUC difference (B vs A, B vs C, C vs A) is bootstrapped <b>10,000 times</b>: resample the
    141 proposals with replacement, recompute both methods' AUC on the identical resample, record the difference,
    repeat. The resulting distribution gives a 95% confidence interval on the true AUC gap and a
    "probability the second method is better" estimate (fraction of resamples where it was). A gap is only called
    statistically significant if the 95% CI excludes zero.</p>
    <h3>Check 2 &mdash; Per-fold dimension-selection stability, and a fully-nested re-evaluation</h3>
    <p>The "which 3 dimensions?" decision is independently re-derived <b>5 times</b>, using only each training
    fold's own data (Spearman correlation between each dimension and <code>funded_pct</code>, computed on the
    training fold only, keeping the top 3 by correlation). This checks whether the same three dimensions are
    picked when the full dataset isn't visible. A <b>fully-nested Method B</b> is then evaluated: for each fold,
    both the dimension selection <i>and</i> the decision threshold are fit using only that fold's training data,
    and the resulting composite score is evaluated only on the held-out fold &mdash; the methodologically strongest
    version of Method B, with no part of the number touching the full 141-proposal dataset at any point.</p>
    <p class="cite">Source: kickstarter_score_robustness_check.py (full docstring and implementation).</p>
    """
    return sec(22, "Methodology E &mdash; Robustness Checks (Bootstrap &amp; Nested Validation)", body)


def s21_methodology_ground():
    body = """
    <p>Two further outcome-free audits are documented in the project's existing research paper, applied only to
    the 12-proposal biotech/semiconductor dataset:</p>
    <h3>Claim grounding</h3>
    <p>Report statements are classified into four categories: verifiable facts, numerical inferences, professional
    inferences, and evaluator judgments. Only verifiable facts and numerical inferences count toward the
    groundedness denominator (professional inferences are analyst judgments; scores/confidence/verdicts are
    evaluator outputs, not source-document claims, and are excluded). Each eligible claim is labeled supported,
    partially supported, or unsupported:</p>
    <div class="formula-box">Strict groundedness  G_strict  = N_supported / N_eligible
Lenient groundedness G_lenient = (N_supported + 0.5&middot;N_partial) / N_eligible
Unsupported rate     U         = N_unsupported / N_eligible</div>
    <h3>Controlled content sensitivity</h3>
    <p>PLUS/MINUS proposal-specific variants are constructed at the report/rubric input level for Strategy,
    Objectives, and Advantages (each with 8 fine-grained sub-rubrics). PLUS injects concrete favorable evidence
    (revenue model, regulatory pathway, IP status, benchmark result, market estimate, buyer pathway); MINUS
    replaces the strongest available evidence with explicit absence statements. The check measures whether scores
    move in the theoretically expected direction. This check is explicitly <b>prompt-coupled, not independent</b>:
    because the generation prompt names the target rubric and the desired direction, it measures
    generation&ndash;scoring <i>consistency</i> under rubric-targeted edits, not whether realistic new evidence
    encountered "in the wild" would move scores the same way.</p>

    <h3>Two sensitivity-generation modes</h3>
    <p>The tooling (<code>run_sensitivity_all_dimensions.py</code>) supports two modes: <b>Default (hardcoded)</b>
    &mdash; fixed content, 2 sub-rubrics tested per dimension across strategy/advantages/objectives; and
    <b>Adaptive</b> (<code>--adaptive</code> flag) &mdash; the LLM generates proposal-specific PLUS/MINUS content
    covering all 8 sub-rubrics per dimension (16 checks per proposal), giving denser coverage at the cost of
    variable, LLM-generated injected text rather than a fixed script.</p>

    <h3>The 8 sub-rubrics per dimension (R1&ndash;R8)</h3>
    <p>Each of Strategy, Advantages, and Objectives is additionally scored on 8 fine-grained sub-rubrics, split
    into two groups with different expected AI-vs-human agreement:</p>
    <ul>
      <li><b>R1&ndash;R4 (document-verifiable):</b> milestone dates, revenue model, regulatory pathway, market
      size &mdash; facts that are either stated in the document or not. The AI is expected to track human scoring
      closely here, since both raters are reading the same explicit text.</li>
      <li><b>R5&ndash;R8 (judgment-dependent):</b> competitive-moat credibility, team-strategy fit, market timing
      &mdash; and named specifically in the README as the weakest cells: <b>Problem-Solution Fit, Market Timing,
      and Competitive Defensibility</b>. These require inference and domain judgment beyond what's explicitly
      written, and the AI is expected to diverge more from human raters here.</li>
    </ul>
    <p>A blank <code>human_scorecard_template.csv</code> exists in the repository specifically to collect a
    <i>second</i> human rater's sub-rubric scores, so a human-vs-human agreement ceiling can eventually be computed
    and compared against the AI-vs-human numbers &mdash; this has not yet been done (see Limitations, Section 33).</p>

    <h3>How PLUS/MINUS variants are actually constructed (implementation detail)</h3>
    <p>An internal implementation review (<code>sensitivity_analysis_report.html</code>) documents the exact
    mechanics, which sharpen the "prompt-coupled" caveat above into something more specific and mechanistic:</p>
    <ul>
      <li><b>PLUS injection</b> prepends a fixed block of pre-written evidence text immediately before a specific
      report section heading (falling back to inserting before the Q&amp;A section if that heading isn't found).
      The injected text is <b>identical across all 12 proposals</b> for a given dimension &mdash; it is not
      proposal-specific, generated evidence.</li>
      <li><b>MINUS stripping</b> only removes narrative bullet/numbered lines containing specific target keywords
      and applies a fixed list of inline regex substitutions. <b>It does not touch the report's Q&amp;A evidence
      section</b>, which frequently re-states the same underlying concepts in different vocabulary &mdash; so a
      MINUS variant can leave much of the model's actual evidence base intact even when the narrative section
      reads as stripped. If a proposal's narrative never used the targeted keywords in the first place, its MINUS
      variant is byte-for-byte identical to its baseline.</li>
      <li><b>Variant re-scoring captures only the numeric score</b>, not the evidence/reasoning fields the baseline
      scorer captures &mdash; a deliberate speed tradeoff that means root-cause analysis of any given score
      movement (or non-movement) has to be inferred from the report text and rubric definitions after the fact,
      not read directly off a captured explanation.</li>
    </ul>
    <p class="cite">Source: research_paper.pdf Sections 5.4, 5.5; README.md "Research &amp; Calibration Tools";
    results/Dataset1/sensitivity_analysis_report.html.</p>
    """
    return sec(23, "Methodology F &mdash; Groundedness &amp; Content-Sensitivity Audits", body)


# ── Part IV: Results ─────────────────────────────────────────────────────────

def _pid_short(fname: str) -> str:
    head = fname.split("-")[0].split("_")[0].split("附")[0]
    return head.strip()


def _load_dataset1_per_item():
    wb = openpyxl.load_workbook(DATASET1_XLSX_PATH, data_only=True)
    ws = wb["per_item"]
    rows = list(ws.iter_rows(values_only=True))[1:]
    out = []
    for r in rows:
        out.append({
            "pid": _pid_short(r[0]),
            "verdict_human": r[3],
            "team_h": r[6], "team_a": r[7],
            "obj_h": r[9], "obj_a": r[10],
            "strat_h": r[12], "strat_a": r[13],
            "adv_h": r[15], "adv_a": r[16],
            "fea_h": r[18], "fea_a": r[19],
            "overall_h": r[21], "overall_a": r[22],
        })
    return out


def s24_results_human(d1):
    m = d1["metrics"]
    rows = "".join(
        f"""<tr><td class="dim-name">{dim.capitalize()}</td><td>{m[dim]['weighted_kappa']:.3f}
        [{m[dim]['weighted_kappa_ci95'][0]:.3f}, {m[dim]['weighted_kappa_ci95'][1]:.3f}]</td>
        <td class="{'pos' if m[dim]['spearman_r']>=0 else 'neg'}">{m[dim]['spearman_r']:+.3f}</td>
        <td>{m[dim]['spearman_p']:.4f}</td><td>{m[dim]['mean_human']:.3f}</td><td>{m[dim]['mean_ai']:.3f}</td></tr>"""
        for dim in ("team", "objective", "strategy", "advantages", "feasibility")
    )
    ov = m["overall_ranking"]
    ro = d1["rank_order"]
    discordant_rows = "".join(
        f"""<tr><td class="pid">{_pid_short(e['a'])}</td><td class="pid">{_pid_short(e['b'])}</td>
        <td>{e['human_a']:.3f}</td><td>{e['human_b']:.3f}</td>
        <td>{e['ai_a']:.3f}</td><td>{e['ai_b']:.3f}</td></tr>"""
        for e in ro["discordant_examples"][:10]
    )
    per_item = _load_dataset1_per_item()
    per_item_rows = "".join(
        f"""<tr><td class="pid">{r['pid']}</td><td>{r['verdict_human']}</td>
        <td>{r['team_h']:.3f}</td><td>{r['team_a']:.3f}</td>
        <td>{r['obj_h']:.3f}</td><td>{r['obj_a']:.3f}</td>
        <td>{r['strat_h']:.3f}</td><td>{r['strat_a']:.3f}</td>
        <td>{r['adv_h']:.3f}</td><td>{r['adv_a']:.3f}</td>
        <td>{r['fea_h']:.3f}</td><td>{r['fea_a']:.3f}</td>
        <td>{r['overall_h']:.3f}</td><td>{r['overall_a']:.3f}</td></tr>"""
        for r in per_item
    )
    per_item_table = f"""
    <table>
      <thead><tr><th rowspan="2">PID</th><th rowspan="2">Funded</th>
        <th colspan="2">Team</th><th colspan="2">Objective</th><th colspan="2">Strategy</th>
        <th colspan="2">Advantages</th><th colspan="2">Feasibility</th><th colspan="2">Overall</th></tr>
      <tr><th>H</th><th>AI</th><th>H</th><th>AI</th><th>H</th><th>AI</th><th>H</th><th>AI</th><th>H</th><th>AI</th><th>H</th><th>AI</th></tr></thead>
      <tbody>{per_item_rows}</tbody>
    </table>"""
    body = f"""
    <p>Full current results from <code>results/Dataset1/evaluation/evaluation_summary.json</code>
    (run {d1['run_id']}, {d1['matched_rows']} matched proposals, {d1['skipped_rows']} skipped).</p>
    <div class="plain"><b>Historical context:</b> this is the result of an earlier rubric redesign. Before the
    investment-rubric rewrite (Section 10) and OR-logic threshold recalibration (Section 13), an earlier version of
    the pipeline achieved an overall Spearman of only <b>roughly +0.11</b> against the same 12 human-expert scores
    &mdash; the redesigned rubric and threshold together raised this to the current <b>+0.589</b>.</div>
    <h3>Per-dimension agreement (weighted &kappa; on binned scores; Spearman on raw 0&ndash;1 scores)</h3>
    <table>
      <thead><tr><th>Dimension</th><th>Weighted &kappa; (95% CI)</th><th>Spearman &rho;</th><th>p-value</th><th>Mean human</th><th>Mean AI</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <h3>Overall ranking (continuous 0&ndash;1 score)</h3>
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>95% CI</th></tr></thead>
      <tbody>
        <tr><td class="pid">ICC(2,1)</td><td>{ov['icc']:.4f}</td><td>[{ov['icc_ci95'][0]:.4f}, {ov['icc_ci95'][1]:.4f}]</td></tr>
        <tr class="row-total"><td class="pid">Spearman &rho;</td><td>{ov['spearman_r']:.4f}</td><td>p={ov['spearman_p']:.4f}</td></tr>
        <tr><td class="pid">Mean bias (AI&minus;human)</td><td>{ov['mean_diff']:+.4f}</td><td>bias p={ov['bias_p']:.4f}</td></tr>
        <tr><td class="pid">Mean human / Mean AI</td><td colspan="2">{ov['mean_human']:.4f} / {ov['mean_ai']:.4f}</td></tr>
      </tbody>
    </table>
    <div class="plain"><b>Statistical caveat:</b> the overall Spearman &rho;=0.589 has an uncorrected p=0.0437, but
    a <b>Holm-adjusted p=0.262</b> across the family of six Spearman tests &mdash; the paper treats this result as
    exploratory, not confirmatory. AI scores are also biased low vs. human scores by 0.136 on average (paired-t
    unadjusted p=0.0015, Holm-adjusted p=0.009 &mdash; this bias direction is more statistically robust than the
    correlation itself).</div>
    <h3>Pairwise rank concordance</h3>
    <p>Of {ro['pairwise_comparable']} human-ordered proposal pairs (excluding {ro['human_tie_pairs_excluded']} tied
    pairs), <b>{ro['pairwise_concordant']} were concordant and {ro['pairwise_discordant']} discordant</b> with the
    AI's ranking &mdash; a pairwise rank concordance of <b>{ro['pairwise_rank_concordance']*100:.1f}%</b>.</p>

    <h4>Discordant pairs (first 10 of {ro['pairwise_discordant']}, raw 0&ndash;1 scores)</h4>
    <table>
      <thead><tr><th>Proposal A</th><th>Proposal B</th><th>Human A</th><th>Human B</th><th>AI A</th><th>AI B</th></tr></thead>
      <tbody>{discordant_rows}</tbody>
    </table>
    <p>In each row, the human expert ranked A below B (or vice versa) while the AI ranked them the opposite way
    &mdash; concrete, citable examples of the 24.1% of pairs where AI and expert priority ordering disagree.</p>

    <h3>Full per-proposal comparison (all 12 proposals, raw 0&ndash;1 scores)</h3>
    <p>Every human and AI score underlying the aggregate statistics above, one row per proposal. Verified to
    reproduce the exact reported means (e.g. team: human 0.7292 / AI 0.6864) to four decimal places.</p>
    {per_item_table}
    <p class="cite">Source: results/Dataset1/evaluation/evaluation_report.xlsx, sheet "per_item" (live). Human
    scores from ExpertsEvaluations.xlsx, normalized 1&ndash;5&rarr;0&ndash;1 and 1&ndash;20&rarr;0&ndash;1. AI
    scores are Stage 8's direct re-scoring of the compiled report on the same human rubric scale (not Stage 6's
    blended investment score). Note: this sheet's own stored per-proposal AI-verdict column does not reproduce the
    83.3% verdict accuracy reported elsewhere in this document (it implies a lower match rate) and is therefore
    omitted here as unverified against the live evaluation_summary.json figures &mdash; only the dimension/overall
    scores, which do verify exactly, are shown.</p>

    <h3>Verdict accuracy: before vs. after OR-logic recalibration</h3>
    <table>
      <thead><tr><th>Rule</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP/FP/FN/TN</th></tr></thead>
      <tbody>
        <tr class="row-old"><td class="pid">Previous: {d1['verdict_rule_prev']}</td>
          <td>{d1['verdict_prev_accuracy']*100:.1f}%</td><td>{d1['verdict_prev_precision']*100:.1f}%</td>
          <td>{d1['verdict_prev_recall']*100:.1f}%</td><td>{d1['verdict_prev_f1']*100:.1f}%</td>
          <td>{d1['verdict_prev_tp']}/{d1['verdict_prev_fp']}/{d1['verdict_prev_fn']}/{d1['verdict_prev_tn']}</td></tr>
        <tr class="row-total"><td class="pid">Current: {d1['verdict_rule']}</td>
          <td>{d1['verdict_accuracy']*100:.1f}%</td><td>{d1['verdict_precision']*100:.1f}%</td>
          <td>{d1['verdict_recall']*100:.1f}%</td><td>{d1['verdict_f1']*100:.1f}%</td>
          <td>{d1['verdict_tp']}/{d1['verdict_fp']}/{d1['verdict_fn']}/{d1['verdict_tn']}</td></tr>
      </tbody>
    </table>

    <h3>Direct raw-document baseline (from the research paper, ablation comparison)</h3>
    <p>A version of the system that scores directly off the raw proposal document (no structured fact-extraction /
    dimension-construction pipeline) achieves Spearman = <b>0.423</b> (lower than the full pipeline's 0.589) but
    slightly <b>higher ICC (0.155 vs 0.119)</b>. Critically, its scores collapse to only <b>3 distinct overall-score
    values</b> across all 12 proposals, correctly distinguishing only 17 of 58 ordered pairs (41 tied) &mdash;
    versus the full pipeline's 44/58. With the same OR-threshold search, the baseline reaches 75.0% tuned verdict
    accuracy (vs. 83.3% for the full pipeline), predicting only one GO proposal.</p>

    <h3>Is the confidence score itself trustworthy? (research paper, Section 7.2)</h3>
    <p>The pipeline emits a separate <code>confidence</code> value (0&ndash;1) alongside every score, ranging
    <b>0.633&ndash;0.818</b> across the 12 proposals. Two checks were run on whether this number means what it
    claims to mean:</p>
    <table>
      <thead><tr><th>Check</th><th>Spearman r<sub>s</sub></th><th>p-value</th><th>Interpretation</th></tr></thead>
      <tbody>
        <tr><td class="pid">Confidence vs. absolute score error</td><td>&minus;0.413</td><td>0.183</td><td style="text-align:left">Negative direction is correct (higher confidence &rarr; smaller error) but not statistically reliable at n=12</td></tr>
        <tr><td class="pid">Confidence vs. verdict correctness</td><td>&minus;0.048</td><td>0.882</td><td style="text-align:left">Essentially no relationship &mdash; confidence does not predict whether the GO/NO-GO verdict matches the human verdict</td></tr>
      </tbody>
    </table>
    <div class="plain">The paper's conclusion, worth quoting directly: confidence should be treated as
    <b>"an empirical decision feature, not a validated probability of correctness."</b> It is a real number the
    pipeline uses operationally (it is one half of the OR-logic verdict gate, Section 13), but it has not been
    shown to reliably track whether the pipeline is actually right.</div>

    <h3>Claim grounding (Table 2 of the research paper)</h3>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td class="pid">Total classified report statements</td><td>188</td></tr>
        <tr><td class="pid">Excluded (professional inference / evaluator judgment)</td><td>114 + 41 = 155</td></tr>
        <tr><td class="pid">Eligible factual/numerical claims</td><td>33</td></tr>
        <tr><td class="pid">Supported / partially supported / unsupported</td><td>25 / 4 / 4</td></tr>
        <tr class="row-total"><td class="pid">Strict groundedness</td><td>75.8% [59.0%, 87.2%]</td></tr>
        <tr class="row-total"><td class="pid">Lenient (half-credit) groundedness</td><td>81.8% [65.6%, 91.4%]</td></tr>
        <tr><td class="pid">Unsupported-claim rate</td><td>12.1% [4.8%, 27.3%]</td></tr>
      </tbody>
    </table>

    <h3>Content-sensitivity results (Table 3 of the research paper; Cohen's h from README)</h3>
    <table>
      <thead><tr><th>Condition</th><th>Passed / Total</th><th>Rate</th><th>Cohen's h</th><th>Effect size</th></tr></thead>
      <tbody>
        <tr><td class="pid">Strategy PLUS</td><td>88/96</td><td>91.7%</td><td>0.99</td><td>Large</td></tr>
        <tr><td class="pid">Strategy MINUS</td><td>49/55</td><td>89.1%</td><td>0.90</td><td>Large</td></tr>
        <tr><td class="pid">Objectives PLUS</td><td>91/95</td><td>95.8%</td><td>1.16</td><td>Large</td></tr>
        <tr><td class="pid">Objectives MINUS</td><td>43/43</td><td>100.0%</td><td>1.57</td><td>Large</td></tr>
        <tr><td class="pid">Advantages PLUS</td><td>64/95</td><td>67.4%</td><td>0.35</td><td>Medium</td></tr>
        <tr><td class="pid">Advantages MINUS</td><td>31/33</td><td>93.9%</td><td>1.07</td><td>Large</td></tr>
        <tr class="row-total"><td class="pid">Overall</td><td>366/417</td><td>87.7% [83.3%, 91.7%]</td><td>0.86</td><td>Large</td></tr>
      </tbody>
    </table>
    <h4>Combined pass rate per dimension (PLUS+MINUS pooled)</h4>
    <table style="max-width:340px">
      <thead><tr><th>Dimension</th><th>Combined rate</th></tr></thead>
      <tbody>
        <tr><td class="pid">Objectives</td><td>134/138 = 97%</td></tr>
        <tr><td class="pid">Strategy</td><td>137/151 = 90%</td></tr>
        <tr><td class="pid">Advantages</td><td>95/128 = 74%</td></tr>
      </tbody>
    </table>
    <p>Advantages-PLUS is the weakest cell (67.4%, medium Cohen's h=0.35) &mdash; the scorer stays skeptical of
    self-reported competitive claims even when injected explicitly. The README additionally names which
    judgment-dependent sub-rubrics drove this variability: <b>Problem-Solution Fit, Market Timing,</b> and
    <b>Competitive Defensibility</b> were the specific sub-rubrics most likely to move inconsistently on MINUS
    removals, versus document-verifiable sub-rubrics (milestone dates, revenue model, regulatory path, market
    size) which tracked injected/removed evidence far more reliably. Overall pass rate is significant vs. a 50%
    baseline (one-sided binomial p&lt;0.001).</p>
    <div class="plain"><b>A resolved historical bug, worth noting for context:</b> an internal diagnosis
    (<code>score_clustering_diagnosis.html</code>, June 2026) found an earlier version of the pipeline producing
    every dimension score clustered in a 0.034-wide band (0.413&ndash;0.447) regardless of proposal content, with
    every alignment sub-score locked at exactly 0.35 and every confidence value at exactly 0.50. Three stacked
    causes were identified: <code>search_hints</code> were never populated by Stage 3 (forcing the alignment
    sub-score to its hardcoded 0.35 fallback), the evidence-count/authority/coverage weights in the Stage 5 scoring
    config were set to 0.0, and every LLM answer's confidence field was returning the 0.50 floor. All three are
    fixed in the current codebase (Section 9's scoring-weight formulas show non-zero evidence weights), and the
    current human-expert results (well-differentiated scores, mean AI dimension scores 0.65&ndash;0.72, Section 24
    above) confirm the fix holds &mdash; but it is a concrete illustration of how easily a scoring pipeline can
    silently produce uniform, non-discriminating output without a diagnostic like this one to catch it.</div>
    <p class="cite">Source: results/Dataset1/evaluation/evaluation_summary.json (live); research_paper.pdf Tables 1-3, Section 7.1; README.md "Results" and "Research &amp; Calibration Tools"; results/Dataset1/sensitivity_report.html; results/Dataset1/score_clustering_diagnosis.html.</p>
    """
    return sec(24, "Results &mdash; Human-Expert Study (Datasets 1 &amp; 2)", body)


def s23_results_kickstarter(ks):
    def group_rows(g):
        rows = ""
        for dim in ("team", "objective", "strategy", "advantages", "feasibility"):
            d = g["dimension_spearman"][dim]
            rows += f"""<tr><td class="dim-name">{dim.capitalize()}</td><td class="{'pos' if d['rho']>=0 else 'neg'}">{d['rho']:+.4f}</td><td>{d['p_value']:.4f}</td><td>{d['holm_p_value']:.4f}</td></tr>"""
        return rows

    def cls_row(g):
        c = g["classification"]
        return f"""<td>{c['accuracy']*100:.1f}%</td><td>{'&mdash;' if c['precision'] is None else f"{c['precision']*100:.1f}%"}</td><td>{c['recall']*100:.1f}%</td><td>{c['tp']}/{c['fp']}/{c['fn']}/{c['tn']}</td>"""

    tor, mon, comb = ks["toronto"], ks["montreal"], ks["combined"]
    body = f"""
    <p>Full current results from <code>results/Kickstarter/evaluation/evaluation_summary.json</code> (generated
    {ks['generated_at']}). Ground truth: {ks['ground_truth']}. Success threshold: {ks['success_threshold']}.
    n_total={ks['n_total_dataset']}, n_evaluated={ks['n_evaluated']}.</p>

    <h3>Overall Spearman correlation (AI score vs. real funding %)</h3>
    <table>
      <thead><tr><th>Group</th><th>n</th><th>Spearman &rho;</th><th>p-value</th><th>Holm-adjusted p</th></tr></thead>
      <tbody>
        <tr><td class="pid">Toronto</td><td>{tor['n_proposals']}</td><td>{tor['overall_spearman']['rho']:+.4f}</td><td>{tor['overall_spearman']['p_value']:.4f}</td><td>{tor['overall_spearman']['holm_p_value']:.4f}</td></tr>
        <tr><td class="pid">Montreal</td><td>{mon['n_proposals']}</td><td>{mon['overall_spearman']['rho']:+.4f}</td><td>{mon['overall_spearman']['p_value']:.4f}</td><td>{mon['overall_spearman']['holm_p_value']:.4f}</td></tr>
        <tr class="row-total"><td class="pid">Combined</td><td>{comb['n_proposals']}</td><td>{comb['overall_spearman']['rho']:+.4f}</td><td>{comb['overall_spearman']['p_value']:.4f}</td><td>{comb['overall_spearman']['holm_p_value']:.4f}</td></tr>
      </tbody>
    </table>

    <h3>Per-dimension Spearman (combined, n=141)</h3>
    <table>
      <thead><tr><th>Dimension</th><th>Spearman &rho;</th><th>p-value</th><th>Holm-adjusted p</th></tr></thead>
      <tbody>{group_rows(comb)}</tbody>
    </table>

    <h3>Classification (AI GO verdict vs. actual funding success)</h3>
    <table>
      <thead><tr><th>Group</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>TP/FP/FN/TN</th></tr></thead>
      <tbody>
        <tr><td class="pid">Toronto</td>{cls_row(tor)}</tr>
        <tr><td class="pid">Montreal</td>{cls_row(mon)}</tr>
        <tr class="row-total"><td class="pid">Combined</td>{cls_row(comb)}</tr>
      </tbody>
    </table>
    <div class="plain">At the production threshold (score &ge; 0.725 OR confidence &ge; 0.811), the pipeline never
    predicted GO for any of the 141 Kickstarter proposals &mdash; TP=FP=0 in every group. The reported "accuracy"
    figures (82&ndash;86%) are therefore driven entirely by the base rate of failed campaigns, not by genuine
    discrimination. This is the finding that motivated the threshold-calibration study (Section 27).</div>
    <p class="cite">Source: results/Kickstarter/evaluation/evaluation_summary.json (live).</p>
    """
    return sec(25, "Results &mdash; Kickstarter Correlation Study", body)


def s23b_kickstarter_highlights(cal):
    pp = cal["per_proposal"]
    top5 = sorted(pp, key=lambda r: -r["score"])[:5]
    bottom5 = sorted(pp, key=lambda r: r["score"])[:5]
    funded = [r["score"] for r in pp if r["success"] == 1]
    unfunded = [r["score"] for r in pp if r["success"] == 0]
    mean_funded = sum(funded) / len(funded)
    mean_unfunded = sum(unfunded) / len(unfunded)

    def rows(items):
        out = ""
        for r in items:
            tag = '<span class="pos">funded</span>' if r["success"] == 1 else '<span class="neg">not funded</span>'
            out += f"""<tr><td class="pid">{r['pid']}</td><td>{r['city']}</td><td>{r['score']:.3f}</td>
            <td>{r['confidence']:.3f}</td><td>{r['funded_pct']:.0f}%</td><td>{tag}</td></tr>"""
        return out

    body = f"""
    <p>Individual-proposal detail from <code>results/KickstarterCalibrated/evaluation/calibration_results.json</code>
    (per-proposal array, {len(pp)} records) &mdash; concrete examples illustrating the correlation-study results
    above.</p>
    <h3>Highest AI-scored proposals</h3>
    <table>
      <thead><tr><th>PID</th><th>City</th><th>AI score</th><th>Confidence</th><th>Actual funded</th><th>Outcome</th></tr></thead>
      <tbody>{rows(top5)}</tbody>
    </table>
    <h3>Lowest AI-scored proposals</h3>
    <table>
      <thead><tr><th>PID</th><th>City</th><th>AI score</th><th>Confidence</th><th>Actual funded</th><th>Outcome</th></tr></thead>
      <tbody>{rows(bottom5)}</tbody>
    </table>
    <div class="plain"><b>The single highest AI score in the entire 141-proposal dataset is kt49 at 0.640</b> &mdash;
    yet that campaign only raised 4% of its goal, a real failure. Meanwhile <b>kt10 scored almost identically
    (0.639)</b> and was actually funded at 151%. Near-identical AI scores, opposite real outcomes &mdash; a direct
    illustration of the ceiling on what proposal text alone can predict, independent of any threshold-calibration
    question. Averaging across all 141 proposals: mean AI score for the 24 proposals that were <b>actually
    funded is {mean_funded:.3f}</b>, versus <b>{mean_unfunded:.3f}</b> for the 117 that were not &mdash; a
    consistent, directionally correct gap of {mean_funded-mean_unfunded:+.3f}, even though individual cases like
    kt49/kt10 show it is far from a reliable case-by-case signal.</div>
    <p class="cite">Source: results/KickstarterCalibrated/evaluation/calibration_results.json (live, per_proposal array).</p>
    """
    return sec(26, "Kickstarter Highlight Examples", body)


def s24_results_calibration(cal):
    before, is_, cv = cal["before_calibration"], cal["in_sample_calibration"], cal["cross_validation"]["aggregated_held_out"]
    fold_rows = "".join(
        f"""<tr><td class="pid">Fold {f['fold']}</td><td>{f['n_test']}</td><td>{f['n_test_success']}</td>
        <td>{f['tau_r']}</td><td>{f['tau_c']}</td><td>{f['train_f1']}</td><td>{f['f1'] if f['f1'] is not None else '&mdash;'}</td>
        <td>{f'{f["precision"]*100:.0f}%' if f['precision'] is not None else '&mdash;'}</td>
        <td>{f'{f["recall"]*100:.0f}%' if f['recall'] is not None else '&mdash;'}</td></tr>"""
        for f in cal["cross_validation"]["per_fold"]
    )
    body = f"""
    <p>Full current results from <code>results/KickstarterCalibrated/evaluation/calibration_results.json</code>
    (generated {cal['generated_at']}). n_total={cal['n_total']}, n_success={cal['n_success']}
    ({cal['success_rate']*100:.1f}%), 5-fold stratified CV, seed={cal['cross_validation']['seed']}.</p>
    <p><b>Threshold-free discrimination of the raw production score:</b> ROC-AUC=<b>{cal['discrimination']['roc_auc']:.4f}</b>,
    Average Precision=<b>{cal['discrimination']['average_precision']:.4f}</b> (vs. a {cal['success_rate']*100:.1f}%
    random baseline).</p>
    <table>
      <thead><tr><th>Scenario</th><th>&tau;<sub>r</sub></th><th>&tau;<sub>c</sub></th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>
        <tr class="row-old"><td class="pid">Before (production, transferred)</td><td>{before['tau_r']}</td><td>{before['tau_c']}</td><td>{before['tp']}</td><td>{before['fp']}</td><td>{before['fn']}</td><td>{before['tn']}</td><td>{before['accuracy']*100:.1f}%</td><td>&mdash;</td><td>{before['recall']*100:.1f}%</td><td>&mdash;</td></tr>
        <tr><td class="pid">In-sample (overfitting reference)</td><td>{is_['tau_r']}</td><td>{is_['tau_c']}</td><td>{is_['tp']}</td><td>{is_['fp']}</td><td>{is_['fn']}</td><td>{is_['tn']}</td><td>{is_['accuracy']*100:.1f}%</td><td>{is_['precision']*100:.1f}%</td><td>{is_['recall']*100:.1f}%</td><td>{is_['f1']}</td></tr>
        <tr class="row-total"><td class="pid">Cross-validated (honest, held-out)</td><td>varies/fold</td><td>varies/fold</td><td>{cv['tp']}</td><td>{cv['fp']}</td><td>{cv['fn']}</td><td>{cv['tn']}</td><td>{cv['accuracy']*100:.1f}%</td><td>{cv['precision']*100:.1f}%</td><td>{cv['recall']*100:.1f}%</td><td>{cv['f1']}</td></tr>
      </tbody>
    </table>
    <h3>Per-fold detail</h3>
    <table>
      <thead><tr><th>Fold</th><th>Test n</th><th>Test successes</th><th>&tau;<sub>r</sub></th><th>&tau;<sub>c</sub></th><th>Train F1</th><th>Held-out F1</th><th>Held-out precision</th><th>Held-out recall</th></tr></thead>
      <tbody>{fold_rows}</tbody>
    </table>
    <div class="plain">Recalibrating the threshold to the Kickstarter score distribution &mdash; while validating
    honestly out-of-sample &mdash; lifts recall on real successes from <b>0% to {cv['recall']*100:.1f}%</b>, and
    precision from undefined to <b>{cv['precision']*100:.1f}%</b> ({cv['precision']/0.17:.2f}&times; the 17.0% base
    rate). The score threshold &tau;<sub>r</sub> stays in a tight 0.55&ndash;0.61 band across all 5 folds, an
    encouraging sign of stability. The in-sample vs. held-out gap (F1 {is_['f1']} vs {cv['f1']}) is a direct,
    measured illustration of overfitting risk.</div>
    <p class="cite">Source: results/KickstarterCalibrated/evaluation/calibration_results.json (live).</p>
    """
    return sec(27, "Results &mdash; Threshold Calibration Study", body)


def s25_results_improve(imp, rows_for_chart):
    a, b, c = imp["method_a_overall_score"], imp["method_b_simple_composite"], imp["method_c_logistic_regression"]

    def row(label, res, muted=False):
        cv = res["cv_aggregated_held_out"]
        cls = ' class="row-old"' if muted else ""
        return f"""<tr{cls}><td class="pid">{label}</td><td>{res['roc_auc']}</td><td>{res['average_precision']}</td>
        <td>{cv['precision']*100:.1f}%</td><td>{cv['recall']*100:.1f}%</td><td>{cv['f1']}</td></tr>"""

    coef = c["full_dataset_fit_coefficients_for_interpretation_only"]["coef"]
    coef_rows = "".join(f'<tr><td class="pid">{d.capitalize()}</td><td class="st-num">{v:+.4f}</td></tr>' for d, v in sorted(coef.items(), key=lambda kv: -kv[1]))
    body = f"""
    <p>Full current results from <code>results/KickstarterCalibrated/evaluation/score_improvement_results.json</code>
    (generated {imp['generated_at']}). All three methods use a single score threshold (no confidence OR-gate),
    5-fold CV, seed={imp['seed']}, L2 regularization C={imp['l2_C']}.</p>
    <table>
      <thead><tr><th>Method</th><th>ROC-AUC</th><th>Avg. Precision</th><th>CV Precision</th><th>CV Recall</th><th>CV F1</th></tr></thead>
      <tbody>
        {row("A: overall_score (production, 5 dims)", a)}
        {row("B: simple composite (feasibility+strategy+advantages)/3 &mdash; naive", b, muted=True)}
        {row("C: logistic regression (5 dims, CV-fit)", c)}
      </tbody>
    </table>
    <p><b>Note on Method B:</b> this "naive" result (ROC-AUC {b['roc_auc']}) used dimension selection derived from
    the full 141-proposal dataset. Section 29 shows this is optimistic; the fully-nested, honest version scores
    lower (ROC-AUC 0.7731). See Section 29 for the corrected number and why the gap exists.</p>
    <h3>Logistic regression coefficients (fit on full dataset, interpretation only &mdash; not used for the CV metrics above)</h3>
    <table style="max-width:340px"><thead><tr><th>Dimension</th><th>Learned weight</th></tr></thead><tbody>{coef_rows}</tbody></table>
    <p>The learned weights lean toward feasibility and strategy but do not sharply zero out team/objective &mdash;
    L2 regularization keeps all five weights in a similar modest range given only ~19 positive training examples
    per fold, which is why the learned model underperforms the hand-selected composite (Section 29 explains this
    in more detail).</p>
    <p class="cite">Source: results/KickstarterCalibrated/evaluation/score_improvement_results.json (live).</p>
    """
    return sec(28, "Results &mdash; Score Improvement Study (Methods A/B/C)", body)


def s26_results_robust(rob, imp):
    boot = rob["bootstrap"]
    nb = rob["nested_method_b"]
    ds = rob["dimension_stability"]

    def boot_row(key, label):
        v = boot[key]
        sig = "yes" if v["significant_at_0.05"] else "no"
        return f"""<tr><td class="pid">{label}</td><td>{v['mean_diff']:+.4f}</td><td>[{v['ci_95_lower']:+.4f}, {v['ci_95_upper']:+.4f}]</td><td>{v['prob_method_2_better']*100:.1f}%</td><td>{sig}</td></tr>"""

    fold_rows = "".join(
        f"""<tr><td class="pid">Fold {f['fold']}</td><td>{', '.join(f['top3_selected'])}</td><td>{'yes' if f['matches_fixed_selection'] else 'no'}</td></tr>"""
        for f in ds["per_fold"]
    )
    a_score = imp["method_a_overall_score"]["roc_auc"]
    body = f"""
    <p>Full current results from <code>results/KickstarterCalibrated/evaluation/score_robustness_check.json</code>
    (generated {rob['generated_at']}), {boot['n_boot']} bootstrap iterations per comparison.</p>
    <h3>Bootstrap 95% confidence intervals on ROC-AUC differences</h3>
    <table>
      <thead><tr><th>Comparison</th><th>Mean diff</th><th>95% CI</th><th>P(2nd method better)</th><th>Significant?</th></tr></thead>
      <tbody>
        {boot_row('B_vs_A', 'B (naive) vs. A')}
        {boot_row('B_vs_C', 'B (naive) vs. C')}
        {boot_row('C_vs_A', 'C vs. A')}
      </tbody>
    </table>
    <div class="plain">None of the three comparisons reach conventional statistical significance at n=141/24
    positives &mdash; every 95% CI includes zero, narrowly. Method B looks better than A/C in ~95% of bootstrap
    resamples, suggestive of a real effect but short of the standard bar for "proven."</div>

    <h3>Per-fold dimension-selection stability</h3>
    <table>
      <thead><tr><th>Fold</th><th>Top-3 dimensions (train-fold correlation only)</th><th>Matches feasibility/strategy/advantages?</th></tr></thead>
      <tbody>{fold_rows}</tbody>
    </table>
    <p><b>{ds['n_folds_matching_fixed_selection']} of {ds['k_folds']}</b> folds independently selected the exact
    same three dimensions using only their own training data (fold 4 substituted "team" for "advantages").</p>

    <h3>Fully-nested Method B &mdash; the honest number</h3>
    <table>
      <thead><tr><th>Metric</th><th>Naive (Section 28)</th><th>Fully nested (honest)</th></tr></thead>
      <tbody>
        <tr><td class="pid">ROC-AUC</td><td>0.8048</td><td class="row-total">{nb['roc_auc']}</td></tr>
        <tr><td class="pid">CV Precision</td><td>35.4%</td><td class="row-total">{nb['cv_aggregated_held_out']['precision']*100:.1f}%</td></tr>
        <tr><td class="pid">CV Recall</td><td>70.8%</td><td class="row-total">{nb['cv_aggregated_held_out']['recall']*100:.1f}%</td></tr>
        <tr><td class="pid">CV F1</td><td>0.4722</td><td class="row-total">{nb['cv_aggregated_held_out']['f1']}</td></tr>
      </tbody>
    </table>
    <div class="plain"><b>The naive 0.805 was inflated.</b> When dimension selection and threshold are both fit
    strictly within each training fold (no part of the number touches the full 141-proposal dataset), the honest
    ROC-AUC is <b>{nb['roc_auc']}</b> &mdash; still directionally better than the production score
    ({a_score}), but a much smaller edge than first reported. Combined with the bootstrap result above, the correct
    characterization is: <i>"a promising, directionally consistent, but not-yet-statistically-confirmed
    improvement."</i></div>
    <p class="cite">Source: results/KickstarterCalibrated/evaluation/score_robustness_check.json (live).</p>
    """
    return sec(29, "Results &mdash; Robustness Checks (Bootstrap &amp; Nested Validation)", body)


def s29b_audit_matrix():
    rows = [
        ("Expert alignment", "Do scores reproduce expert judgments?", "System measures a different construct", "Kappa, ICC, Spearman", "Done"),
        ("Rank preservation", "Does the system preserve expert ordering?", "Screening priorities reversed", "Pairwise concordance", "Done"),
        ("Calibration", "Are score and verdict mappings aligned?", "Raw scores compressed or threshold-misaligned", "Bias, accuracy, precision/recall", "Done"),
        ("Real-outcome alignment", "Does the score track real-world results, not just expert opinion?", "Score is only valid within the expert-judgment frame", "Spearman vs. funded %, ROC-AUC", "Done (this project)"),
        ("Threshold generalization", "Does a calibrated decision threshold transfer across domains?", "In-sample threshold overfits, fails out-of-domain", "Cross-validated recall/precision", "Done (this project)"),
        ("Pipeline attribution", "Does structured evidence improve over raw scoring?", "Architecture adds complexity without measurement gain", "Direct baseline, ablations", "Pilot"),
        ("Content sensitivity", "Do substantive additions/removals move intended scores?", "Scores ignore relevant evidence or only respond to wording", "Prompt-coupled directional pass rate", "Pilot"),
        ("Groundedness", "Are factual premises supported by proposal evidence?", "Reports use hallucinated evidence", "Corrected groundedness, unsupported-claim rate", "Pilot"),
        ("Repeatability", "Are identical runs stable?", "Recommendations alter with sampling noise", "Score variance, verdict stability", "Future"),
        ("Style robustness", "Does presentation alter scores without new evidence?", "System rewards polish over substance", "Score delta", "Future"),
        ("Halo/spillover", "Are cross-dimension changes rubric-justified?", "Salient traits contaminate unrelated scores", "Target/non-target movement", "Future"),
        ("Provider robustness", "Do findings survive model changes?", "Validity is provider-specific", "Cross-provider agreement", "Future"),
    ]
    status_cls = {"Done": "pos", "Done (this project)": "pos", "Pilot": "", "Future": "neg"}
    rows_html = "".join(
        f"""<tr><td class="pid">{a}</td><td style="text-align:left">{q}</td><td style="text-align:left">{f}</td><td>{m}</td>
        <td class="{status_cls.get(s,'')}">{s}</td></tr>"""
        for a, q, f, m, s in rows
    )
    body = f"""
    <p>The project's research paper organizes its validation program as a matrix of independent audits, each
    targeting a specific failure mode. This session's Kickstarter work added two rows (marked "Done (this
    project)") to the paper's original matrix &mdash; a concrete illustration of the audit framework being
    extended rather than needing to be rebuilt.</p>
    <table>
      <thead><tr><th>Audit</th><th>Question</th><th>Failure mode it catches</th><th>Primary metric</th><th>Status</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p>Four audits remain <b>Future</b> work in both the original paper and after this project's extensions:
    repeatability (Stage 6 runs at non-zero temperature, never formally tested for run-to-run stability),
    style robustness, cross-dimension halo/spillover effects, and cross-provider agreement (this project's results
    all use OpenAI <code>gpt-4o-mini</code> exclusively, despite the pipeline supporting Gemini and DeepSeek).</p>
    <p class="cite">Source: research_paper.pdf Table 4 (Outcome-free audit matrix), extended with this session's
    Kickstarter work.</p>
    """
    return sec(30, "Outcome-Free Audit Coverage Matrix", body)


# ── Part V: Synthesis ────────────────────────────────────────────────────────

def s27_cross_dataset():
    body = """
    <p>The two evaluation programs in this project &mdash; human-expert comparison on 12 biotech/hardware
    proposals, and real-outcome comparison on 141 Kickstarter campaigns &mdash; used the exact same unmodified
    pipeline and rubric, which makes their differences directly informative about what generalizes and what
    doesn't.</p>
    <table>
      <thead><tr><th>Aspect</th><th>Biotech/semiconductor (n=12, vs. experts)</th><th>Kickstarter (n=141, vs. real outcomes)</th></tr></thead>
      <tbody>
        <tr><td class="pid">Ground truth</td><td style="text-align:left">Human expert 1-5/1-20 scores, subjective</td><td style="text-align:left">Real funding %, objective, all-or-nothing</td></tr>
        <tr><td class="pid">Overall ranking correlation</td><td>Spearman 0.589 (Holm p=0.262, not significant)</td><td>Spearman 0.366 (Holm p&lt;0.001, significant)</td></tr>
        <tr><td class="pid">Strongest predictive dimension</td><td style="text-align:left">Team (&kappa;=0.324, &rho;=0.398)</td><td style="text-align:left">Feasibility (&rho;=0.344, p&lt;0.001)</td></tr>
        <tr><td class="pid">Weakest / non-significant dimensions</td><td style="text-align:left">Objective, Strategy, Advantages (negative &kappa;)</td><td style="text-align:left">Team, Objective (not significant)</td></tr>
        <tr><td class="pid">Verdict rule performance</td><td style="text-align:left">83.3% accuracy (calibrated in-sample on this data)</td><td style="text-align:left">0% recall (threshold never recalibrated for this domain)</td></tr>
        <tr><td class="pid">Score ceiling observed</td><td style="text-align:left">Mean AI overall &asymp;0.68</td><td style="text-align:left">Max AI overall = 0.640 (never reaches 0.725 GO cutoff)</td></tr>
      </tbody>
    </table>
    <h3>The central cross-dataset finding: dimension importance is domain-dependent</h3>
    <p><b>Team dominates when experts judge biotech/hardware proposals</b> (highest kappa of any dimension), but
    <b>Team carries no significant signal when predicting real Kickstarter funding</b> (&rho;=0.160, not
    significant). The reverse holds for Feasibility, which is only weakly related to expert judgment
    (&rho;=0.156) but is the single strongest predictor of real crowdfunding success. This is a coherent, explicable
    pattern: VC/biotech experts weigh founder credentials heavily because execution risk is what kills those
    ventures; Kickstarter backers cannot verify founder credentials and instead respond to whether the product
    itself looks buildable and deliverable.</p>
    <h3>The central cross-dataset finding: thresholds do not transfer across domains</h3>
    <p>The GO/HOLD/NO-GO threshold (score &ge; 0.725 OR confidence &ge; 0.811) was derived entirely from the
    12-proposal biotech dataset, in-sample. Applied unmodified to Kickstarter, it never fires GO once across 141
    proposals &mdash; not because the underlying score carries no signal (ROC-AUC 0.764, a real and independently
    confirmed effect), but because the rubric's score ceiling for this domain (max observed: 0.640) sits
    structurally below the transferred cutoff. Recalibrating the same OR-logic rule specifically for Kickstarter,
    validated honestly via cross-validation, restores substantial recall (58.3%) &mdash; demonstrating this is a
    threshold-transfer problem, correctable, not evidence the pipeline is fundamentally broken outside its original
    domain.</p>
    """
    return sec(31, "Cross-Dataset Comparison", body)


def s28_findings():
    body = """
    <div class="findings-list"><ul>
      <li><b>The rubric enforces a deliberate low score ceiling.</b> The investment-scoring prompt explicitly
      instructs "most proposals should land in 0.40-0.70; scores above 0.80 require strong, specific evidence" and
      caps AI-as-product innovation at 0.42-0.56 "NO EXCEPTIONS." Any discussion of why absolute scores run low
      (biotech mean &asymp;0.68, Kickstarter max 0.640) should cite this design choice directly rather than treat it
      as an unexplained artifact.</li>
      <li><b>Cross-domain generalization without retuning:</b> a pipeline built and calibrated entirely on
      biotech/hardware VC proposals produces a statistically significant, positive correlation (Spearman 0.366,
      Holm p&lt;0.001, n=141) with real consumer-crowdfunding outcomes it was never tuned on, and threshold-free
      ROC-AUC of 0.764 &mdash; independent evidence the score itself carries real information in an out-of-domain
      setting.</li>
      <li><b>Dimension importance flips across domains:</b> Team dominates the expert-judged biotech dataset but is
      one of the weakest, non-significant predictors of real Kickstarter outcomes; Feasibility shows the reverse
      pattern. A clean, quotable contrast between "what experts value" and "what crowds actually fund."</li>
      <li><b>A threshold-transfer failure, precisely diagnosed and fixed:</b> the in-sample-calibrated biotech
      threshold produces 0% recall on Kickstarter; recalibrating the identical decision-rule shape via proper
      cross-validation restores 58.3% recall &mdash; a controlled before/after demonstration of covariate shift in
      an LLM-based scoring rubric, and a methodological improvement over the biotech threshold's own in-sample-only
      calibration.</li>
      <li><b>An honest overfitting demonstration, produced in this project:</b> in-sample threshold calibration
      (F1=0.478) vs. properly held-out cross-validation (F1=0.364) on the same data is a small, self-contained,
      reproducible illustration of exactly the failure mode outcome-free validation research should guard against
      &mdash; arguably including the existing biotech threshold's own un-held-out calibration.</li>
      <li><b>A quiet leak, caught by nesting the whole pipeline:</b> a "zero-fitting" composite score that simply
      dropped two known-weak dimensions still leaked information through its dimension-<i>selection</i> step (which
      used the full dataset); fully nesting that selection inside cross-validation dropped ROC-AUC from an
      overstated 0.805 to an honest 0.773 &mdash; illustrating that leakage doesn't require model fitting to sneak
      in.</li>
      <li><b>A modest, statistically-honest improvement, reported at its true confidence level:</b> the fully-nested
      composite score beats the production score on point estimate (0.773 vs 0.764) with ~95% bootstrap probability
      of being a real effect, but does not clear a 95% significance threshold at this sample size &mdash;
      deliberately reported as "promising, not proven" rather than rounded up.</li>
      <li><b>Small positive-class sizes make single train/test splits statistically unreliable:</b> with only 24
      Kickstarter successes among 141 proposals, stratified 5-fold cross-validation was necessary to obtain a
      stable estimate at all &mdash; directly relevant methodological guidance for any future outcome-based
      validation of this system on similarly rare-success datasets.</li>
      <li><b>A real-world validation channel that scales for free:</b> unlike the 12-proposal expert study (bounded
      by scarce human expert time, and explicitly not statistically well-powered at n=12), the Kickstarter
      real-outcome channel is free, objective, and scales to any sample size &mdash; a methodological point worth
      making about how LLM-based screening instruments can be validated more cheaply going forward.</li>
    </ul></div>
    """
    return sec(32, "Key Findings for the Paper", body)


def s29_limitations():
    body = """
    <div class="outcome-box outcome-bad">
      <h4>Limitations carried over from the existing research paper (12-proposal study)</h4>
      <ul>
        <li>n=12 is statistically underpowered (needs |&rho;|&asymp;0.73 for 80% power); the overall Spearman
        result does not survive Holm correction across the six-test family.</li>
        <li>No second human rater exists &mdash; the expert file is "an operational reference, not universal
        ground truth," and no human-vs-human agreement ceiling has been established.</li>
        <li>AI scores are biased low vs. human scores by 0.136 on average, a statistically robust finding (Holm
        p=0.009).</li>
        <li>Objective and Advantages dimensions show negative or near-zero kappa/correlation with expert judgment
        &mdash; unresolved reliability gaps.</li>
        <li>QA-only scoring (&lambda;=0) actually reverses expert ranking (Spearman &minus;0.325) &mdash; a
        component of the pipeline that would fail badly used in isolation.</li>
        <li>The content-sensitivity check is explicitly prompt-coupled, not an independent test of real-world
        evidentiary responsiveness.</li>
        <li>The groundedness audit (33 eligible claims) is a pilot, not a validated hallucination benchmark;
        labels are model-assisted, not independently double-annotated by humans.</li>
        <li>Stage 6's dimension-commentary call runs at temperature 0.25 (non-zero) &mdash; documented but
        unquantified run-to-run non-determinism; repeatability is listed as future work, not yet tested.</li>
        <li>Fairness risk (undemonstrated but flagged): team scoring could leak demographic/gender-coded signals
        from names/affiliations into execution-capacity judgments; no masking or audit currently implemented.</li>
      </ul>
    </div>
    <div class="outcome-box outcome-bad">
      <h4>Limitations specific to the Kickstarter study (this session's work)</h4>
      <ul>
        <li>Precision after calibration remains modest (26-33% depending on method) &mdash; roughly 2-3 in 4
        flagged campaigns still fail; not strong enough to automate a decision, only to triage.</li>
        <li>Composite-score improvement (Sections 28-29) does not reach conventional statistical significance;
        should be reported as directionally suggestive, not confirmed, until validated on a larger sample.</li>
        <li>Fold-level metrics are noisy (F1 ranges 0.0-0.47 across folds) because each held-out fold contains only
        ~5 real successes &mdash; wide implicit confidence intervals on any single-fold number.</li>
        <li>The recalibrated Kickstarter threshold and composite score are specific to Kickstarter's score
        distribution; neither should be assumed to transfer to a third domain without its own recalibration and
        validation, by the same logic that motivated this entire study.</li>
        <li>Kickstarter outcomes are driven substantially by factors outside the proposal document itself (video
        production, existing social following, campaign-period marketing, reward-tier pricing psychology, timing)
        &mdash; a text-only scoring system has a structural ceiling on achievable precision/recall regardless of
        calibration quality.</li>
      </ul>
    </div>
    <div class="plain">
      <b>Deployment/governance recommendations (from the research paper's Discussion section):</b> minimize
      retained proposal text beyond what's needed for audit; restrict access to intermediate pipeline artifacts
      (raw facts, dimension evidence, candidate answers); log the exact prompt and model version behind every
      recommendation; support expert override at every stage; treat dimension-specific reliability as a routing
      signal (route Objective/Advantages-heavy proposals to closer human review, given their weaker measured
      agreement); and mask or perturb demographic-adjacent cues (founder names, affiliations, nationalities) in
      Team-dimension scoring, logging separately when a judgment depends on affiliation rather than documented
      execution evidence. None of these are currently implemented; they are stated design targets for any
      operational deployment of this system.
    </div>
    """
    return sec(33, "Limitations", body)


def appendix_glossary():
    terms = [
        ("Spearman correlation (&rho;)", "Rank-based correlation: do higher-ranked items by one measure tend to be higher-ranked by another? +1 = perfect agreement, 0 = none, &minus;1 = perfectly reversed."),
        ("Cohen's weighted &kappa;", "Agreement between two raters beyond what chance alone would produce, on binned/categorical scores; 'weighted' penalizes larger disagreements more than adjacent-bin disagreements."),
        ("ICC (Intraclass Correlation)", "Measures absolute-value agreement between raters on a continuous scale (not just rank agreement) &mdash; sensitive to systematic scale offsets that Spearman would miss."),
        ("Holm correction", "A family-wise multiple-testing correction, applied when running several hypothesis tests at once, more powerful than Bonferroni but still conservative; a result 'surviving Holm correction' is safer to trust than an uncorrected p-value."),
        ("Pairwise rank concordance", "Of all pairs of items with a definite reference ordering, what fraction does the system order the same way? Directly reflects whether relative priority/ranking is preserved."),
        ("ROC-AUC", "The probability a randomly chosen positive example scores higher than a randomly chosen negative example; 0.5 = random, 1.0 = perfect; independent of any specific decision threshold."),
        ("Average Precision", "Area under the precision-recall curve; like ROC-AUC but more sensitive to performance on the rare (positive) class, useful under class imbalance."),
        ("Precision / Recall / F1", "Precision: of items flagged positive, what fraction truly are? Recall: of items that are truly positive, what fraction got flagged? F1: harmonic mean of the two, useful when class imbalance makes plain accuracy misleading."),
        ("Base rate", "How often the positive outcome occurs with no model involved at all &mdash; the only fair comparison point for judging whether precision/recall represents real signal or not."),
        ("K-fold cross-validation", "Splits data into k folds; repeats train-on-(k-1)/test-on-1 k times, rotating which fold is held out, so every data point is evaluated exactly once by a model/threshold that never saw it during fitting."),
        ("Stratified", "When splitting into folds, ensuring each fold has a proportional share of the rare (positive) class, so no fold ends up with too few positive examples to evaluate reliably."),
        ("In-sample vs. out-of-sample / held-out", "In-sample: fit and evaluated on the same data (optimistic, not generalizable). Out-of-sample / held-out: evaluated on data never used during fitting (the honest estimate)."),
        ("Overfitting", "When a model or threshold is tuned so precisely to one dataset that it fails to generalize to new, unseen data of the same type."),
        ("Bootstrap confidence interval", "Resampling the dataset with replacement many times and recomputing a statistic each time, to estimate how much that statistic would vary by chance &mdash; used here to test whether an observed improvement could just be sampling noise."),
        ("L2 regularization", "A penalty added when fitting a model that discourages large coefficient values, reducing overfitting risk, especially valuable with few training examples relative to the number of features."),
        ("OR-logic threshold rule", "A decision rule of the form 'predict positive if metric A clears its threshold OR metric B clears its threshold' &mdash; either condition alone is sufficient to trigger the positive prediction."),
        ("Groundedness", "The fraction of a report's checkable factual claims that are actually supported by the source proposal document, as opposed to invented or unsupported."),
        ("All-or-nothing funding", "Kickstarter's rule that campaigns reaching less than 100% of their goal collect $0 &mdash; makes 'funded % &ge; 100' an unambiguous binary success label."),
    ]
    items = "".join(f'<div class="gloss-item"><div class="gloss-term">{t}</div><div class="gloss-def">{d}</div></div>' for t, d in terms)
    body = f'<div class="glossary"><div class="gloss-grid">{items}</div></div>'
    return sec("A", "Glossary", body)


def appendix_scripts():
    scripts = [
        ("run_multi_provider_lifecycle.py", "src/tools/", "Main orchestrator: runs all 9 stages per provider, in independent threads, for inter-rater comparison."),
        ("prepare_proposal_text.py", "src/tools/", "Stage 0: document text extraction, OCR fallback, vision-model chart/table description."),
        ("extract_facts_by_chunk.py", "src/tools/", "Stage 1: atomic, source-linked fact extraction from chunked proposal text."),
        ("build_dimensions_from_facts.py", "src/tools/", "Stage 2: buckets facts into 5 dimensions, synthesizes grounded evidence packets."),
        ("generate_questions.py", "src/tools/", "Stage 3: dimension-anchored, fact-linked evaluation question generation."),
        ("llm_answering.py", "src/tools/", "Stage 4: answers every question using only proposal facts, across all providers."),
        ("post_processing.py", "src/tools/", "Stage 5: per-answer quality scoring, best-answer selection, dimension/overall QA scores."),
        ("ai_expert_opinion.py", "src/tools/", "Stage 6: expert commentary + investment-rubric scoring (INVEST_SCORE_SYSTEM) + verdict_rule()."),
        ("generate_final_report.py", "src/tools/", "Stage 7: compiles the final Markdown report (executive summary, review, Q&amp;A evidence)."),
        ("generate_llm_scores.py", "src/tools/", "Stage 8: re-scores the compiled report on the human 1-5/1-20 rubric scale."),
        ("evaluate_cohens_kappa.py", "src/tools/", "Computes weighted kappa, ICC, Spearman, bias, and bootstrap CIs vs. human expert scores."),
        ("evaluate_groundedness.py", "src/tools/", "Classifies report claims and computes strict/lenient groundedness rates."),
        ("evaluate_sweep.py", "src/tools/", "Reruns Stage 8 scoring N times to measure self-consistency / non-determinism."),
        ("evaluate_sensitivity.py / run_sensitivity_all_dimensions.py", "src/tools/, root", "PLUS/MINUS controlled content-injection sensitivity testing."),
        ("calibration_sweep.py", "root", "Zero-LLM-call grid search over INVEST_WEIGHT and verdict thresholds, using cached scores."),
        ("kickstarter_dataset.py", "src/tools/", "Pid&harr;file mapping and funded(%) ground-truth loader for the Kickstarter dataset."),
        ("evaluate_kickstarter_correlation.py", "root", "AI score vs. real Kickstarter funding correlation + classification, per city and combined."),
        ("calibrate_kickstarter_threshold.py", "root", "5-fold cross-validated GO-threshold recalibration for Kickstarter (this project's work)."),
        ("improve_kickstarter_score.py", "root", "Tests 3 composite-score variants (production / simple / logistic regression) via 5-fold CV."),
        ("kickstarter_score_robustness_check.py", "root", "Bootstrap CIs + nested per-fold dimension-selection validation of the score improvement result."),
        ("generate_kickstarter_report.py", "root", "Builds the Kickstarter correlation-study HTML/PDF report."),
        ("generate_kickstarter_calibration_report.py", "root", "Builds the threshold-calibration + score-improvement HTML/PDF report."),
        ("generate_full_project_report.py", "root", "Builds this document."),
    ]
    rows = "".join(f'<tr><td class="pid">{n}</td><td>{loc}</td><td style="text-align:left">{d}</td></tr>' for n, loc, d in scripts)
    body = f"""<table class="script-table"><thead><tr><th>Script</th><th>Location</th><th>Purpose</th></tr></thead><tbody>{rows}</tbody></table>"""
    return sec("B", "Script Inventory", body)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_html", default="results/full_project_report.html")
    ap.add_argument("--out_pdf", default="results/full_project_report.pdf")
    args = ap.parse_args()

    d1 = json.loads(DATASET1_PATH.read_text(encoding="utf-8"))
    d1_sweep = json.loads(DATASET1_SWEEP_PATH.read_text(encoding="utf-8"))
    ks = json.loads(KS_EVAL_PATH.read_text(encoding="utf-8"))
    cal = json.loads(KS_CALIB_PATH.read_text(encoding="utf-8"))
    imp = json.loads(KS_IMPROVE_PATH.read_text(encoding="utf-8"))
    rob = json.loads(KS_ROBUST_PATH.read_text(encoding="utf-8"))

    generated_at = datetime.now().strftime("%B %d, %Y  %H:%M")

    body = "".join([
        build_cover(generated_at),
        build_toc(),
        part_divider("Part I &mdash; System Architecture", "The complete pipeline, from document upload to final report, stage by stage."),
        s1_philosophy(), s2_task_definition(), s3_stage_overview(), s3_stage0(), s4_stage1(), s5_stage2(), s6_stage3(),
        s7_stage4(), s8_stage5(), s9_stage6(), s10_stage7(), s11_stage8(), s12_verdict(d1_sweep), s13_operational(),
        s15_related_work(),

        part_divider("Part II &mdash; Datasets", "Every dataset this pipeline has been evaluated against."),
        s14_dataset12(), s15_kickstarter(),

        part_divider("Part III &mdash; Evaluation Methodology", "How each study was designed, and why."),
        s16_methodology_human(), s17_methodology_outcome(), s18_methodology_calib(),
        s19_methodology_improve(), s20_methodology_robust(), s21_methodology_ground(),

        part_divider("Part IV &mdash; Full Results", "Complete, current, numeric results for every study."),
        s24_results_human(d1), s23_results_kickstarter(ks), s23b_kickstarter_highlights(cal),
        s24_results_calibration(cal), s25_results_improve(imp, None), s26_results_robust(rob, imp),
        s29b_audit_matrix(),

        part_divider("Part V &mdash; Synthesis", "What the two evaluation programs together imply."),
        s27_cross_dataset(), s28_findings(), s29_limitations(),

        part_divider("Appendix", "Reference material."),
        appendix_glossary(), appendix_scripts(),
    ])

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>The YangtzeDelta Proposal Analyser &mdash; Full Project Report</title>
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
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
        f"--print-to-pdf={pdf_path}", "--print-to-pdf-no-header", "--no-pdf-header-footer",
        str(html_path),
    ]
    print("[..] Generating PDF via Chrome headless ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size // 1024
        print(f"[OK] PDF written  -> {pdf_path}  ({size_kb} KB)")
    else:
        print("[FAIL] PDF generation failed.")
        print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
