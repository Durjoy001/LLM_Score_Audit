# -*- coding: utf-8 -*-
"""
Shared utility: extract 0-1 dimension scores from a final report markdown.

Reads the Dimension Score Overview table produced by Stage 5/6 and returns
raw 0-1 float scores plus the verdict. No integer conversion is applied —
callers decide what scale they need.
"""

import re
import sys
from typing import Any, Dict, List

_DIM_PATTERNS = {
    "team":        r'\|\s*[^|]*\(team\)\s*\|\s*([\d.]+)',
    "objective":   r'\|\s*[^|]*\(objectives\)\s*\|\s*([\d.]+)',
    "strategy":    r'\|\s*[^|]*\(strategy\)\s*\|\s*([\d.]+)',
    "advantages":  r'\|\s*[^|]*\(innovation\)\s*\|\s*([\d.]+)',
    "feasibility": r'\|\s*[^|]*\(feasibility\)\s*\|\s*([\d.]+)',
}

_DEFAULT_SCORE = 0.5


def extract_report_scores(report_md: str) -> Dict[str, Any]:
    """
    Extract 0-1 dimension scores and verdict from a final_report.md string.

    Keys:
      team, objective, strategy, advantages, feasibility  (float, 0-1)
      overall_ranking                                      (float, 0-1)
      confidence                                           (float, 0-1)
      verdict                                              ("Y" or "N")

    Warns to stderr if any pattern fails to match — callers should treat
    missing scores as data-quality issues, not silent neutrals.
    """
    missing: List[str] = []

    m = re.search(r'\*\*Overall verdict:\*\*\s*(GO|HOLD|NO-GO)\b', report_md)
    verdict = "Y" if (m and m.group(1) == "GO") else "N"

    m = re.search(r'\*\*Overall score:\*\*\s*([\d.]+)', report_md)
    if m:
        overall = round(float(m.group(1)), 4)
    else:
        overall = _DEFAULT_SCORE
        missing.append("overall_ranking")

    m = re.search(r'\*\*Confidence:\*\*\s*([\d.]+)', report_md)
    if m:
        confidence = round(float(m.group(1)), 4)
    else:
        confidence = _DEFAULT_SCORE
        missing.append("confidence")

    scores: Dict[str, Any] = {}
    for key, pattern in _DIM_PATTERNS.items():
        m = re.search(pattern, report_md)
        if m:
            scores[key] = round(float(m.group(1)), 4)
        else:
            scores[key] = _DEFAULT_SCORE
            missing.append(key)

    if missing:
        print(
            f"[WARN] score_extraction: regex missed {missing}; defaulting to {_DEFAULT_SCORE}. "
            "Check report markdown formatting.",
            file=sys.stderr,
            flush=True,
        )

    scores["overall_ranking"] = overall
    scores["confidence"] = confidence
    scores["verdict"] = verdict
    return scores
