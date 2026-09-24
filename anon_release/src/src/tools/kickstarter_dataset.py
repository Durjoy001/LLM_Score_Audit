# -*- coding: utf-8 -*-
"""
Shared helpers for the Kickstarter dataset: pid<->file mapping and the
`funded (%)` ground truth read from Kickstarter/dataset_info.xlsx.

Pid scheme: kt<n> for Toronto/<n>.pdf, km<n> for Montreal/<n>.pdf.
Uses the per-city xlsx sheets ("Toronto, Canada" / "Montreal, Canada"), not
the "Combined" sheet, since Montreal's `pdf` ids restart at 1 there and
collide with Toronto's without an extra disambiguating column.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import openpyxl

BASE_DIR = Path(__file__).resolve().parents[2]
KICKSTARTER_DIR = BASE_DIR / "Kickstarter"
DATASET_XLSX = KICKSTARTER_DIR / "dataset_info.xlsx"

# (pid prefix, folder/sheet name)
CITIES = [
    ("kt", "Toronto, Canada"),
    ("km", "Montreal, Canada"),
]


def list_proposal_files() -> List[Tuple[str, Path]]:
    """Return [(pid, pdf_path), ...] for every Kickstarter proposal, sorted numerically within each city."""
    out: List[Tuple[str, Path]] = []
    for prefix, folder_name in CITIES:
        folder = KICKSTARTER_DIR / folder_name
        files = sorted(folder.glob("*.pdf"), key=lambda p: int(p.stem))
        out.extend((f"{prefix}{p.stem}", p) for p in files)
    return out


def load_funded_pct() -> Dict[str, float]:
    """Return {pid: funded_pct} for every Kickstarter proposal."""
    wb = openpyxl.load_workbook(DATASET_XLSX, data_only=True)
    result: Dict[str, float] = {}
    for prefix, sheet_name in CITIES:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        header = [str(c or "").strip().lower() for c in rows[0]]
        pdf_idx = header.index("pdf")
        pct_idx = header.index("funded (%)")
        for row in rows[1:]:
            if row[pdf_idx] is None:
                continue
            n = int(row[pdf_idx])
            pct = float(row[pct_idx])
            result[f"{prefix}{n}"] = pct
    return result


def pilot_pids(n_toronto: int = 10, n_montreal: int = 4) -> List[str]:
    """
    Pick a spread of pids per city, stratified by *unique* funded_pct values, for a pilot run.

    Roughly 40-45% of proposals in each city are tied at exactly 0% funded (true
    zero-backer failures), so stratifying by index would over-sample that tied
    cluster. Stratifying by unique value instead guarantees the pilot spans the
    full range (a few zeros, some partial/near-100%, and the high outliers)
    rather than mostly zeros.
    """
    funded = load_funded_pct()
    out: List[str] = []
    for prefix, count in (("kt", n_toronto), ("km", n_montreal)):
        city_items = sorted(((p, v) for p, v in funded.items() if p.startswith(prefix)), key=lambda kv: kv[1])
        unique_vals = sorted({v for _, v in city_items})
        if len(unique_vals) <= count:
            chosen_vals = unique_vals
        else:
            idxs = sorted({round(i * (len(unique_vals) - 1) / (count - 1)) for i in range(count)})
            chosen_vals = [unique_vals[i] for i in idxs]
        picked: List[str] = []
        for val in chosen_vals:
            for pid, v in city_items:
                if v == val and pid not in picked:
                    picked.append(pid)
                    break
        out.extend(picked[:count])
    return out
