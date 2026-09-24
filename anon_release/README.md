
# FinNLP 2026 — Reproducibility Material

Anonymous supplementary material for "Score Transfers, Decisions Do Not:
Cross-Domain Threshold Recalibration for Financial LLM Screening"
(FinNLP 2026, EMNLP Workshop, submitted).

## Overview
This repository supports a cross-domain audit of an LLM-based financial
screening pipeline: scores transfer from institutional venture proposals to
crowdfunding campaigns (ROC-AUC 0.764), but the prespecified source-domain
decision rule selects zero target campaigns (0% recall) because both
threshold branches lie above the entire observed target score range.

## Contents
- `data/scores.csv` — per-campaign scores, confidence, funding outcome,
  fold assignment, and out-of-fold predictions for all 141 target campaigns.
- `src/` — pipeline and analysis code, including `b6_verify.py` (verifies
  the structural-failure claim from the data) and threshold-search /
  cross-validation code.
- `prompts/` — LLM prompt templates used by the scoring pipeline.
- `results/b6_threshold_sweep.csv` — score/confidence threshold sweep.
- `configs/model_config.md` — model snapshot and decoding parameters.

## What can be reproduced
All target-domain metrics and figures reported in the paper can be
reproduced from `data/scores.csv` using the scripts in `src/`. Run:

    python src/b6_verify.py data/scores.csv

to verify the structural zero-recall result directly from the data.

## What cannot be reproduced, and why
The 12 source-domain venture proposals and raw crowdfunding campaign texts
are withheld: the former for confidentiality (private deal-flow documents),
the latter due to platform content-licensing and privacy considerations.
The scoring stage is not fully deterministic, so fresh API calls will not
exactly match the released canonical score set; the canonical set is
provided for exact reproduction of reported numbers.

## Environment
See `requirements.txt`. Developed with Python 3.x.

## License
Code released under the MIT License.
