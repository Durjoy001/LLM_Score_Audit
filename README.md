---
title: YangtzeDelta Proposal Analyser
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# YangtzeDelta Proposal Analyser

YangtzeDelta Proposal Analyser is a FastAPI-based AI proposal review system. It ingests proposal documents, extracts text and evidence, builds review facts and dimensions, generates questions and LLM answers, then produces an expert-style report and rubric scores for evaluation.

The supported web entry point is:

```text
src/api/server.py
```

The legacy CLI file `src/main.py` still references missing `backend.chains.*` modules and is not the current supported way to run the project.

## Review Dimensions

The main pipeline works around five review dimensions:

- `team`: team members, governance, collaboration, roles, delivery capacity
- `objectives`: project goals, milestones, success metrics, market need
- `strategy`: technical route, implementation path, market/regulatory strategy
- `innovation`: novelty, differentiation, IP, technical evidence
- `feasibility`: resources, funding, platforms, schedule, risks, mitigations

Stage 8 maps these to the human scoring rubric:

```text
team -> team
objectives -> objective
strategy -> strategy
innovation -> advantages
feasibility -> feasibility
```

The main evaluation target is rank agreement with human reviewers: when humans rank proposal A above proposal B, the AI evaluator should usually preserve that ordering.

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install system dependencies for PDF OCR. On macOS:

```bash
brew install poppler tesseract
```

OCR defaults to Chinese plus English:

```bash
TESS_LANG=chi_sim+eng
```

If Chinese OCR language data is not installed, install it or set `TESS_LANG=eng`.

Create `.env` from `.env.example` and configure at least one provider:

```bash
PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_BASE=https://api.openai.com/v1
```

Other supported provider settings are `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` and `GEMINI_API_KEY` / `GEMINI_MODEL`.

If multiple API keys are present, the research-facing stages can run all configured providers:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

`PROVIDER` still matters. It chooses the preferred canonical output copied back to compatibility paths when more than one provider succeeds:

```bash
PROVIDER=openai   # preferred canonical provider when OpenAI succeeds
PROVIDER=gemini   # preferred canonical provider when Gemini succeeds
PROVIDER=deepseek # preferred canonical provider when DeepSeek succeeds
```

For research runs, `run_multi_provider_lifecycle.py` creates independent provider lifecycles from Stage 0 through Stage 8. Each provider gets its own artifacts, and blocked providers do not stop the others.

Useful optional settings:

```bash
ENABLE_VISION=true
VISION_MODEL=gpt-4o
HTTP_TIMEOUT_CONNECT=12
HTTP_TIMEOUT_READ=60
RESEND_API_KEY=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
SMTP_FROM=
```

Start the web app:

```bash
./start.sh
```

Open:

```text
http://localhost:8000
```

The browser UI supports upload, progress tracking, report preview/download, optional email notification, and evaluation workflows.

## Supported Upload Types

The API accepts:

- PDF
- DOCX / DOC
- PPTX / PPT
- TXT
- Markdown

Uploaded files are saved under `src/data/proposals/` with a generated proposal id (`pid`) and a metadata file.

## End-To-End Pipeline

The web upload flow runs these 9 stages in order:

| Stage | Script | Purpose |
|---|---|---|
| 0 | `prepare_proposal_text` | Extract text from the uploaded document |
| 1 | `extract_facts_by_chunk` | Extract atomic facts |
| 2 | `build_dimensions_from_facts` | Build the five review dimensions |
| 3 | `generate_questions` | Generate evaluation questions |
| 4 | `llm_answering` | Produce answer candidates |
| 5 | `post_processing` | Filter and select the best answers |
| 6 | `ai_expert_opinion` | Write the expert-style review |
| 7 | `generate_final_report` | Compile the final report |
| 8 | `generate_llm_scores` | Produce rubric scores |

`src/api/server.py` runs each stage as a subprocess, keeps job state in memory, streams progress, writes per-stage progress files under `src/data/`, and records a session log under `output/`.

## Command Lifecycle From `input.pdf`

This section is the copy-paste path for a new user. Put an example file at `./input.pdf`, choose a proposal id, and run either the automatic provider lifecycle or each stage manually.

### Option A: Full Provider Lifecycle

This runs Stage 0 through Stage 8 for every healthy provider found in `.env`. Non-working providers are skipped after a small preflight check.

```bash
source .venv/bin/activate

INPUT=./input.pdf
PID=input_pdf_demo

.venv/bin/python src/tools/run_multi_provider_lifecycle.py \
  --file "$INPUT" \
  --pid "$PID" \
  --providers auto \
  --workers 1
```

Provider-specific outputs use ids like:

```text
input_pdf_demo__openai
input_pdf_demo__gemini
input_pdf_demo__deepseek
```

The preferred successful provider from `PROVIDER` is also copied to the base compatibility paths:

```text
src/data/reports/input_pdf_demo_final_report.md
src/data/llm_scores/input_pdf_demo/llm_scores.json
```

### Option B: Manual Stage-By-Stage Run

Use this when you want to understand or debug every stage one by one. Set `PROVIDER` to the single provider you want this manual path to use.

```bash
source .venv/bin/activate

INPUT=./input.pdf
PID=input_pdf_demo
PROVIDER=openai

# Stage 0: extract document text.
.venv/bin/python src/tools/prepare_proposal_text.py \
  --file "$INPUT" \
  --proposal_id "$PID"

# Stage 1: extract atomic facts.
.venv/bin/python src/tools/extract_facts_by_chunk.py \
  --proposal_id "$PID"

# Stage 2: build review dimensions from facts.
.venv/bin/python src/tools/build_dimensions_from_facts.py \
  --proposal_id "$PID"

# Stage 3: generate evaluation questions.
.venv/bin/python src/tools/generate_questions.py \
  --proposal_id "$PID" \
  --llm_provider "$PROVIDER"

# Stage 4: answer generated questions.
.venv/bin/python src/tools/llm_answering.py \
  --proposal_id "$PID" \
  --qs_file "src/data/questions/$PID/generated_questions.json" \
  --providers "$PROVIDER"

# Stage 5: score/filter/select answers.
.venv/bin/python src/tools/post_processing.py \
  --pid "$PID" \
  --qs_file "src/data/questions/$PID/generated_questions.json"

# Stage 6: generate AI expert opinion.
.venv/bin/python src/tools/ai_expert_opinion.py \
  --pid "$PID" \
  --provider "$PROVIDER"

# Stage 7: compile final Markdown report.
.venv/bin/python src/tools/generate_final_report.py \
  --pid "$PID"

# Stage 8: generate rubric scores for evaluation.
.venv/bin/python src/tools/generate_llm_scores.py \
  --pid "$PID" \
  --provider "$PROVIDER"
```

After this manual run, the main report and score files are:

```text
src/data/reports/input_pdf_demo_final_report.md
src/data/llm_scores/input_pdf_demo/llm_scores.json
```

### Optional Multi-Provider Stage 6/8 Comparison

If you already ran Stages 0 through 5 for one `PID`, these commands generate provider-specific expert opinions and rubric scores for every healthy provider in `.env`:

```bash
.venv/bin/python src/tools/generate_multi_provider_expert_opinions.py \
  --pid "$PID" \
  --providers auto \
  --workers 1

.venv/bin/python src/tools/generate_multi_provider_scores.py \
  --pid "$PID" \
  --providers auto \
  --workers 1
```

### Evaluation Commands

Run Cohen's kappa / human-vs-AI agreement after Stage 8 scores exist. The human Excel file must be present at the path you pass with `--human_xlsx`.

```bash
.venv/bin/python src/tools/evaluate_cohens_kappa.py \
  --human_xlsx src/data/evaluations/human/human_scores.xlsx \
  --provider "$PROVIDER"
```

Run groundedness for the final report:

```bash
.venv/bin/python src/tools/evaluate_groundedness.py \
  --pid "$PID" \
  --provider "$PROVIDER"
```

Run self-consistency sweep for Stage 8 scoring:

```bash
.venv/bin/python src/tools/evaluate_sweep.py \
  --pid "$PID" \
  --runs 5 \
  --provider "$PROVIDER"
```

Run sensitivity only after you have at least two completed proposal ids, for example a baseline and one variant:

```bash
BASELINE_PID=input_pdf_demo
VARIANT_PID=input_pdf_variant

.venv/bin/python src/tools/evaluate_sensitivity.py \
  --pids "$BASELINE_PID,$VARIANT_PID" \
  --baseline_pid "$BASELINE_PID" \
  --provider "$PROVIDER"
```

Evaluation outputs are written under:

```text
src/data/evaluations/runs/<eval_id>/
```

## Pipeline Stages

### Stage 0: Text Preparation

Script:

```text
src/tools/prepare_proposal_text.py
```

Purpose:

- Extract stage-ready text from PDF, DOCX, PPTX, TXT, or Markdown files.
- Preserve page/slide boundaries, source type, offsets, and page diagnostics.
- Prefer `pdfplumber` for PDF text, then OCR when needed.
- Optionally call a vision model for images, charts, tables, diagrams, and slide/document pictures.

Outputs:

```text
src/data/prepared/<pid>/full_text.txt
src/data/prepared/<pid>/pages.json
src/data/prepared/<pid>/stage0_audit.json
```

### Stage 1: Fact Extraction

Script:

```text
src/tools/extract_facts_by_chunk.py
```

Purpose:

- Read `full_text.txt`, chunk it, and extract atomic proposal facts.
- Attach facts to review dimensions and fact types.
- Preserve chunk/character traceability and retry suspiciously sparse chunks.

Outputs:

```text
src/data/extracted/<pid>/raw_facts.jsonl
src/data/extracted/<pid>/stage1_audit.json
```

### Stage 2: Dimension Building

Script:

```text
src/tools/build_dimensions_from_facts.py
```

Purpose:

- Bucket Stage 1 facts into the five dimensions.
- Build grounded summaries, key points, risks, and mitigations.
- Write the parsed dimension files used downstream.

Outputs:

```text
src/data/extracted/<pid>/dimension_facts.json
src/data/extracted/<pid>/dimensions_v2.json
src/data/parsed/parsed_dimensions.clean.llm.json
```

### Stage 3: Question Generation

Script:

```text
src/tools/generate_questions.py
```

Purpose:

- Read `dimensions_v2.json`.
- Generate dimension-specific questions anchored to facts, risks, mitigations, and missing evidence.
- Produce both answer-facing and audit-facing question sets.

Outputs:

```text
src/data/questions/<pid>/generated_questions.json
src/data/questions/<pid>/generated_questions_detail.json
src/data/questions/<pid>/stage_questions_audit.json
src/data/config/question_sets/generated_questions.json
```

### Stage 4: LLM Answering

Script:

```text
src/tools/llm_answering.py
```

Purpose:

- Answer each question using proposal-derived facts.
- Produce structured answer candidates and keep grounded claims separate from general context.
- Support multiple answer variants/providers where configured.

Outputs:

```text
src/data/refined_answers/<pid>/all_refined_items.json
src/data/refined_answers/<pid>/chatgpt_raw.json
src/data/refined_answers/<pid>/gemini_raw.json
src/data/refined_answers/<pid>/deepseek_raw.json
src/data/refined_answers/<pid>/stage4_answering_audit.json
```

Stage 4 is usually the slowest stage because it can make many LLM calls. It auto-detects configured providers, runs usable providers concurrently, and merges all provider candidates into `all_refined_items.json`. The stage logs include provider health checks, call count, per-call timings, slow-call locations, and per-provider raw output paths.

### Stage 5: Post-Processing

Script:

```text
src/tools/post_processing.py
```

Purpose:

- Score, filter, select, and aggregate Stage 4 answer candidates.
- Build the final payload for expert opinion and report generation.
- Record confidence, alignment, drift, provider selection, drops, and dimension-level scoring.

Outputs:

```text
src/data/refined_answers/<pid>/postproc/metrics.json
src/data/refined_answers/<pid>/postproc/selected_by_question.json
src/data/refined_answers/<pid>/postproc/final_payload.json
src/data/refined_answers/<pid>/postproc/report.md
src/data/refined_answers/<pid>/postproc/drops_debug.json
src/data/refined_answers/<pid>/postproc/stage5_post_processing_audit.json
```

### Stage 6: AI Expert Opinion

Script:

```text
src/tools/generate_multi_provider_expert_opinions.py
```

Purpose:

- Convert Stage 5 selected answers and metrics into an expert-style review.
- Generate provider-specific expert opinions, then copy the preferred one back to the canonical Stage 6 paths used by Stage 7.
- Record blocked or failed providers in `multi_provider_expert_opinions.json`.

Outputs:

```text
src/data/expert_reports/<pid>/openai/ai_expert_opinion.json
src/data/expert_reports/<pid>/gemini/ai_expert_opinion.json
src/data/expert_reports/<pid>/deepseek/ai_expert_opinion.json
src/data/expert_reports/<pid>/multi_provider_expert_opinions.json
src/data/expert_reports/<pid>/ai_expert_opinion.prompt.json
src/data/expert_reports/<pid>/ai_expert_opinion.json
src/data/expert_reports/<pid>/ai_expert_opinion.md
src/data/expert_reports/<pid>/stage6_ai_expert_opinion_audit.json
```

### Stage 7: Final Report

Script:

```text
src/tools/generate_final_report.py
```

Purpose:

- Compile the final user-facing Markdown report.
- Combine executive summary, expert opinion, dimension-level review, and selected Q&A traceability.
- Cap visible Q&A rows per dimension using `MAX_QA_PER_DIM`.

Outputs:

```text
src/data/reports/<pid>_final_report.md
src/data/reports/<pid>_stage7_final_report_audit.json
```

### Stage 8: LLM Rubric Scores

Script:

```text
src/tools/generate_multi_provider_scores.py
```

Purpose:

- Score the final report against the human evaluation rubric with every configured provider.
- Produce `team`, `objective`, `strategy`, `advantages`, `feasibility`, `overall_ranking`, and `verdict`.
- Save the exact scoring prompt for audit and keep strict provider scoring.
- Copy the preferred successful provider back to `src/data/llm_scores/<pid>/llm_scores.json` for compatibility.

Outputs:

```text
src/data/llm_scores/<pid>/openai/llm_scores.json
src/data/llm_scores/<pid>/gemini/llm_scores.json
src/data/llm_scores/<pid>/deepseek/llm_scores.json
src/data/llm_scores/<pid>/multi_provider_scores.json
src/data/llm_scores/<pid>/llm_scores.prompt.json
src/data/llm_scores/<pid>/llm_scores.json
```

## Research Multi-Provider Lifecycle

For model-comparison experiments, run the full lifecycle from Stage 0 with one provider-specific path per LLM:

```bash
.venv/bin/python src/tools/run_multi_provider_lifecycle.py \
  --file path/to/proposal.pdf \
  --pid "$PID" \
  --providers auto
```

If `.env` contains OpenAI, Gemini, and DeepSeek keys, this creates:

```text
<pid>__openai
<pid>__gemini
<pid>__deepseek
```

Each provider lifecycle writes its own artifacts under the normal data roots:

```text
src/data/prepared/<pid>__<provider>/
src/data/extracted/<pid>__<provider>/
src/data/questions/<pid>__<provider>/
src/data/refined_answers/<pid>__<provider>/
src/data/expert_reports/<pid>__<provider>/
src/data/reports/<pid>__<provider>_final_report.md
src/data/llm_scores/<pid>__<provider>/llm_scores.json
```

The lifecycle summary is written to:

```text
src/data/provider_lifecycles/<pid>/provider_lifecycle_summary.json
```

The preferred successful provider, controlled by `PROVIDER`, is copied back to compatibility paths such as `src/data/reports/<pid>_final_report.md` and `src/data/llm_scores/<pid>/llm_scores.json`.

## Running Stages Manually

The browser upload flow uses the provider lifecycle runner by default. Set `USE_PROVIDER_LIFECYCLE=false` to return to the older single-PID pipeline while debugging. For a single-provider path, run stages manually with a fixed `PID`:

```bash
PID=my_proposal_debug

.venv/bin/python src/tools/prepare_proposal_text.py \
  --file path/to/proposal.pdf \
  --proposal_id "$PID"

.venv/bin/python src/tools/extract_facts_by_chunk.py \
  --proposal_id "$PID"

.venv/bin/python src/tools/build_dimensions_from_facts.py \
  --proposal_id "$PID"

.venv/bin/python src/tools/generate_questions.py \
  --proposal_id "$PID"

.venv/bin/python src/tools/llm_answering.py \
  --proposal_id "$PID" \
  --qs_file "src/data/questions/$PID/generated_questions.json" \
  --providers auto

.venv/bin/python src/tools/post_processing.py \
  --pid "$PID" \
  --qs_file "src/data/questions/$PID/generated_questions.json"

.venv/bin/python src/tools/generate_multi_provider_expert_opinions.py \
  --pid "$PID" \
  --providers auto

.venv/bin/python src/tools/generate_final_report.py \
  --pid "$PID"

.venv/bin/python src/tools/generate_multi_provider_scores.py \
  --pid "$PID" \
  --providers auto
```

Generate provider-specific Stage 8 scores for comparison:

```bash
.venv/bin/python src/tools/generate_multi_provider_scores.py \
  --pid "$PID" \
  --providers auto
```

Provider-specific scores are written to:

```text
src/data/llm_scores/<pid>/openai/llm_scores.json
src/data/llm_scores/<pid>/gemini/llm_scores.json
src/data/llm_scores/<pid>/deepseek/llm_scores.json
src/data/llm_scores/<pid>/multi_provider_scores.json
```

There is also an automatic latest-proposal runner:

```bash
.venv/bin/python src/tools/run_pipeline.py
```

That runner assumes each stage can auto-detect the latest proposal when identifiers are omitted. The web server is safer for normal use because it passes explicit file and proposal id arguments to every stage.

## API Overview

Main proposal flow:

```text
POST  /api/upload                  Upload proposal file and optional email
POST  /api/run/{job_id}            Start the background pipeline
PATCH /api/jobs/{job_id}/email     Add or update notification email
GET   /api/status/{job_id}         Poll job status
GET   /api/events/{job_id}         Stream status via server-sent events
GET   /api/report/{job_id}         Return final report Markdown as JSON
GET   /api/download/{job_id}       Download final report Markdown
```

Evaluation flow:

```text
POST /api/evaluation/human_upload                  Upload human_scores.xlsx
POST /api/evaluation/run                           Run Cohen's kappa evaluation
POST /api/evaluation/groundedness/run              Run groundedness check for a report
POST /api/evaluation/sweep/run                     Re-run Stage 8 scoring multiple times
POST /api/evaluation/sensitivity/upload_and_run    Upload baseline + variants, run pipelines, compare
POST /api/evaluation/sensitivity/run               Compare existing proposal IDs
GET  /api/evaluation/status/{eval_id}              Poll evaluation status
GET  /api/evaluation/pids                          List report-backed proposal IDs
```

## Evaluation Tools

The evaluation scripts live under `src/tools/`:

- `run_multi_provider_lifecycle.py`: runs Stage 0 through Stage 8 independently for every configured provider.
- `generate_multi_provider_scores.py`: writes strict Stage 8 scores per provider.
- `evaluate_cohens_kappa.py`: compares human Excel scores with Stage 8 outputs.
- `evaluate_groundedness.py`: checks whether report claims are supported by the proposal text.
- `evaluate_sweep.py`: runs Stage 8 multiple times to measure self-consistency.
- `evaluate_sensitivity.py`: compares baseline and variant reports/scores to check directionality.

Run human-vs-AI rank evaluation for one provider:

```bash
.venv/bin/python src/tools/evaluate_cohens_kappa.py \
  --human_xlsx src/data/evaluations/human/human_scores.xlsx \
  --provider openai

.venv/bin/python src/tools/evaluate_cohens_kappa.py \
  --human_xlsx src/data/evaluations/human/human_scores.xlsx \
  --provider gemini
```

Run groundedness, self-consistency, and sensitivity with a specific provider:

```bash
.venv/bin/python src/tools/evaluate_groundedness.py --pid "$PID" --provider openai
.venv/bin/python src/tools/evaluate_sweep.py --pid "$PID" --runs 5 --provider openai
.venv/bin/python src/tools/evaluate_sensitivity.py --pids "$BASELINE_PID,$VARIANT_PID" --provider openai
```

Use the same commands with `--provider gemini` when the Gemini API key/project is enabled for the Generative Language API.

Evaluation outputs are written under:

```text
src/data/evaluations/human/human_scores.xlsx
src/data/evaluations/runs/<eval_id>/evaluation_summary.json
src/data/evaluations/runs/<eval_id>/evaluation_report.xlsx
src/data/evaluations/runs/<eval_id>/groundedness_result.json
src/data/evaluations/runs/<eval_id>/sweep_summary.json
src/data/evaluations/runs/<eval_id>/sweep_report.xlsx
src/data/evaluations/runs/<eval_id>/sweep_bundle.zip
src/data/evaluations/runs/<eval_id>/sensitivity_summary.json
src/data/evaluations/runs/<eval_id>/sensitivity_report.xlsx
```

## Research & Calibration Tools

Beyond the core evaluation scripts, the repo has a layer of research tooling used to calibrate the pipeline against the 12-proposal human-expert dataset (`DeltaProjectDatasets/`) and to stress-test whether the AI scorer is actually reading proposal content.

### Batch pipeline runners

- `run_all_proposals.py`: runs Stages 0-8 for every proposal in `DeltaProjectDatasets/Dataset1` and `Dataset2`, then runs the Cohen's kappa evaluation against `ExpertsEvaluations.xlsx`. Outputs land in `DeltaProjectDatasets/Results/`.
- `rerun_stage6_7.py`: re-runs only Stage 6 (expert opinion) + Stage 7 (final report) for all 12 proposals when only the invest-scoring rubric changed, then re-copies reports and re-evaluates. Cheaper than a full re-run when iterating on `src/tools/ai_expert_opinion.py`.

### Calibration sweep

`calibration_sweep.py` performs a parameter search with **no LLM calls** — it reads cached `invest_score_echo` / `qa_score_echo` values already stored in `src/data/expert_reports/<pid>/` and:

- Sweeps `INVEST_WEIGHT` (the blend of invest-model score vs QA-answer score) to maximize Spearman correlation with the 12 human expert scores.
- Runs `sweep_verdict_thresholds()`, a 2-D grid search over separate overall-score and confidence thresholds combined with OR logic, to maximize GO/HOLD verdict accuracy against the human verdict column.

Results are written to `DeltaProjectDatasets/Results/calibration_sweep.json`. Because it never calls an LLM, it can be re-run cheaply after any change to blend weights or thresholds — a Stage 6/7 re-run (`rerun_stage6_7.py`) is only needed to pick up new invest scores.

### Granular sub-rubric scoring (human-vs-AI diagnostic)

Three tools score a completed proposal on **8 fine-grained sub-rubrics per dimension** (R1-R8), separately from the main pipeline, to diagnose where AI and human judgment agree vs diverge:

| Dimension | Scoring tool | Batch runner | Comparison script |
|---|---|---|---|
| Strategy | `src/tools/generate_strategy_rubric_scores.py` | `run_strategy_rubric_batch.py` | `rubric_comparison.py` |
| Advantages/Innovation | `src/tools/generate_advantages_rubric_scores.py` | `run_advantages_rubric_batch.py` | `advantages_rubric_comparison.py` |
| Objectives | `src/tools/generate_objectives_rubric_scores.py` | `run_objectives_rubric_batch.py` | `objectives_rubric_comparison.py` |

Each dimension's 8 sub-rubrics split into two families:

- **R1-R4, document-verifiable**: facts stated explicitly in the proposal (milestone dates, revenue model, regulatory path, market size). AI is expected to track human raters closely here.
- **R5-R8, judgment-dependent**: signals that require tacit investor judgment (competitive moat credibility, team-strategy fit, market timing). This is where AI is expected to diverge from human experts.

Results are stored per proposal at `src/data/<dimension>_rubric_scores/<pid>/<dimension>_rubric_scores.json` and compared against a human baseline (from `ExpertsEvaluations.xlsx`) by each `*_rubric_comparison.py` script, which prints a Spearman correlation per sub-rubric plus an averaged score for each family. `human_scorecard_template.csv` is a blank template for collecting a second human rater's sub-rubric scores on the same 8 rubrics, to eventually compute a human-vs-human agreement ceiling for the AI-vs-human comparison.

### Content-sensitivity experiment

`run_sensitivity_all_dimensions.py` tests whether the AI scorer is actually reading proposal content rather than reacting to writing style or length. For each proposal it builds two edited variants of the report text:

- **PLUS**: injects concrete, domain-specific evidence for a sub-rubric (e.g. a named revenue model, a filed regulatory pathway) that should make the relevant score *rise*.
- **MINUS**: strips the key evidence backing a sub-rubric (e.g. removes named partners, timing rationale) that should make the relevant score *fall*.

Two modes:

- **Default (hardcoded)**: fixed PLUS/MINUS content per dimension, covering 2 sub-rubrics each for strategy, advantages, and objectives.
- **Adaptive (`--adaptive`)**: an LLM generates proposal-specific PLUS/MINUS content covering all 8 sub-rubrics per dimension — 16 hypothesis checks per proposal instead of 4.

Supporting scripts:

- `present_sensitivity_results.py`: reads cached results (no LLM calls) and prints a proposal × rubric matrix for baseline, PLUS, and MINUS scores, plus binomial-test pass rates.
- `generate_analysis_report.py`: produces a deep-dive HTML/PDF (`sensitivity_analysis_report.html` / `.pdf`) explaining exactly what content was injected/stripped per rubric, root-causing unexpected score movements (PLUS cross-contamination into non-target rubrics, MINUS stochastic noise on proposals whose text never contained the target vocabulary, incomplete stripping), and flagging which hypothesis checks are valid vs confounded.
- `run_sensitivity_experiment.py`: earlier single-dimension (strategy-only) version of the experiment, kept for reference.

Outputs: `src/data/sensitivity_results/sensitivity_<dimension>.json`, `sensitivity_report.html` / `.pdf` (sub-rubric-level results), `sensitivity_analysis_report.html` / `.pdf` (methodology deep-dive).

## Results

Numbers below are from the 12-proposal Dataset1 + Dataset2 run (`DeltaProjectDatasets/Results/`, human baseline `DeltaProjectDatasets/ExpertsEvaluations.xlsx`, OpenAI provider). Re-running `run_all_proposals.py` or `rerun_stage6_7.py` regenerates `DeltaProjectDatasets/Results/evaluation/evaluation_summary.json`; exact figures can shift slightly between runs (see Known Limitations below).

### Rank agreement with human experts

| Dimension | Weighted κ | Spearman r | Mean human | Mean AI |
|---|---|---|---|---|
| Team | 0.32 | 0.398 | 0.73 | 0.69 |
| Objective | -0.14 | -0.275 | 0.83 | 0.65 |
| Strategy | -0.13 | 0.324 | 0.71 | 0.65 |
| Advantages | -0.08 | 0.061 | 0.67 | 0.72 |
| Feasibility | 0.25 | 0.156 | 0.67 | 0.68 |
| **Overall ranking** | 0.12 (ICC) | **0.589** | 0.82 | 0.68 |

- Pairwise rank concordance on overall ranking: **75.9%** (44/58 comparable pairs concordant; 8 human-tied pairs excluded).
- An earlier rubric redesign (documented in `ai_expert_opinion.py`'s `INVEST_SCORE_SYSTEM`) raised overall Spearman from roughly +0.11 to the current +0.589.
- Objective and Advantages dimensions still show near-zero or negative Spearman/kappa and are the main open gap.

### Verdict (GO/HOLD) accuracy

The verdict rule was recalibrated from an AND-combination to an OR-combination of overall score and confidence, using `calibration_sweep.sweep_verdict_thresholds()`:

| Rule | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Previous: `score ≥ 0.65 AND confidence ≥ 0.65` | 58.3% | 44.4% | 100% | 61.5% |
| Current: `score ≥ 0.725 OR confidence ≥ 0.811` | **83.3%** | **100%** | 50% | 66.7% |

This was a zero-LLM-call, thresholds-only change and left overall Spearman unchanged at 0.589. The OR rule trades recall for precision: it now issues zero false GO verdicts (down from 5 false positives), at the cost of missing half of the true GO proposals (2 false negatives vs 0 before).

### Content-sensitivity experiment (does the AI read the document?)

Across 417 structurally-testable hypothesis checks (12 proposals × 3 dimensions × 8 sub-rubrics × PLUS/MINUS, with floor/ceiling-effect checks excluded), scores moved in the expected direction **87% of the time** (366/417; one-sided binomial p < 0.001 vs the 50% chance baseline; Cohen's h = 0.86, a large effect):

| Dimension / Variant | Checks passed | Pass rate | Cohen's h |
|---|---|---|---|
| Strategy — PLUS | 88/96 | 91% | 0.99 (large) |
| Strategy — MINUS | 49/55 | 89% | 0.90 (large) |
| Advantages — PLUS | 64/95 | 67% | 0.35 (medium) |
| Advantages — MINUS | 31/33 | 93% | 1.07 (large) |
| Objectives — PLUS | 91/95 | 95% | 1.16 (large) |
| Objectives — MINUS | 43/43 | 100% | 1.57 (large) |

PLUS injections (adding explicit evidence) were rewarded almost universally. MINUS removals (stripping evidence) were more variable on judgment-dependent sub-rubrics (e.g. Problem-Solution Fit, Market Timing, Competitive Defensibility) — the model sometimes infers these from contextual cues that survive the stripping operation. Advantages-PLUS is the weakest cell (67%, medium effect) because the scorer stays skeptical of self-reported competitive claims even when injected explicitly. Full per-rubric detail: `sensitivity_report.html`; root-cause analysis: `sensitivity_analysis_report.html`.

### Known limitations

- **Spearman ceiling around 0.60**: human experts appear to draw on implicit domain knowledge beyond what the proposal document states; the AI evaluator only scores what the text explicitly says, which caps achievable agreement.
- **Stage 6 invest scoring is not fully deterministic** run-to-run — temperature is low but non-zero on the main dimension-analysis call, so borderline proposals near a verdict threshold can flip between runs. Treat single-run metrics as a point estimate, not an exact reproducible constant.
- **Sub-rubric human-vs-AI comparison** currently compares against a single human rater's proposal-level score, not a second rater's sub-rubric scores; `human_scorecard_template.csv` exists to collect that second rating for a proper human-vs-human agreement ceiling.

## Output Locations

Runtime data is written under `src/data/`:

```text
src/data/proposals/          uploaded files and upload metadata
src/data/prepared/           extracted text, pages, Stage 0 audits
src/data/extracted/          extracted facts and dimension files
src/data/parsed/             active parsed dimension file
src/data/questions/          per-proposal question sets
src/data/config/             active generated question set
src/data/refined_answers/    LLM answers and post-processing outputs
src/data/expert_reports/     expert opinion JSON/Markdown
src/data/reports/            final Markdown reports
src/data/llm_scores/         Stage 8 rubric scores
src/data/evaluations/        human/evaluation uploads and run outputs
```

Durable per-session logs are written under:

```text
output/
```

Example:

```text
output/20260513_093904_<pid>_<job_id>_pipeline_log.txt
```

The log captures upload metadata, command lines, stdout/stderr for every stage, start/finish markers, failure tails, final report path, and email status.

Find the latest log with:

```bash
ls -lt output | head
```

## Troubleshooting

### Quick checks

- OCR too empty: verify `pdftoppm`, `tesseract`, `TESS_LANG`, and inspect `src/data/prepared/<pid>/pages.json` plus `stage0_audit.json`.
- Stage 4 slow: reduce question count or provider count, and inspect `[LLM_CALL]` timing in logs.
- Stage 6 fallback: inspect `src/data/expert_reports/<pid>/stage6_ai_expert_opinion_audit.json`; `mode=llm` means LLM mode, otherwise local fallback.
- Stage 8 fallback: inspect `src/data/llm_scores/<pid>/llm_scores.json`; if `meta.mode=local_fallback`, check provider keys and `PROVIDER`.
- Fewer Q&A rows in the final report: expected, because Stage 7 caps visible rows with `MAX_QA_PER_DIM`; full data remains in `src/data/refined_answers/<pid>/postproc/final_payload.json`.
- Job disappeared after restart: browser job state is in memory only; durable artifacts stay in `src/data/` and `output/`.
