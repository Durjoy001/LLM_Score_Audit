# Pipeline Time Sink Analysis Method

This note explains how the "big time sinks" table was produced from the successful run log:

`output/20260508_224733_bp1_82a34876_82a34876_pipeline_log.txt`

## 1. Identify The Pipeline Steps

I first searched the log for runner-level step markers:

```bash
rg -n "\[STEP\] Starting|\[STEP\] Completed|STEP DONE|COMMAND:" output/20260508_224733_bp1_82a34876_82a34876_pipeline_log.txt
```

Those lines show the full ordered pipeline:

1. `prepare_proposal_text`
2. `extract_facts_by_chunk`
3. `build_dimensions_from_facts`
4. `generate_questions`
5. `llm_answering`
6. `post_processing`
7. `ai_expert_opinion`
8. `generate_final_report`

For each step, I used the timestamp on `[STEP] Starting ...` and the timestamp on `[STEP] Completed ...` or `[OK] STEP DONE ...` to estimate wall-clock duration.

## 2. Compare Runner Wall Time With Script Internal Timing

Then I searched for script-level timing messages:

```bash
rg -n "(elapsed_sec|TIMING|SUMMARY|Completed Stage)" output/20260508_224733_bp1_82a34876_82a34876_pipeline_log.txt
```

This matters because there are two different clocks:

- Runner wall time: how long the pipeline step occupied the overall job.
- Internal script timing: how much time the script itself reported for measured work.

When those two numbers differ sharply, it is a signal that something happened outside the measured section, such as network blocking, process suspension, uninstrumented startup/shutdown, or machine sleep.

The main mismatch was `extract_facts_by_chunk`: the runner spent about 5 hours and 8 minutes on it, while the script reported `elapsed_sec=1478.16`, about 24 minutes and 38 seconds.

## 3. Drill Into Stage 1 Chunk Timings

For fact extraction, I inspected both the log segment and the Stage 1 audit file:

```bash
sed -n '360,594p' output/20260508_224733_bp1_82a34876_82a34876_pipeline_log.txt
jq '{elapsed_sec, chunk_count, chunks: [.chunks[] | {n:.chunk_number, elapsed:.elapsed_sec, calls:.llm_calls}]}' src/data/extracted/bp1_82a34876/stage1_audit.json
```

This showed that chunks 22 and 23 each had one OpenAI call taking about 10 minutes:

- chunk 22: `elapsed_sec=612.60`
- chunk 23: `elapsed_sec=606.42`

Most other chunks were roughly 5-20 seconds. That is why Stage 1 was flagged as a major sink, and why an explicit OpenAI SDK timeout was added.

## 4. Analyze Stage 4 Call Plan And Fallbacks

For LLM answering, I used the Stage 4 audit:

```bash
jq '.call_plan, .timing, .providers.openai.dimensions, .summary' src/data/refined_answers/bp1_82a34876/stage4_answering_audit.json
```

The call plan showed:

- 34 questions
- 3 variants: `default`, `risk`, `implementation`
- 39 estimated OpenAI generation batch calls
- 102 estimated refine calls
- 141 estimated total calls

But the timing showed:

- `batch_llm_call`: 13 calls
- `batch_length_mismatch`: 13 failures
- `single_llm_call`: 102 calls
- `refine_llm_call`: 102 calls

So every attempted batch failed and fell back to single-question mode. The log confirmed this pattern with lines like:

```text
[BATCH_CALL] length_mismatch provider=openai ... expected=3 got=not_list
[BATCH] ChatGPT ... status=failed_fallback_to_single
```

The cause was found in `src/tools/llm_answering.py`: the batch prompt asked for `{"answers":[...]}`, but then appended the single-answer schema from `_schema_structured()`, which told the model to return one object with keys like `answer`, `claims`, and `evidence_hints`. The model followed the single-answer shape, so the parser saw `got=not_list`.

## 5. Classify Each Stage

I classified each stage using the evidence above:

| Stage | Why It Was Classified That Way |
|---|---|
| `prepare_proposal_text` | 39 page-level vision/OCR operations, sequential; internal summary reported `elapsed_sec=215.71`. |
| `extract_facts_by_chunk` | Major wall-time mismatch plus two 10-minute chunk LLM calls. |
| `build_dimensions_from_facts` | Runner wall time was about 16 minutes, but the script lacked detailed internal per-call timing. |
| `generate_questions` | Five sequential dimension-question LLM calls; internal summary reported `elapsed_sec=103.15`. |
| `llm_answering` | Audit proved batch failure caused 13 wasted batch calls plus 102 single calls and 102 refine calls. |
| `post_processing` | Internal timing was under one second, so not a bottleneck. |
| `ai_expert_opinion` | One LLM call around 31 seconds, acceptable relative to other stages. |
| `generate_final_report` | Internal timing was near zero, so not a bottleneck. |

## 6. Optimization Conclusions

The first optimization was to fix the Stage 4 batch schema conflict so batches can succeed instead of always falling back.

The second optimization was to add an `OPENAI_TIMEOUT_SECONDS` setting, defaulting to 120 seconds, to OpenAI SDK clients in the long-running model-call stages. This prevents one stalled request from consuming 10 minutes before returning.

Further useful improvements would be:

- Add per-call timing logs to `build_dimensions_from_facts.py`.
- Parallelize independent dimensions where rate limits allow.
- Add fast/quality pipeline modes, such as `--variants default` or `--refine 0`.
- Add resumability or cache reuse for expensive LLM steps.

## 7. Before Vs After Optimization Numbers

Two same-file pipeline logs were compared:

- Baseline / before optimization: `output/20260508_224733_bp1_82a34876_82a34876_pipeline_log.txt`
- Optimized / after optimization: `output/20260513_101345_bp1_b47a91bf_b47a91bf_pipeline_log.txt`

Note: the labels above are based on the observed timings. The May 13 run is the faster run and shows the Stage 4 batch fix taking effect.

### End-To-End Wall Time

| Run | Pipeline Start | Pipeline End | Total Wall Time |
|---|---:|---:|---:|
| Before | `2026-05-08T16:47:33.051645+00:00` | `2026-05-09T02:52:24.516190+00:00` | 36,291.46 sec, about 10h 4m 51s |
| After | `2026-05-13T04:13:45.661356+00:00` | `2026-05-13T04:42:17.970313+00:00` | 1,712.31 sec, about 28m 32s |

Overall wall-time improvement:

- Saved: 34,579.15 sec, about 9h 36m 19s
- Reduction: 95.3%
- Speedup: about 21.2x faster

### Stage-Level Wall Time

| Stage | Before Wall Time | After Wall Time | Saved | Reduction |
|---|---:|---:|---:|---:|
| `prepare_proposal_text` | 217.08 sec | 203.92 sec | 13.16 sec | 6.1% |
| `extract_facts_by_chunk` | 32,922.81 sec | 175.04 sec | 32,747.77 sec | 99.5% |
| `build_dimensions_from_facts` | 962.04 sec | 32.54 sec | 929.50 sec | 96.6% |
| `generate_questions` | 103.84 sec | 60.74 sec | 43.10 sec | 41.5% |
| `llm_answering` | 2,053.57 sec | 1,209.92 sec | 843.65 sec | 41.1% |
| `post_processing` | 0.52 sec | 0.41 sec | 0.12 sec | 22.4% |
| `ai_expert_opinion` | 31.86 sec | 29.68 sec | 2.18 sec | 6.8% |
| `generate_final_report` | 0.07 sec | 0.07 sec | 0.00 sec | 5.6% |

### Script-Reported Internal Timing

For stages that reported internal `elapsed_sec`, the comparison was:

| Stage | Before Internal Time | After Internal Time | Saved | Reduction |
|---|---:|---:|---:|---:|
| `prepare_proposal_text` | 215.71 sec | 202.71 sec | 13.00 sec | 6.0% |
| `extract_facts_by_chunk` | 1,478.16 sec | 174.30 sec | 1,303.86 sec | 88.2% |
| `generate_questions` | 103.15 sec | 60.22 sec | 42.93 sec | 41.6% |
| `llm_answering` | 1,913.18 sec | 1,209.30 sec | 703.88 sec | 36.8% |
| `post_processing` | 0.431 sec | 0.318 sec | 0.113 sec | 26.2% |
| `ai_expert_opinion` | 31.319 sec | 29.465 sec | 1.854 sec | 5.9% |
| `generate_final_report` | 0.009 sec | 0.007 sec | 0.002 sec | 22.2% |

The `build_dimensions_from_facts` script did not yet print a total internal `elapsed_sec`, so its comparison uses runner wall time only.

### Stage 4 LLM Answering Detail

This is the cleanest evidence that the batch prompt fix worked.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| `chat_completion.count` | 217 | 145 | 72 fewer calls, 33.2% reduction |
| `chat_completion.total_sec` | 1,912.76 sec | 1,208.96 sec | 703.80 sec saved, 36.8% reduction |
| `batch_llm_call.count` | 13 | 37 | More batch calls were allowed to complete |
| `batch_length_mismatch.count` | 13 | 2 | 11 fewer failures, 84.6% reduction |
| `single_llm_call.count` | 102 | 6 | 96 fewer fallback calls, 94.1% reduction |
| `single_llm_call.total_sec` | 817.27 sec | 38.21 sec | 779.06 sec saved, 95.3% reduction |
| `refine_llm_call.count` | 102 | 102 | Same candidate refine count |
| `refine_llm_call.total_sec` | 1,000.39 sec | 739.96 sec | 260.43 sec saved, 26.0% reduction |
| `chat_completion.max_sec` | 91.76 sec | 68.54 sec | 23.22 sec lower max call time |

Before optimization, every attempted batch failed with `batch_length_mismatch` and the pipeline fell back to single-question calls. After optimization, only 2 batch calls failed, and fallback single calls dropped from 102 to 6. That is the main measured Stage 4 improvement.

### Important Interpretation

The largest end-to-end improvement is dominated by `extract_facts_by_chunk`, where wall time dropped from about 9h 8m to under 3m. The internal Stage 1 timing also improved sharply, from 1,478.16 sec to 174.30 sec. This suggests the newer run avoided the two very long stalled OpenAI calls seen in the baseline log and also avoided a large runner-vs-script timing gap.

Stage 4 improved more directly from the code change: the batch schema conflict was removed, batch success increased, and fallback single calls fell by 94.1%.

## 8. Enzyme Proposal Run: Remaining Bottlenecks

The recent enzyme proposal run was:

`output/20260513_093904_2-附件1_2025_0410新型高端酶制剂的智能化创制与应用_68e62a41_68e62a41_pipeline_log.txt`

This run completed successfully, but still took about 25m 51s end to end:

| Stage | Wall Time | Internal Time | Notes |
|---|---:|---:|---|
| `prepare_proposal_text` | 243.10 sec | 241.26 sec | 50 pages; many page-level vision calls at roughly 2-7 sec each. |
| `extract_facts_by_chunk` | 295.88 sec | 295.04 sec | 42 independent chunk LLM calls, mostly 4-10 sec each. |
| `build_dimensions_from_facts` | 23.27 sec | not reported | 5 dimension summaries. |
| `generate_questions` | 53.98 sec | 53.12 sec | 5 sequential question-generation calls. |
| `llm_answering` | 888.45 sec | 887.76 sec | Biggest bottleneck; 120 total model calls. |
| `post_processing` | 0.37 sec | 0.276 sec | Not a bottleneck. |
| `ai_expert_opinion` | 45.75 sec | 45.519 sec | One large expert-opinion LLM call. |
| `generate_final_report` | 0.08 sec | 0.010 sec | Not a bottleneck. |

Stage 4 was the main remaining problem:

- 30 questions
- 3 variants per question
- 30 batch generation calls
- 90 refine calls
- 120 total chat completions
- `batch_length_mismatch` was gone, so this was no longer a bad fallback problem; it was simply serialized high-quality work.

Dimension timings in Stage 4 were:

| Dimension | Time |
|---|---:|
| `team` | 158.18 sec |
| `objectives` | 165.99 sec |
| `strategy` | 172.75 sec |
| `innovation` | 185.03 sec |
| `feasibility` | 205.75 sec |

These dimensions are independent once the question set and dimension context are loaded. Inside each dimension, the three variants for a batch are also independent, and the three candidate refinements for a question are independent. That gives safe concurrency without lowering quality or removing any calls.

## 9. Added Concurrency Controls

To reduce runtime without reducing quality, two stages now have bounded concurrency.

### Stage 1: Parallel Chunk Fact Extraction

File: `src/tools/extract_facts_by_chunk.py`

Added:

- `STAGE1_CHUNK_WORKERS`, default `4`
- `[CONCURRENCY] stage1_chunks workers=... chunks=...` log line
- Parallel LLM extraction for chunks
- Ordered writing back to `raw_facts.jsonl`, sorted by original chunk index
- `chunk_workers` recorded in `stage1_audit.json`

Expected effect on the enzyme run:

- Before: 42 chunks took about 295 sec serialized.
- With 4 workers, wall time should move closer to the slowest waves of chunk calls rather than the sum of all calls.
- Expected practical range: roughly 75-120 sec, depending on API latency and rate limits.

### Stage 4: Parallel Answering Work

File: `src/tools/llm_answering.py`

Added:

- `STAGE4_DIM_WORKERS`, default `2`
- `STAGE4_VARIANT_WORKERS`, default `3`
- `STAGE4_REFINE_WORKERS`, default `3`
- Thread-safe timing aggregation for parallel LLM timing buckets
- `[CONCURRENCY] openai_dimensions ...` monitoring
- `[CONCURRENCY] batch_variants_parallel ...` monitoring
- `[CONCURRENCY] refine_parallel ...` monitoring
- Concurrency settings written into `stage4_answering_audit.json`

Quality is preserved because the same prompts, same variants, same batch generation, same refinement, and same post-processing are still used. The only change is that independent calls are allowed to run side by side.

Expected effect on the enzyme run:

- Before: Stage 4 took 887.76 sec.
- Dimension-level concurrency alone with 2 workers would likely reduce this by running the five dimensions in waves.
- Variant and refine concurrency should reduce each dimension further because each batch previously waited for variant calls and refinement calls serially.
- Expected practical range: roughly 3-6 minutes for Stage 4, depending on API limits.

### New Runtime Knobs

These can be placed in `.env` or exported in the shell:

```bash
STAGE1_CHUNK_WORKERS=4
STAGE4_DIM_WORKERS=2
STAGE4_VARIANT_WORKERS=3
STAGE4_REFINE_WORKERS=3
OPENAI_TIMEOUT_SECONDS=120
```

If rate limits appear, lower the worker counts first:

```bash
STAGE1_CHUNK_WORKERS=2
STAGE4_DIM_WORKERS=1
STAGE4_VARIANT_WORKERS=2
STAGE4_REFINE_WORKERS=2
```

## 10. Remaining Opportunities

The next safe target is `prepare_proposal_text`, especially page-level vision extraction. The enzyme run had 50 pages and vision calls dominated the 241 sec stage time. A careful implementation could extract text sequentially, then run page image conversion and vision description in a bounded worker pool while merging page results back in page order.

This was not changed in the current pass because PDF page extraction, image conversion, OCR, and page ordering are more sensitive than Stage 1 and Stage 4. It is still likely worth doing after validating the new Stage 1 and Stage 4 concurrency on one or two runs.
