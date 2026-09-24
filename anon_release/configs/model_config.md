# Model configuration

Every value below was read out of the code in `src/`. Where the code does not
set a value, the row says **TODO: confirm** rather than guessing — the request
sampled at that point uses the provider's own default, which is not pinned by
this repository.

## Provider and model

| Setting | Value in code | Where |
| --- | --- | --- |
| Provider (default) | `openai` | `llm_provider.py::active_provider()` — `PROVIDER` env var, default `"openai"` |
| Model (OpenAI) | `gpt-4o-mini` | `llm_provider.py::default_model()` — `OPENAI_MODEL` env var, default `"gpt-4o-mini"` |
| Model (Gemini, cross-provider check only) | `gemini-2.5-flash` | `llm_provider.py::default_model()` — `GEMINI_MODEL` env var |
| Model (DeepSeek, adapter present, unused in main run) | `deepseek-chat` | `llm_provider.py::default_model()` |
| Vision model (Stage 0 page images) | inherits `default_model(PROVIDER)` | `prepare_proposal_text.py` — `VISION_MODEL` env var |
| **Model snapshot / dated version** | **TODO: confirm** | No dated snapshot (e.g. `gpt-4o-mini-YYYY-MM-DD`) is pinned anywhere in the code. Only the floating alias is requested, so the served snapshot is whatever the provider routed the alias to at run time. |

## Sampling parameters

`temperature` is set per call site; there is no single global value.

| Stage / call site | temperature | max_tokens |
| --- | --- | --- |
| Stage 1 fact extraction | provider-adapter default `0.0` | adapter default `1000` |
| Stage 2 dimension builder | provider-adapter default `0.0` | adapter default `1000` |
| Stage 3 question generation | provider-adapter default `0.0` | adapter default `1000` |
| Stage 4 answering — `default` variant | `0.25` | `2200` (CLI `--max_tokens`) |
| Stage 4 answering — `risk` variant | `0.35` | `2200` |
| Stage 4 answering — `implementation` variant | `0.30` | `2200` |
| Stage 4 self-review / refine pass | `0.2` | `600` |
| Stage 6 investment score | `0.0` | `600` |
| Stage 6 dimension opinion | `0.25` | `2600` |
| Stage 8 rubric scoring | `0.2` (CLI `--temperature`) | `900` |
| Provider health probe | `0.0` | `2` |

Sources: `llm_answering.py::TEMP_BY_VARIANT` and its argparse defaults;
`ai_expert_opinion.py` call sites; `generate_llm_scores.py::call_openai_chat()`;
`llm_provider.py::chat_completion()` signature defaults
(`temperature=0.0, max_tokens=1000`).

| Setting | Value |
| --- | --- |
| **top_p** | **TODO: confirm** — `top_p` is never sent. It does not appear in any request body built in `llm_provider.py`, so the provider default applies and is not recorded by this code. |

## Determinism

| Setting | Value in code | Where |
| --- | --- | --- |
| `seed` (Stage 6 investment score) | `42` | `ai_expert_opinion.py` |
| `seed` (Stage 4 answering) | CLI `--seed`, default `None` | `llm_answering.py`; when supplied it seeds Python's `random` and is forwarded to the API body |
| `seed` forwarding | included in the OpenAI-compatible request body only when not `None` | `llm_provider.py::_call_openai_compatible()` |
| `response_format` | `{"type": "json_object"}` when JSON mode is requested | `llm_provider.py` |
| Cross-validation seed | `42`, 5 stratified folds | `calibrate_kickstarter_threshold.py` (`SEED`, `K_FOLDS`) |
| Bootstrap seed | `42`, 1000 resamples, percentile method | `compute_spearman_bootstrap_cis.py` |

The OpenAI `seed` parameter is best-effort, not a determinism guarantee. The
scoring stage is therefore treated as stochastic throughout; see the README
section "What cannot be reproduced and why".

## Retry logic

| Call site | Policy |
| --- | --- |
| Stage 4 answering (`llm_answering.py::_with_retry`) | `max_tries=4`, exponential backoff `0.8 * 2**i` seconds plus up to `0.2 s` of jitter |
| Stage 4 JSON mode | on failure, one fallback retry in plain-text mode, itself wrapped in `_with_retry` |
| Stage 6 (`ai_expert_opinion.py::_call_llm`) | `max_retries=3`, backoff `1.8 ** attempt` seconds; raises `RuntimeError` after the last attempt |
| Stage 1 (`extract_facts_by_chunk.py`) | one retry on invalid JSON with a stricter JSON instruction; one extra "dense" retry when a long chunk yields too few facts |
| Stage 6 overall | falls back to local, non-LLM composition if the LLM call fails |
| HTTP timeouts | connect `12 s`, read `120 s` (`HTTP_TIMEOUT_CONNECT` / `HTTP_TIMEOUT_READ` env vars) |

## Not configured in code

The following are read from a `.env` file that is deliberately **not** included
in this bundle: `OPENAI_API_KEY`, `GEMINI_API_KEY`. Set them in the environment
before running any stage that issues API calls. No metric in
"What can be reproduced" requires them.
