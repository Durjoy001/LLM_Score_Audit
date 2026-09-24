# INTERNAL NOTE — groundedness audit: taxonomy mismatch & missing per-claim records

**Written 2026-07-28. Internal record only — not for the paper, not for review
distribution.** Deliberately excluded from the anonymous review bundle whitelist.

Two independent problems with the RQ2 groundedness audit, recorded so they are not
rediscovered later. Neither changes any published number; both concern
reproducibility and the audit's paper trail.

---

## 1. The committed script does not implement the published methodology

`src/tools/evaluate_groundedness.py` and methodology §5.2 (both papers, and
`generate_full_project_report.py` §21/§24) describe **different procedures**.

### 1a. Middle label is a different construct

| | Script | Published methodology |
|---|---|---|
| Label | `inferred` | `partially supported` |
| Definition | "reasonably implied by context but not explicitly stated" (script docstring L7; prompt L167–168, L173) | "based on proposal evidence but incomplete, overgeneralized, approximate, or only partly justified" |

*Implied but unstated* and *stated but incomplete* are not the same category, and a
claim can easily fall in one and not the other. Both feed an identical half-credit
formula — script L238–242 computes `(n_supported + 0.5·n_inferred)/total`, the paper
`(N_supported + 0.5·N_partial)/N_eligible` — so **the arithmetic matches while the
semantics do not**. That makes the mismatch easy to miss on a casual read.

### 1b. No statement-type classification stage, so the paper's funnel cannot come from this script

Table 3 reports a four-way classification (verifiable facts / numerical inferences /
professional inferences / evaluator judgments) and a denominator funnel:

```
188 classified report statements
 −114 professional inferences   (excluded)
 − 41 evaluator judgments       (excluded)
 =  33 eligible factual/numerical claims
```

The script has **no such stage**. `extract_claims()` (L108–133) applies a single-pass
inclusion/exclusion filter in the extraction prompt: it excludes evaluation outputs
(score, ranking, verdict, confidence) and absence-framed evaluator judgments
("lacks X", "does not provide Y", "no evidence of"). That overlaps the paper's
*evaluator judgments* bucket, but:

- it never defines or excludes **professional inferences** — the paper's *largest*
  exclusion bucket at 114 of 188;
- it never distinguishes **verifiable facts** from **numerical inferences** (the paper
  states the denominator holds "33 verifiable facts and no numerical inferences");
- it emits **no counts for excluded statements**. `summarise()` (L225–253) returns
  `total_claims` = extracted claims only. There is no 188 denominator and no 114/41
  tally anywhere in its output schema.

So the Table 3 funnel is not producible from this script's outputs. Whatever generated
the published numbers was a different or modified procedure — consistent with the
papers' own wording ("labels were model-assisted rather than independently
double-annotated"; "a manual spot-check of all 33 eligible claims").

### 1c. Minor: framing

Script names the third metric `hallucination_rate` (L13, L243); the papers use
"unsupported-claim rate" throughout and avoid the hallucination framing.

---

## 2. The per-claim annotations were never persisted

The script's declared output path is
`src/data/evaluations/runs/<run_id>/groundedness_result.json`. **That directory was
never created**, in either project.

Searched 2026-07-28, read-only:

| Surface | Result |
|---|---|
| `<REPO_ROOT>/src/data/evaluations/` | only `human/` (9 xlsx). No `runs/` |
| Upstream project `src/data/evaluations/` | identical — only `human/`. `src/data/results/` empty |
| `groundedness_result*` anywhere on disk | 0 files |
| `claim_id` content search, unlimited depth, 5,755 upstream files | 2 hits: the script, and `results/Dataset1/evaluation_report.html` — both are *prompt-schema documentation*, not data |
| Git history — 3 commits (thesis repo) + 58 commits (upstream) | only path ever matching `ground`/`claim` is the script itself. Nothing committed then deleted |
| All 10 HTML reports + 8 result PDFs | aggregate counts and methodology prose only; zero per-claim rows |
| `.xlsx`/`.csv`/`.docx`/`.ipynb`/`.tex` across `Documents/Thesis` + `Documents/Ebovir` | nothing related |

The two project trees are near-duplicates (`results/Dataset1/evaluation_report.html`
is byte-identical between them), so searching upstream added little.

### What survives

Only the aggregate, consistent across three places:

- `generate_full_project_report.py` L1235–1237 — `25 / 4 / 4`, strict 75.8 %
  [59.0, 87.2], lenient 81.8 % [65.6, 91.4]; L1664 flags the audit as a pilot
- `research_paper.pdf` Table 3
- `LLM_Assisted_Venture_Screeing.pdf` §6.2 and Figure 4 (a three-bar count chart)

Full published figures: 188 statements → 114 + 41 excluded → 33 eligible; 25 supported,
4 partially supported, 4 unsupported; strict 0.758 [0.590, 0.872], lenient 0.818
[0.656, 0.914], unsupported rate 0.121 [0.048, 0.273]; 11/12 proposals had eligible
claims.

---

## 3. Decisions taken (2026-07-28)

1. **Publish the aggregate only.** No per-claim illustrative table in the paper. A
   regenerated sample would not be the sample whose 25/4/4 counts are published —
   the same "new sample, not a correction" rule `canonical/README.md` applies to the
   score sets. No reconstruction, no re-run.
2. **Anonymous review bundle built but NOT published**, precisely because it would
   ship `evaluate_groundedness.py` alongside a methodology section it does not
   implement — an avoidable provenance contradiction in a paper about provenance.
   Bundle location: `../anon_review_bundle` (25 files, verified clean, no `.git`).

## 4. Open items, if this is ever revisited

- Reconcile script and methodology — either implement the four-way classification
  stage and rename `inferred` → `partially supported` with the paper's definition, or
  amend §5.2 to describe what the script actually does. Until then the two must not
  ship together.
- If the audit is ever re-run, persist `groundedness_result.json` **and** commit it;
  the whole problem here is an unwritten intermediate.
- Any future re-run is a new sample and must be reported as such, not as a correction
  to 25/4/4.
