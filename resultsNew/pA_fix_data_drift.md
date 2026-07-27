# pA fix + AI report-cache drift — evidence

## 1. pA matcher bug (FIXED)

`evaluate_cohens_kappa._find_record` matched `A_安海半导体…v1.4-YF.pdf` to `kt4__openai` because `kt4`'s source file `4.pdf` produced a degenerate 1-char key `4` that substring-matched `v1.**4**`. Fix: reject keys whose extension-stripped core is <4 chars or all-digits; exact match first; WARN on any fuzzy fallback. Post-fix, all 12 proposals resolve correctly (pA → pA__openai).

## 2. Stale-cache drift (NOT fixed — needs a decision)

`results/Dataset1` (paper baseline) is NOT reproducible from the current `src/data/reports`. Per-proposal AI overall score + verdict, stale file vs current cache (pA-corrected):

| file | stale pid | stale ai_ovr | stale verdict | current pid | current ai_ovr | current verdict |
|---|---|---|---|---|---|---|
| 1-附件1：心衰专病大模型BP.pdf | p1__openai | 0.68 | Y | p1__openai | 0.68 | N |
| 2-附件1：2025_0410新型高端酶制剂的智能化 | p2__openai | 0.69 | N | p2__openai | 0.69 | N |
| 3-附件1：基于非天然氨基酸调控的高效抗体结合蛋白. | p3__openai | 0.709 | Y | p3__openai | 0.694 | N |
| 4-附件1：（简介版）中国首创的第二代肿瘤治疗电场（ | p4__openai | 0.691 | Y | p4__openai | 0.717 | Y |
| 5-bp1.pdf | p5__openai | 0.658 | Y | p5__openai | 0.667 | N |
| 6-bp2.pdf | p6__openai | 0.685 | Y | p6__openai | 0.713 | Y |
| 7-04-1.（项目经理汇报）LPNP-mRNA免疫 | p7__openai | 0.726 | Y | p7__openai | 0.733 | N |
| 8-2025_10 Ebovir_LNP 拨投结合项 | p8__openai | 0.651 | N | p8__openai | 0.657 | N |
| A_安海半导体产业化方案v1.4-YF.pdf | pA__openai | 0.681 | Y | pA__openai | 0.68 | N |
| B_BMT技术（基石资本)v01.pptx | pB__openai | 0.714 | Y | pB__openai | 0.713 | Y |
| C_1.ZX学院商业计划书20210328.pptx | pC__openai | 0.626 | N | pC__openai | 0.618 | N |
| D_210614大瞬科技投决报告.pptx | pD__openai | 0.647 | N | pD__openai | 0.668 | N |

- Stale GO(Y) verdicts: **8/12**; current GO(Y) verdicts: **3/12**.
- Overall-ranking Spearman ρ (single rater): stale **0.589**, current buggy **0.662**, current fixed **0.5359**.
- Implication: the ρ=0.589 headline reflects an older AI cache, not the current one. The pA fix moves the current-cache single-rater ρ from 0.662→0.5359; it does not explain the gap to 0.589 (that is cache drift).
