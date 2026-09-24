#!/usr/bin/env python3
"""B6 structural-failure verification (FinNLP 2026 repro package).
Reads scores.csv; asserts every published structural claim; writes
b6_threshold_sweep.csv. Run: python b6_verify.py scores.csv"""
import sys, pandas as pd, numpy as np

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "scores.csv")
y, r, c = df.label.values, df.score.values, df.confidence.values
TR, TC = 0.725, 0.811

assert len(df) == 141 and int(y.sum()) == 24, "sample mismatch"
assert r.max() < TR, f"score branch reachable: max={r.max()}"
assert c.max() < TC, f"confidence branch reachable: max={c.max()}"
for pred in [(r>=TR)|(c>=TC), (r>=TR)&(c>=TC), r>=TR, c>=TC]:
    assert pred.sum() == 0, "a rule variant selected something"
rule_pred = ((r >= TR) | (c >= TC)).astype(int)
assert abs((rule_pred == y).mean() - 117 / 141) < 1e-9, "accuracy != all-negative baseline"
print(f"OK: max score {r.max():.3f} < {TR} (gap {TR-r.max():.3f}); "
      f"max confidence {c.max():.3f} < {TC} (gap {TC-c.max():.3f}); "
      f"all four rule variants select 0/141; all-negative accuracy {117/141:.3f}")

rows = []
for t in np.round(np.arange(0.40, 0.86, 0.005), 3):
    ps, pc = (r >= t), (c >= t)
    rows.append({"threshold": t,
                 "score_selected": int(ps.sum()), "score_TP": int((ps & (y == 1)).sum()),
                 "conf_selected": int(pc.sum()), "conf_TP": int((pc & (y == 1)).sum())})
pd.DataFrame(rows).to_csv("b6_threshold_sweep.csv", index=False)
print("wrote b6_threshold_sweep.csv")
