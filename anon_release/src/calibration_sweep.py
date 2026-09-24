#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration sweep: find best INVEST_WEIGHT and verdict threshold
without any LLM calls. Reads stored invest_score_echo / qa_score_echo
from existing expert_reports JSONs and sweeps parameters to maximise
Spearman correlation with human expert scores.
"""
import json, math
from pathlib import Path

BASE   = Path(__file__).resolve().parent
DATA   = BASE / "src" / "data"
PIDS   = ["p1","p2","p3","p4","p5","p6","p7","p8","pA","pB","pC","pD"]
DIMS   = ["team","objectives","strategy","innovation","feasibility"]
# "objectives" in AI JSON, "objective" in human xlsx — note the mapping
DIM_DISPLAY = ["team","objective","strategy","advantages","feasibility"]

# ── Human scores from ExpertsEvaluations.xlsx (read once) ────────────
HUMAN_RAW = {
    # pid: {dim: 1-5 score, rank: 1-20, verdict: Y/N}
    "p1":  dict(team=5, objective=4, strategy=3, advantages=3, feasibility=2, rank=14, verdict="N"),
    "p2":  dict(team=4, objective=5, strategy=4, advantages=3, feasibility=3, rank=17, verdict="N"),
    "p3":  dict(team=4, objective=5, strategy=5, advantages=5, feasibility=5, rank=19, verdict="Y"),
    "p4":  dict(team=4, objective=5, strategy=4, advantages=3, feasibility=4, rank=18, verdict="N"),
    "p5":  dict(team=3, objective=4, strategy=4, advantages=4, feasibility=3, rank=15, verdict="N"),
    "p6":  dict(team=3, objective=4, strategy=4, advantages=3, feasibility=4, rank=15, verdict="N"),
    "p7":  dict(team=5, objective=5, strategy=4, advantages=5, feasibility=4, rank=20, verdict="Y"),
    "p8":  dict(team=3, objective=5, strategy=3, advantages=4, feasibility=3, rank=18, verdict="N"),
    "pA":  dict(team=4, objective=3, strategy=3, advantages=3, feasibility=3, rank=13, verdict="N"),
    "pB":  dict(team=5, objective=4, strategy=4, advantages=3, feasibility=5, rank=18, verdict="Y"),
    "pC":  dict(team=2, objective=3, strategy=4, advantages=4, feasibility=4, rank=13, verdict="N"),
    "pD":  dict(team=5, objective=5, strategy=4, advantages=4, feasibility=4, rank=18, verdict="Y"),
}

def human_norm(raw_score: int, is_rank=False) -> float:
    if is_rank:
        return round((raw_score - 1) / 19, 4)
    return round((raw_score - 1) / 4, 4)

# AI dim name → human dim name mapping
AI_TO_HUMAN_DIM = {
    "team": "team", "objectives": "objective", "strategy": "strategy",
    "innovation": "advantages", "feasibility": "feasibility",
}

# ── Load invest & QA scores from stored expert JSONs ─────────────────
def load_ai_raw():
    scores = {}
    for pid in PIDS:
        path = DATA / "expert_reports" / f"{pid}__openai" / "ai_expert_opinion.json"
        if not path.exists():
            print(f"[WARN] missing: {path}")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        dims = d.get("dimensions", {})
        scores[pid] = {}
        for ai_dim in DIMS:
            blk = dims.get(ai_dim, {})
            scores[pid][ai_dim] = {
                "invest": float(blk.get("invest_score_echo", 0.0) or 0.0),
                "qa":     float(blk.get("qa_score_echo",     0.0) or 0.0),
            }
        # load confidence from metrics.json (not affected by INVEST_WEIGHT)
        metrics_path = DATA / "refined_answers" / f"{pid}__openai" / "postproc" / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            scores[pid]["_conf"] = float((m.get("overall") or {}).get("overall_confidence", 0.0) or 0.0)
            scores[pid]["_dim_weights"] = (m.get("config_used") or {}).get("dimension_weight") or {}
        else:
            scores[pid]["_conf"] = 0.0
            scores[pid]["_dim_weights"] = {}
    return scores

# ── Spearman correlation ──────────────────────────────────────────────
def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    def ranks(v):
        sv = sorted(enumerate(v), key=lambda x: x[1])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sv[j+1][1] == sv[j][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j+1):
                r[sv[k][0]] = avg_rank
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx)/n; my = sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx  = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    dy  = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return round(num / (dx * dy + 1e-12), 4) if dx * dy > 1e-12 else 0.0

def concordance(xs, ys):
    pairs = [(xs[i], ys[i], xs[j], ys[j]) for i in range(len(xs)) for j in range(i+1, len(xs))]
    conc = sum(1 for a, b, c, d in pairs if (a - c) * (b - d) > 0)
    disc = sum(1 for a, b, c, d in pairs if (a - c) * (b - d) < 0)
    total = conc + disc
    return round(conc / total, 4) if total else 0.0

# ── Main sweep ───────────────────────────────────────────────────────
def sweep(ai_raw):
    human_rank  = [human_norm(HUMAN_RAW[p]["rank"], is_rank=True) for p in PIDS]
    human_verdict = [HUMAN_RAW[p]["verdict"] for p in PIDS]

    results = []
    weights = [round(w * 0.05, 2) for w in range(0, 21)]   # 0.00 → 1.00
    thresholds = [round(t * 0.05, 2) for t in range(9, 17)] # 0.45 → 0.80

    for iw in weights:
        for vt in thresholds:
            ai_overall = []
            verdicts   = []
            for pid in PIDS:
                dims_raw = ai_raw.get(pid, {})
                conf     = dims_raw.get("_conf", 0.0)
                dw       = dims_raw.get("_dim_weights", {})
                blended  = {}
                for ai_dim in DIMS:
                    inv = dims_raw.get(ai_dim, {}).get("invest", 0.0)
                    qa  = dims_raw.get(ai_dim, {}).get("qa",     0.0)
                    blended[ai_dim] = iw * inv + (1 - iw) * qa

                # weighted overall (equal weights if none configured)
                scores_list  = [blended[d] for d in DIMS]
                weights_list = [float(dw.get(d, 1.0)) for d in DIMS]
                total_w      = sum(weights_list) or 1.0
                overall      = sum(s * w for s, w in zip(scores_list, weights_list)) / total_w
                ai_overall.append(round(overall, 4))

                # verdict: GO if overall >= vt OR conf >= vt (OR logic, separate thresholds)
                innov = blended.get("innovation", 0.0)
                feas  = blended.get("feasibility", 0.0)
                if overall >= vt or conf >= vt:
                    v = "Y"
                elif overall < 0.40 or innov < 0.30 or feas < 0.30:
                    v = "N"
                else:
                    v = "N"  # HOLD → treat as N for accuracy
                verdicts.append(v)

            sp    = spearman(human_rank, ai_overall)
            conc  = concordance(human_rank, ai_overall)
            vacc  = sum(1 for h, a in zip(human_verdict, verdicts) if h == a) / len(PIDS)
            ai_range = max(ai_overall) - min(ai_overall)

            results.append(dict(
                invest_weight=iw, verdict_threshold=vt,
                spearman=sp, concordance=conc,
                verdict_acc=round(vacc, 4), ai_range=round(ai_range, 4),
                ai_overall=ai_overall,
            ))

    return results


def sweep_verdict_thresholds(ai_raw, iw: float):
    """2-D sweep of separate overall and confidence thresholds (OR logic) at fixed IW."""
    human_verdict = [HUMAN_RAW[p]["verdict"] for p in PIDS]

    # Build ai scores at this IW
    ai_scores = {}
    for pid in PIDS:
        dims_raw = ai_raw.get(pid, {})
        conf = dims_raw.get("_conf", 0.0)
        dw   = dims_raw.get("_dim_weights", {})
        blended = {d: iw * dims_raw.get(d, {}).get("invest", 0.0) +
                      (1 - iw) * dims_raw.get(d, {}).get("qa", 0.0)
                   for d in DIMS}
        scores_list  = [blended[d] for d in DIMS]
        weights_list = [float(dw.get(d, 1.0)) for d in DIMS]
        total_w = sum(weights_list) or 1.0
        overall = sum(s * w for s, w in zip(scores_list, weights_list)) / total_w
        ai_scores[pid] = {"overall": round(overall, 4), "conf": round(conf, 4)}

    def _mid_candidates(vals):
        sv = sorted(vals)
        cands = [0.0]
        for i in range(len(sv) - 1):
            cands.append(round((sv[i] + sv[i+1]) / 2, 4))
        cands.append(1.1)
        return cands

    t_o_cands = _mid_candidates([ai_scores[p]["overall"] for p in PIDS])
    t_c_cands = _mid_candidates([ai_scores[p]["conf"] for p in PIDS])

    best_rows = []
    for t_o in t_o_cands:
        for t_c in t_c_cands:
            preds = ["Y" if (ai_scores[p]["overall"] >= t_o or ai_scores[p]["conf"] >= t_c)
                     else "N" for p in PIDS]
            acc = sum(1 for p, v in zip(PIDS, preds) if HUMAN_RAW[p]["verdict"] == v) / len(PIDS)
            y_pred = sum(1 for v in preds if v == "Y")
            y_corr = sum(1 for p, v in zip(PIDS, preds) if v == "Y" and HUMAN_RAW[p]["verdict"] == "Y")
            best_rows.append((acc, y_corr, t_o, t_c, y_pred))

    best_rows.sort(reverse=True)
    print(f"\n--- 2-D verdict threshold sweep (OR logic, IW={iw}) ---")
    print(f"{'Acc':6s}  {'Y_corr/4':8s}  {'Y_pred':6s}  {'t_overall':10s}  {'t_conf':8s}")
    print("-" * 55)
    seen = set()
    for acc, yc, to, tc, yp in best_rows:
        k = (acc, yc, yp)
        if k not in seen:
            seen.add(k)
            print(f"{acc:.2%}  {yc}/4        {yp:6d}  {to:.4f}      {tc:.4f}")
        if len(seen) >= 8:
            break

    best_acc, best_yc, best_to, best_tc, best_yp = best_rows[0]
    return {"t_overall": best_to, "t_conf": best_tc, "verdict_acc": best_acc}


def print_results(results):
    # Sort by spearman desc
    top = sorted(results, key=lambda x: -x["spearman"])[:20]
    print(f"\n{'IW':5s}  {'VT':5s}  {'Spear':6s}  {'Conc':6s}  {'VAcc':5s}  {'AIrng':6s}")
    print("-" * 50)
    for r in top:
        print(f"{r['invest_weight']:5.2f}  {r['verdict_threshold']:5.2f}  "
              f"{r['spearman']:+.4f}  {r['concordance']:.4f}  "
              f"{r['verdict_acc']:.2%}  {r['ai_range']:.4f}")

    best_sp = top[0]

    print(f"\n★ Best Spearman:  IW={best_sp['invest_weight']}  VT={best_sp['verdict_threshold']}  "
          f"spearman={best_sp['spearman']:+.4f}  verdict_acc={best_sp['verdict_acc']:.0%}")

    print(f"\nPer-dimension Spearman at best overall setting (IW={best_sp['invest_weight']}):")
    ai_raw = load_ai_raw()
    iw = best_sp["invest_weight"]
    for human_dim, ai_dim in [("team","team"),("objective","objectives"),("strategy","strategy"),
                               ("advantages","innovation"),("feasibility","feasibility")]:
        human_scores = [human_norm(HUMAN_RAW[p][human_dim]) for p in PIDS]
        ai_scores    = [iw * ai_raw[p][ai_dim]["invest"] + (1-iw) * ai_raw[p][ai_dim]["qa"]
                        for p in PIDS if p in ai_raw]
        sp = spearman(human_scores, ai_scores)
        print(f"  {human_dim:12s}: {sp:+.4f}")

    return best_sp


if __name__ == "__main__":
    print("Loading AI scores...")
    ai_raw = load_ai_raw()
    missing = [p for p in PIDS if p not in ai_raw]
    if missing:
        print(f"[WARN] Missing expert JSONs for: {missing}")

    print(f"Loaded {len(ai_raw)}/12 proposals. Running sweep...")
    results = sweep(ai_raw)
    best = print_results(results)

    # 2-D verdict threshold search at best IW (independent of Spearman)
    best_vt = sweep_verdict_thresholds(ai_raw, best["invest_weight"])

    # Save full results
    out = BASE / "DeltaProjectDatasets" / "Results" / "calibration_sweep.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull sweep saved → {out}")
    print(f"\nRecommended settings:")
    print(f"  INVEST_WEIGHT     = {best['invest_weight']}  (maximises Spearman)")
    print(f"  verdict logic     = OR  (score >= t_overall OR conf >= t_conf)")
    print(f"  t_overall         = {best_vt['t_overall']}  (maximises verdict accuracy)")
    print(f"  t_conf            = {best_vt['t_conf']}  (maximises verdict accuracy)")
    print(f"  verdict_acc       = {best_vt['verdict_acc']:.0%}")
