#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build prompt variant C — a GENUINE ablation of the "AI-powered" innovation cap.

Why C is needed
---------------
The pre-built variant B (`canonical/ablation_ai_cap.json`, sha 27977547…) removes
RULE 1 and the I-C HARD CAP line, but its own provenance records that cap
language SURVIVES in B:

  * EXAMPLES bullet: "→ AI heart failure decision support system (AI IS the
    product) → I-C, score 0.42-0.56" — this restates the cap verbatim AND
    describes p1's exact archetype, i.e. the very test case the ablation turns on.
  * RULE 2 clause "then DO NOT apply the I-C cap" — a dangling reference to a
    rule that no longer exists in B.
  * I-C header parenthetical "(INCLUDES most AI proposals)" — a directive steer.

So B ablates the rule's *statement* while leaving an instruction that applies the
same rule to the same proposal. A null result under B would be uninterpretable.

What C removes (on top of B)
----------------------------
  1. The EXAMPLES bullet that restates the cap and names p1's archetype.
  2. The "(INCLUDES most AI proposals)" steer in the I-C header.
  3. RULE 2's dangling "then DO NOT apply the I-C cap" clause — reworded to
     preserve RULE 2's substance (AI-as-design-tool → judge the physical output
     on its merits) without referencing a cap that no longer exists.

What C deliberately RETAINS
---------------------------
  * The I-C category bullet "AI/ML/software applied to any existing domain
    (clinical, manufacturing, logistics)". This is the I-C *taxonomy*, not the
    cap. Removing it would ablate the category itself and confound the study.
  * Every other section byte-identical to A/B (team, objectives, strategy,
    feasibility, preamble, output schema).

Output: canonical/b4_prompt_variants.json
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = json.loads((ROOT / "canonical" / "ablation_ai_cap.json").read_text())["prompts"]
A, B = SRC["A_production"]["text"], SRC["B_cap_removed"]["text"]

EDITS = [
    ("examples_bullet_restating_cap",
     "  → AI heart failure decision support system (AI IS the product) → I-C, score 0.42-0.56\n",
     ""),
    ("i_c_header_steer",
     "Category I-C (score 0.40–0.60) — application-layer innovation (INCLUDES most AI proposals):",
     "Category I-C (score 0.40–0.60) — application-layer innovation:"),
    ("rule2_dangling_cap_reference",
     "but the PRIMARY commercialized product is the PHYSICAL OUTPUT (not the AI software), "
     "then DO NOT apply the I-C cap. Evaluate the biological/chemical/physical innovation "
     "on its own merits using the I-A through I-D categories below.",
     "but the PRIMARY commercialized product is the PHYSICAL OUTPUT (not the AI software), "
     "evaluate the biological/chemical/physical innovation on its own merits using the "
     "I-A through I-D categories below."),
]


def main() -> int:
    C = B
    applied = []
    for name, old, new in EDITS:
        if old not in C:
            print(f"[FAIL] edit '{name}' did not match prompt B — aborting, no file written.")
            print(f"       looked for: {old[:120]!r}")
            return 2
        if C.count(old) != 1:
            print(f"[FAIL] edit '{name}' matched {C.count(old)}x (expected 1) — aborting.")
            return 2
        C = C.replace(old, new)
        applied.append({"edit": name, "removed_chars": len(old) - len(new)})

    # --- invariants: only the INNOVATION section may differ between B and C ---
    def sections(t):
        out, cur, buf = {}, "<preamble>", []
        for line in t.split("\n"):
            if line.startswith("=== ") and line.endswith(" ==="):
                out[cur] = "\n".join(buf)
                cur, buf = line, []
            else:
                buf.append(line)
        out[cur] = "\n".join(buf)
        return out

    sb, sc = sections(B), sections(C)
    changed = [k for k in sb if sb[k] != sc.get(k)]
    identical = [k for k in sb if sb[k] == sc.get(k)]
    if changed != ["=== INNOVATION ==="]:
        print(f"[FAIL] C differs from B outside INNOVATION: {changed}")
        return 2

    # --- residual cap language check ---
    CAP_MARKERS = ["NO EXCEPTIONS", "HARD CAP", "RULE 1", "0.42-0.56",
                   "I-C cap", "INCLUDES most AI proposals"]
    residual = {m: {"A": m in A, "B": m in B, "C": m in C} for m in CAP_MARKERS}
    still_in_C = [m for m, v in residual.items() if v["C"]]
    if still_in_C:
        print(f"[FAIL] cap language still present in C: {still_in_C}")
        return 2

    payload = {
        "study": "b4_ablation_prompt_variants",
        "variants": {
            "A_production": {"sha256": hashlib.sha256(A.encode()).hexdigest(),
                             "chars": len(A), "text": A,
                             "role": "condition A — unmodified production prompt (cap present)"},
            "B_cap_removed": {"sha256": hashlib.sha256(B.encode()).hexdigest(),
                              "chars": len(B), "text": B,
                              "role": ("condition B — pre-built variant; removes RULE 1 + HARD CAP "
                                       "line ONLY. Cap language survives (see contamination)."),
                              "contamination": SRC["cap_language_retained_in_B"]},
            "C_cap_fully_removed": {"sha256": hashlib.sha256(C.encode()).hexdigest(),
                                    "chars": len(C), "text": C,
                                    "role": ("condition C — genuine cap ablation; all directive cap "
                                             "language and the p1-archetype worked example removed"),
                                    "edits_on_top_of_B": applied,
                                    "deliberately_retained": [
                                        "I-C bullet 'AI/ML/software applied to any existing domain "
                                        "(clinical, manufacturing, logistics)' — this is the I-C "
                                        "taxonomy, not the cap; removing it would ablate the category."],
                                    "sections_changed_vs_B": changed,
                                    "sections_byte_identical_vs_B": identical},
        },
        "residual_cap_marker_check": residual,
        "chars": {"A": len(A), "B": len(B), "C": len(C),
                  "A_minus_B": len(A) - len(B), "B_minus_C": len(B) - len(C),
                  "A_minus_C": len(A) - len(C)},
    }
    out = ROOT / "canonical" / "b4_prompt_variants.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out}")
    print(f"  A: {len(A)} chars  sha {payload['variants']['A_production']['sha256'][:16]}…")
    print(f"  B: {len(B)} chars  sha {payload['variants']['B_cap_removed']['sha256'][:16]}…")
    print(f"  C: {len(C)} chars  sha {payload['variants']['C_cap_fully_removed']['sha256'][:16]}…")
    print(f"  edits B→C: {[a['edit'] for a in applied]}")
    print(f"  sections changed vs B: {changed}")
    print("  residual cap markers:")
    for m, v in residual.items():
        print(f"    {m:28} A={v['A']!s:5} B={v['B']!s:5} C={v['C']!s:5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
