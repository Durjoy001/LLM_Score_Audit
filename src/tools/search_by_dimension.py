# -*- coding: utf-8 -*-
"""
Semantic Search by Dimension (v2025.12 ProClean; strong-query / early-stop / robust-io / diagnostics)
Keeps downstream compatibility for evidence/* file names and structures; adds *_queries.json query diagnostics.
"""
import os, sys, re, json, time, argparse
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter, defaultdict
from dotenv import load_dotenv

load_dotenv()
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
DATA_DIR = SRC_ROOT / "data"
PROPOSAL_DIR = DATA_DIR / "extracted"
EVIDENCE_ROOT = DATA_DIR / "evidence"
CONFIG_DIR = DATA_DIR / "config"
PARSED_DIR = DATA_DIR / "parsed"
EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))

from backend.utils.model_selector import get_llm_client
from backend.retrievers.web_search import simple_search

llm = get_llm_client()
client = llm["client"]; model_name = llm["model_name"]; provider = llm["provider"]
print(f"QueryGen using {provider.upper()} model: {model_name}")

# ===== Basic parameters =====
IGNORE_DIMS = {"proposal_id", "generated_time", "chunk_count", "coverage_estimate", "meta", "doc_meta", "run_meta"}
FIRST_ROUND_N = 4
MAX_RESULTS_PER_QUERY = 5
MAX_QUERIES_PER_DIM = 14
SLEEP_BETWEEN_QUERIES = 0.8
MIN_SAVE_EVIDENCE = 1

# Per-question early-stop threshold for high-authority hits.
EARLY_STOP_ACADEMIC = {"strategy":4, "objectives":4, "feasibility":4, "innovation":3, "team":3}
# Dimension-level soft threshold; once reached, later questions use fewer queries.
DIM_SOFT_EARLY_STOP = {"strategy":10, "objectives":10, "feasibility":10, "innovation":8, "team":8}

ACADEMIC_SITES = [
    "pubmed.ncbi.nlm.nih.gov","pmc.ncbi.nlm.nih.gov","nature.com","sciencedirect.com",
    "nih.gov","who.int","ema.europa.eu","fda.gov","clinicaltrials.gov",
    "thelancet.com","bmj.com","cell.com","springer.com","biorxiv.org","medrxiv.org","arxiv.org","nejm.org",
    "nmpa.gov.cn"
]

DIM_HINTS = {
    "team": ["founder","leadership","management team","operator","advisor","affiliation","track record"],
    "objectives": ["objective","milestone","KPI","outcome","target user","market need","impact metric"],
    "strategy": ["business model","go-to-market","pricing","procurement","partnership","channel strategy","regulatory pathway"],
    "innovation": ["competitive advantage","novel product","service innovation","platform","IP","benchmark","differentiation"],
    "feasibility": ["budget","unit economics","scale-up","operations","capacity","compliance","supply chain","implementation"]
}

def clean_query(q: str, max_len=280):
    q = re.sub(r"[，。？：；！、]", " ", q or "")
    q = re.sub(r"\s+", " ", q.strip())
    return q[:max_len]

def uniq(seq):
    seen, out = set(), []
    for x in seq:
        sx = str(x or "").strip()
        if not sx: continue
        lx = sx.lower()
        if lx not in seen:
            out.append(sx); seen.add(lx)
    return out

def _extract_bracket_block(t: str, lch: str, rch: str):
    s, e = t.find(lch), t.rfind(rch)
    if s != -1 and e != -1 and e > s:
        frag = t[s:e+1]
        return frag.replace("“","\"").replace("”","\"")
    return None

def safe_json_loads(text: str):
    t = (text or "").strip()
    if "```" in t: t = t.replace("```json","").replace("```","").strip()
    for frag in (t, _extract_bracket_block(t,"[","]"), _extract_bracket_block(t,"{","}")):
        if not frag: continue
        try:
            obj = json.loads(frag)
            if isinstance(obj, list): return obj
            if isinstance(obj, dict) and isinstance(obj.get("questions"), list): return obj["questions"]
        except Exception: continue
    return []

def collect_entities_numbers_terms(dim_content: dict):
    ents, nums, key_terms = [], [], []
    try:
        people = dim_content.get("entities", {}).get("people", []) or []
        ents += [p.get("name","") for p in people if isinstance(p, dict)]
        ents += dim_content.get("entities", {}).get("orgs", []) or []
    except Exception: pass
    ents = [e for e in ents if isinstance(e,str) and e.strip()]

    try:
        nums = [str(n.get("value","")) for n in (dim_content.get("numbers") or [])
                if isinstance(n, dict) and n.get("value")]
    except Exception: pass
    nums = [n for n in nums if n]

    try:
        kt = dim_content.get("key_terms") or []
        if isinstance(kt, list): key_terms = [str(k) for k in kt if k]
    except Exception: pass
    return ents[:8], nums[:6], key_terms[:12]

# ---- Fallback templates ----
FALLBACK_TEMPLATES = {
    "team": [
        '"{PERSON}" founder leadership track record 2018..2026',
        '"{PERSON}" executive profile project experience 2018..2026',
        '"{ORG}" leadership team case study 2018..2026',
        '"{ORG}" company profile partners 2018..2026'
    ],
    "strategy": [
        '"{KEY}" business model go-to-market benchmark 2019..2026',
        '"{KEY}" market analysis competitors pricing 2019..2026',
        '"{KEY}" procurement adoption channel strategy 2019..2026',
        '"{KEY}" regulation compliance guidance 2019..2026'
    ],
    "objectives": [
        '"{KEY}" target market customer need KPI 2019..2026',
        '"{KEY}" outcome metrics benchmark 2019..2026',
        '"{KEY}" impact measurement objectives 2019..2026'
    ],
    "innovation": [
        '"{KEY}" competitive advantage benchmark 2021..2026',
        '"{KEY}" IP patent landscape 2021..2026',
        '"{KEY}" product innovation differentiation 2021..2026',
        '"{KEY}" alternatives competitors 2021..2026'
    ],
    "feasibility": [
        '"{KEY}" implementation budget operations 2019..2026',
        '"{KEY}" unit economics cost model 2019..2026',
        '"{KEY}" supply chain capacity scale-up 2019..2026',
        '"{KEY}" compliance risk mitigation 2019..2026'
    ]
}

def _inject_fallbacks(dimension: str, entities: list, keywords: list):
    toks = uniq((entities or []) + (keywords or []))
    outs = []
    for tpl in FALLBACK_TEMPLATES.get(dimension.lower(), []):
        if "{PERSON}" in tpl and entities:
            outs.append(tpl.replace("{PERSON}", entities[0]))
        elif "{ORG}" in tpl and len(entities) > 1:
            outs.append(tpl.replace("{ORG}", entities[1]))
        elif "{KEY}" in tpl and toks:
            outs.append(tpl.replace("{KEY}", toks[0]))
    return uniq(outs)

# ---- Load and expand generated_questions.json query_templates ----
def load_query_templates():
    qset_path = CONFIG_DIR / "question_sets" / "generated_questions.json"
    try:
        raw = json.loads(qset_path.read_text(encoding="utf-8"))
        return raw.get("query_templates", {}) or {}
    except Exception:
        return {}

def expand_templates_for_dim(dimension: str, templates: list, entities: list, key_terms: list, numbers: list, time_hint: str):
    people = [e for e in entities if e]
    orgs = [e for e in entities if e]
    terms = [t for t in key_terms if t]
    nums  = [n for n in numbers if n]

    p_opts = people[:2] or [""]
    o_opts = orgs[:2] or [""]
    t_opts = terms[:3] or [""]
    n_opts = nums[:2]  or [""]

    out = []
    for tpl in templates or []:
        tpl = str(tpl or "")
        for p in p_opts:
            for o in o_opts:
                for t in t_opts:
                    for n in n_opts:
                        q = tpl.replace("{PERSON}", p).replace("{ORG}", o).replace("{TERM}", t).replace("{NUM}", n)
                        q = q.replace("  ", " ").strip()
                        if time_hint and "20" in time_hint and time_hint not in q:
                            q = f'{q} {time_hint}'
                        out.append(clean_query(q))
    return uniq([q for q in out if q])[:MAX_QUERIES_PER_DIM]

def llm_generate_queries(question, context, dimension, hints=None, entities=None, numbers=None, key_terms=None):
    time_hint = "2019..2026" if dimension in ("strategy","objectives","feasibility") else "2021..2026"
    dim_hints = "; ".join(DIM_HINTS.get(dimension.lower(), []))
    hint_text = "; ".join(hints or [])
    ent_text = "; ".join(entities or [])[:240]
    num_text = ", ".join(numbers or [])[:80]
    key_text = "; ".join(key_terms or [])[:240]

    theme_terms = uniq((key_terms or []) + (entities or []))[:8]
    mh_text = "; ".join(theme_terms) if theme_terms else "use the most specific domain terms present in the summary"

    prompt = f"""
You are a sector-agnostic diligence search expert. Generate {FIRST_ROUND_N} high-quality search queries from the question, dimension, summary, hints, entities, and numbers.
Requirements:
- Infer the business/domain context from the provided text. Do not force biomedical, software, manufacturing, finance, public-sector, or any other sector terms unless the text supports them.
- Include entities, organizations, product/service names, model names, registration/certification IDs, contract or patent IDs, and key numbers when available.
- Use site: filters and the time window ({time_hint}); queries may be English or Chinese if useful, but should be concise and directly usable in Google.
- Include at least one of these theme terms when possible: {mh_text}
- Prefer authoritative sources relevant to the inferred context: official company/project pages, regulator/standards bodies, patent databases, government/procurement pages, academic sources, reputable market sources, or credible industry sources.
- Return a strict JSON array of strings only, with no explanation.
Question: {question}
Dimension: {dimension}
Summary: {context[:900]}
Hints: {hint_text}; {dim_hints}
Entities: {ent_text}
Numbers: {num_text}
Keywords: {key_text}
"""
    try:
        rsp = client.chat.completions.create(
            model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.35
        )
        content = rsp.choices[0].message.content.strip()
        queries = safe_json_loads(content)
        queries = [clean_query(q["query"] if isinstance(q, dict) and "query" in q else str(q)) for q in queries]
        contextual_variants = [f"{q} benchmark evidence market operations compliance {time_hint}" for q in queries]
        merged = uniq(queries + contextual_variants)
        return merged[:MAX_QUERIES_PER_DIM] if merged else [question]
    except Exception as e:
        print(f"LLM query generation failed: {e}")
        return [question]

# ===== Base must/should clause builder =====
def build_base_clause(dim_name: str, qcfg: dict, qsets_meta: dict):
    doc_policy = (qsets_meta.get("doc_policy") or {})
    must_terms = list(dict.fromkeys((qcfg.get("search", {}).get("must_terms") or []) + (doc_policy.get("must_terms") or [])))
    should_terms = list(dict.fromkeys((qcfg.get("search", {}).get("should_terms") or []) + (doc_policy.get("should_terms") or [])))

    def qwrap(t):
        t = str(t).strip()
        if not t: return ""
        return f"({t})" if " " in t and not (t.startswith('"') and t.endswith('"')) else t

    must_clause = " ".join(qwrap(t) for t in must_terms if t)
    should_clause = ""
    if should_terms:
        should_clause = " (" + " OR ".join(qwrap(t) for t in should_terms if t) + ")"
    base = (must_clause + should_clause).strip()
    return base, must_terms, should_terms

# ============ Main flow ============

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", help="Only search the first two questions per dimension for debugging.")
args = parser.parse_args()

# 1) Read cleaned dimensions from the fixed path.
parsed_path = PARSED_DIR / "parsed_dimensions.clean.llm.json"
if not parsed_path.exists():
    print(f"Cleaned dimension file not found: {parsed_path}. Run strict_cleanup_llm.py first.")
    sys.exit(1)

try:
    dimensions = json.loads(parsed_path.read_text(encoding="utf-8"))
except Exception as e:
    print(f"Failed to read dimension file: {e}"); sys.exit(1)

# 2) Read question set from the fixed path.
qset_path = CONFIG_DIR / "question_sets" / "generated_questions.json"
if not qset_path.exists():
    print(f"Question set not found: {qset_path}"); sys.exit(1)
try:
    question_sets = json.loads(qset_path.read_text(encoding="utf-8"))
except Exception as e:
    print(f"Failed to read question set: {e}"); sys.exit(1)

# 3) Recover proposal_id from run_meta.source_path.
proposal_id = "current_proposal"
try:
    src_path = (dimensions.get("run_meta") or {}).get("source_path", "")
    if src_path:
        p = Path(src_path)
        if p.name.endswith("_dimensions.json"):
            proposal_id = p.stem.replace("_dimensions", "")
except Exception:
    pass
os.environ["CURRENT_PROPOSAL_ID"] = proposal_id
print(f"Current proposal file: {proposal_id}")

EVIDENCE_DIR = EVIDENCE_ROOT / proposal_id
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Read top-level query_templates from the question set.
QUERY_TEMPLATES_ALL = question_sets.get("query_templates", {}) or {}

global_domain_counter, stats = Counter(), {}
debug_overview = {}

# Metadata used by base clauses.
qsets_meta = question_sets.get("meta", {}) or {}

for dim, ctx in dimensions.items():
    if dim in IGNORE_DIMS or dim not in question_sets:
        continue

    print(f"\nStarting dimension: {dim}")
    t0 = time.time()
    qcfg = question_sets[dim]
    questions = qcfg.get("questions", []) or []
    if not questions:
        print("This dimension has no questions; skipping."); continue
    if args.fast:
        questions = questions[:2]; print("Fast mode: only searching the first two questions.")

    # Collect dimension context.
    ents, nums, key_terms = collect_entities_numbers_terms(ctx or {})
    dim_evidences = []
    per_q_domain_counter = Counter()
    success, fail = 0, 0
    dim_academic_hits = 0

    all_queries_fired = []
    needed_hits = EARLY_STOP_ACADEMIC.get(dim, 3)
    dim_soft_cap = DIM_SOFT_EARLY_STOP.get(dim, 8)

    # Dimension templates from generated_questions.json query_templates.
    dim_templates = (QUERY_TEMPLATES_ALL.get(dim, []) or [])[:10]
    templates_before = len(dim_templates)

    # Build base must/should clauses and merge search_hints.
    base_clause, must_terms_used, should_terms_used = build_base_clause(dim, qcfg, qsets_meta)
    merged_hints = list(dict.fromkeys((qcfg.get("search_hints") or []) + (qsets_meta.get("doc_policy", {}) or {}).get("query_hints_merged", [])))[:10]

    for q in questions:
        if dim_academic_hits >= dim_soft_cap:
            print(f"Dimension academic-hit soft cap reached ({dim_soft_cap}); reducing later query fanout.")
            max_per_question = 2
        else:
            max_per_question = MAX_QUERIES_PER_DIM

        print(f"\nQuestion: {q}")
        # 1) Generate first-round strong queries with the LLM.
        llm_queries = llm_generate_queries(
            q, (ctx or {}).get("summary",""), dim,
            hints=merged_hints,
            entities=ents, numbers=nums, key_terms=key_terms
        )

        # 2) Expand dimension templates and inject the time window.
        time_hint = "2019..2025" if dim in ("strategy","objectives","feasibility") else "2021..2025"
        expanded_tpl = expand_templates_for_dim(dim, dim_templates, ents, key_terms, nums, time_hint)

        # 3) Fallback templates.
        fallbacks = _inject_fallbacks(dim, ents, [q] + DIM_HINTS.get(dim, []))

        # 4) Standalone hint-based queries.
        hint_queries = []
        for h in merged_hints:
            if not h: continue
            hq = h
            if base_clause:
                hq = f"{base_clause} {h}".strip()
            hint_queries.append(clean_query(hq))

        # Merge order: original question -> LLM -> templates -> fallback -> standalone hints.
        merged_queries = uniq([q] + llm_queries + expanded_tpl + fallbacks + hint_queries)

        # Prefix each query with the base must/should clause.
        if base_clause:
            merged_queries = [clean_query(f"{base_clause} {qq}") for qq in merged_queries]

        # Cap query fanout per question.
        merged_queries = merged_queries[:max_per_question]

        # ---- Execute queries ----
        per_question_academic_hits = 0
        empty_hits = 0
        for i, query in enumerate(merged_queries, start=1):
            print(f"Search ({i}/{len(merged_queries)}): {query}")
            all_queries_fired.append(query)
            try:
                texts, urls = simple_search(
                    query, max_results=MAX_RESULTS_PER_QUERY,
                    dimension=dim, hints=merged_hints, source="LLM"
                )
                got = 0
                for t, u in zip(texts, urls):
                    dim_evidences.append({"query": query, "text": t, "url": u})
                    host = urlparse(u).hostname or ""
                    if host:
                        per_q_domain_counter[host] += 1
                        global_domain_counter[host] += 1
                        if any(ad in host for ad in ACADEMIC_SITES):
                            per_question_academic_hits += 1
                            dim_academic_hits += 1
                    got += 1
                if got == 0: empty_hits += 1
                success += 1 if got > 0 else 0
                fail += 1 if got == 0 else 0
            except Exception as e:
                print(f"Search failed: {e}"); fail += 1; empty_hits += 1

            if per_question_academic_hits >= needed_hits:
                print("This question has enough academic-source hits; stopping expansion early."); break
            time.sleep(SLEEP_BETWEEN_QUERIES)

        # Empty-hit fallback: if all previous searches were empty, try a bag-of-words site-filter query.
        if per_question_academic_hits == 0 and empty_hits >= len(merged_queries):
            bag = uniq((merged_hints or []) + ents + key_terms)
            bag = [b for b in bag if len(b) >= 2][:6]
            if bag:
                bag_q = " ".join(f'"{b}"' for b in bag)
                bag_q = f'{bag_q} (site:nih.gov OR site:fda.gov OR site:ema.europa.eu OR site:who.int OR site:clinicaltrials.gov) {time_hint}'
                bag_q = clean_query(bag_q)
                print(f"Fallback search: {bag_q}")
                all_queries_fired.append(bag_q)
                try:
                    texts, urls = simple_search(
                        bag_q, max_results=MAX_RESULTS_PER_QUERY,
                        dimension=dim, hints=merged_hints, source="LLM"
                    )
                    for t, u in zip(texts, urls):
                        dim_evidences.append({"query": bag_q, "text": t, "url": u})
                        host = urlparse(u).hostname or ""
                        if host:
                            per_q_domain_counter[host] += 1
                            global_domain_counter[host] += 1
                            if any(ad in host for ad in ACADEMIC_SITES):
                                per_question_academic_hits += 1
                                dim_academic_hits += 1
                    success += 1 if texts else 0
                    fail += 1 if not texts else 0
                except Exception as e:
                    print(f"Fallback search failed: {e}")

    # Record dimension queries with richer diagnostics.
    queries_record = {
        "dimension": dim,
        "templates_loaded": templates_before,
        "templates_expanded": len(expanded_tpl) if 'expanded_tpl' in locals() else 0,
        "queries_total": len(all_queries_fired),
        "queries": all_queries_fired
    }
    (EVIDENCE_DIR / f"{dim}_queries.json").write_text(
        json.dumps(queries_record, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    if len(dim_evidences) < MIN_SAVE_EVIDENCE:
        print(f"{dim} has too little evidence ({len(dim_evidences)}); skipping save.")
        continue

    raw_ev = EVIDENCE_DIR / f"{dim}_evidence_raw.json"
    try:
        raw_ev.write_text(json.dumps(dim_evidences, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Failed to write raw evidence: {e}")

    elapsed = round(time.time() - t0, 1)
    success_rate = round(success / (success + fail + 1e-6), 2)
    print(f"{dim} complete: success_rate={success_rate}, evidence={len(dim_evidences)}, elapsed_sec={elapsed}")

    stats[dim] = {
        "question_count": len(questions), "success": success, "failure": fail,
        "success_rate": success_rate, "evidence_count": len(dim_evidences),
        "top_domains": per_q_domain_counter.most_common(6), "elapsed_sec": elapsed,
        "dimension_academic_hits": dim_academic_hits
    }
    debug_overview[dim] = {
        "queries_total": queries_record["queries_total"],
        "evidence_kept": len(dim_evidences),
        "elapsed_s": elapsed,
        "early_stop_threshold_per_question": EARLY_STOP_ACADEMIC.get(dim, 3),
        "early_stop_soft_dim": DIM_SOFT_EARLY_STOP.get(dim, 8)
    }

report = {
    "proposal_id": proposal_id, "provider": provider, "model": model_name,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "stats": stats, "top_domains_global": global_domain_counter.most_common(12)
}
summary_path = EVIDENCE_DIR / "dimension_summary_index.json"
summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

(EVIDENCE_DIR / "dimension_debug_overview.json").write_text(
    json.dumps(debug_overview, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("\nSearch complete. Report saved to:", summary_path)
print("top_domains_global:", report["top_domains_global"])
print(f"output_dir={EVIDENCE_DIR}")
print(f"combined_files_generated={len(list(EVIDENCE_DIR.glob('*_combined.json')))}")
