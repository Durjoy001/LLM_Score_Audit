# -*- coding: utf-8 -*-
"""
Fusion Search (v2025.12 ProClean; Safe & Adaptive + Domain Hygiene)
Keeps output compatibility with src/data/fused_evidence/{proposal_id}/{dimension}_fused.json.
"""
import os
# Stabilize parallel/thread behavior.
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT","1")

import sys, re, json, numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))

from sentence_transformers import SentenceTransformer, util
from backend.utils.model_selector import get_llm_client

DATA_DIR = SRC_ROOT / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
OUTPUT_DIR = DATA_DIR / "fused_evidence"
for d in [EVIDENCE_DIR, OUTPUT_DIR]: d.mkdir(parents=True, exist_ok=True)

# Prefer the proposal id from the environment so this stays aligned with upstream search.
_env_pid = os.getenv("CURRENT_PROPOSAL_ID", "").strip()
if _env_pid and (EVIDENCE_DIR / _env_pid).exists():
    proposal_dir = EVIDENCE_DIR / _env_pid
else:
    subdirs = [d for d in EVIDENCE_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError("No evidence subdirectory found. Run search_by_dimension.py first.")
    proposal_dir = max(subdirs, key=lambda d: d.stat().st_mtime)

proposal_id = proposal_dir.name
print(f"Current fusion proposal target: {proposal_id}")

FUSION_DIR = OUTPUT_DIR / proposal_id
FUSION_DIR.mkdir(parents=True, exist_ok=True)

# Academic/regulatory allowlist, aligned with web_search.
ACADEMIC_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov","pmc.ncbi.nlm.nih.gov","nature.com","sciencedirect.com",
    "nih.gov","who.int","ema.europa.eu","fda.gov","clinicaltrials.gov",
    "thelancet.com","bmj.com","cell.com","springer.com","biorxiv.org","medrxiv.org","arxiv.org","nejm.org",
    "nmpa.gov.cn"
]
INSTITUTIONAL_HINTS = (".edu",".ac.","university","hospital")

DOMAIN_BLACKLIST = (
    "sol-war.ru","moomoo.com","money.finance.","islandenergy.je","xmind.com","scribd.com",
    "pinterest.","medium.com","reddit.","bilibili.","zhihu.","weibo.","press","news","careers","recruit"
)

def _is_whitelisted_domain(host: str) -> bool:
    h = (host or "").lower()
    return any(h.endswith(d) for d in ACADEMIC_DOMAINS) or any(x in h for x in INSTITUTIONAL_HINTS)

EMBED_MODEL = "BAAI/bge-m3"
print(f"Loading embedding model on CPU: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
_EMB_KW = dict(normalize_embeddings=True, batch_size=16, show_progress_bar=False, convert_to_numpy=True)

llm = get_llm_client()
llm_client = llm["client"]; llm_model = llm["model_name"]; provider = llm["provider"]
print(f"Using {provider.upper()} model: {llm_model}")

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()

def extract_domain(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", re.search(r"https?://([^/]+)/?", url).group(1))
    except Exception:
        return "unknown"

def load_evidence_files(proposal_path: Path):
    dim_map = defaultdict(list); total_loaded = 0
    for f in proposal_path.glob("*_combined.json"):
        dimension = f.stem.replace("_combined","")
        try:
            data = json.loads(f.read_text(encoding="utf-8")) or []
            for item in data:
                txt = clean_text(item.get("text",""))
                url = item.get("url",""); conf = float(item.get("confidence",0.5))
                if len(txt) > 100 and url:
                    dom = extract_domain(url)
                    # Fallback blacklist.
                    if any(bad in dom for bad in DOMAIN_BLACKLIST): continue
                    # Drop homepages unless the domain is trusted or the text is long.
                    if re.match(r"^https?://[^/]+/?$", url) and (not _is_whitelisted_domain(dom)) and len(txt) < 800:
                        continue
                    dim_map[dimension].append({"url": url,"domain": dom,"text": txt,"confidence": conf})
            print(f"{dimension} loaded {len(dim_map[dimension])} combined records.")
            total_loaded += len(dim_map[dimension])
        except Exception as e:
            print(f"Failed to read {f.name}: {e}")

    if not dim_map:
        print("No combined files detected; trying *_cache.json.")
        for f in proposal_path.glob("*_cache.json"):
            dimension = f.stem.replace("_cache","")
            try:
                cache_data = json.loads(f.read_text(encoding="utf-8"))
                all_items = []
                if isinstance(cache_data, dict):
                    for v in cache_data.values(): all_items.extend(v)
                else:
                    all_items = cache_data
                for item in all_items:
                    txt = clean_text(item.get("text",""))
                    url = item.get("url",""); conf = float(item.get("confidence",0.5))
                    if len(txt) > 100 and url:
                        dom = extract_domain(url)
                        if any(bad in dom for bad in DOMAIN_BLACKLIST): continue
                        if re.match(r"^https?://[^/]+/?$", url) and (not _is_whitelisted_domain(dom)) and len(txt) < 800:
                            continue
                        dim_map[dimension].append({"url": url,"domain": dom,"text": txt,"confidence": conf})
                print(f"{dimension} loaded {len(dim_map[dimension])} records from cache.")
                total_loaded += len(dim_map[dimension])
            except Exception as e:
                print(f"Failed to read {f.name}: {e}")

    print(f"\nLoaded dimensions={len(dim_map)} total_evidence={total_loaded}")
    return dim_map

def llm_chat(prompt: str, temperature: float = 0.35) -> str:
    try:
        resp = llm_client.chat.completions.create(
            model=llm_model, messages=[{"role":"user","content":prompt}], temperature=temperature
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM summary failed: {e}")
        return ""

def summarize_with_llm(dimension: str, fused_blocks: list) -> str:
    if not fused_blocks: return "Summary generation failed."
    ranked = sorted(fused_blocks, key=lambda b: (b.get("avg_conf",0.0), len(b.get("text",""))), reverse=True)[:6]
    joined = "\n\n".join([b["text"] for b in ranked])[:9000]
    # Reference set; prefer high-authority domains when available.
    refs = []
    for i,b in enumerate(ranked,1):
        for u in b.get("urls",[])[:2]:
            refs.append((i,u))
    # Ensure academic-source share when references are available.
    def _is_academic(u: str) -> bool:
        host = extract_domain(u)
        return any(host.endswith(d) for d in ACADEMIC_DOMAINS)
    if refs:
        ac_ratio = sum(1 for _,u in refs if _is_academic(u)) / len(refs)
        if ac_ratio < 0.7:
            # If academic share is too low, keep only academic/regulatory references.
            refs = [(i,u) for i,u in refs if _is_academic(u)]
    refs_text = "\n".join([f"[{i}] {u}" for i,u in refs[:6]])

    prompt = f"""
You are a sector-agnostic expert reviewer for business, startup, product, research, operating, public-sector, and investment diligence.
Based on the high-confidence integrated content below, write an objective summary for "{dimension}".
Requirements: keep it factual; cite references as [1][2]; any year, approval, funding, market, customer, certification, or regulatory statement must come from the provided sources; use 180-250 English words.
Content:
{joined}
Sources:
{refs_text}
At the end, include "Summary Quality: high/medium/low".
"""
    return llm_chat(prompt) or "Summary generation failed."

def representative_score(doc_len: int, conf: float, mean_len: float, domain: str = "") -> float:
    len_norm = min(doc_len / max(mean_len, 1.0), 2.0)
    bonus = 0.05 if _is_whitelisted_domain(domain) else 0.0
    return 0.7 * float(conf + bonus) + 0.3 * float(len_norm)

def _domain_penalty(domain: str) -> float:
    d = (domain or "").lower()
    if any(x in d for x in ["news","blog","medium.com"]): return 0.05
    return 0.0

def _is_homepage(url: str) -> bool:
    return bool(re.match(r"^https?://[^/]+/?$", url or ""))

def greedy_grouping(embs: np.ndarray, texts: list, threshold: float):
    if len(embs) == 0: return []
    clusters, centers = [], []
    for idx, v in enumerate(embs):
        placed = False
        for ci, c in enumerate(centers):
            if float(np.dot(v, c)) >= threshold:
                clusters[ci].append(idx)
                new_center = np.mean([embs[i] for i in clusters[ci]], axis=0)
                new_center = new_center / (np.linalg.norm(new_center) + 1e-12)
                centers[ci] = new_center; placed = True; break
        if not placed:
            clusters.append([idx]); centers.append(v)
    return clusters

def fuse_dimension(dimension: str, docs: list):
    if not docs:
        print(f"{dimension} has no evidence."); return None, None

    print(f"\nFusing dimension: {dimension} ({len(docs)} records)")
    corpus = [d["text"] for d in docs]
    urls_all = [d["url"] for d in docs]
    domains_all = [d["domain"] for d in docs]
    conf_all = [float(d["confidence"]) for d in docs]
    lens_all = [len(t) for t in corpus]
    mean_len = float(np.mean(lens_all)) if lens_all else 400.0
    doc_n = len(docs)

    if mean_len > 800 and doc_n >= 12: threshold, min_k = 0.72, 3
    elif mean_len < 300 or doc_n < 6:  threshold, min_k = 0.58, 2
    else:                               threshold, min_k = 0.65, 3

    embeddings = embedder.encode(corpus, **_EMB_KW)
    try:
        clusters = util.community_detection(embeddings, threshold=threshold, min_community_size=min_k)
    except Exception as e:
        print(f"community_detection failed; using greedy clustering: {e}")
        clusters = [c for c in greedy_grouping(embeddings, corpus, threshold) if len(c) >= min_k]

    print(f"Clusters={len(clusters)} threshold={threshold} min_k={min_k} mean_len={int(mean_len)}")

    fused_blocks, used = [], set()
    for cluster in clusters:
        cluster_docs = [docs[i] for i in cluster]
        filtered = [d for d in cluster_docs if not _is_homepage(d["url"])]
        ranked = sorted((filtered or cluster_docs),
                        key=lambda x: representative_score(len(x["text"]), x["confidence"] - _domain_penalty(x["domain"]), mean_len, x["domain"]),
                        reverse=True)
        texts = [cd["text"] for cd in ranked]
        urls  = [cd["url"]  for cd in ranked]
        confs = np.array([max(float(cd["confidence"]) - _domain_penalty(cd["domain"]), 0.0) for cd in ranked])
        w = confs / (confs.sum() + 1e-6)
        avg_conf = float((w * confs).sum())
        combined = " ".join(sorted(texts, key=len, reverse=True)[:3])[:2000]
        fused_blocks.append({"text": combined, "urls": urls, "avg_conf": round(avg_conf, 2)})
        for i in cluster: used.add(i)

    isolated_idx = [i for i in range(len(corpus)) if i not in used]
    isolated_entries = [{"text": corpus[i], "url": urls_all[i], "confidence": conf_all[i], "domain": domains_all[i]} for i in isolated_idx]

    pre_summary = summarize_with_llm(dimension, fused_blocks)
    if isolated_entries and pre_summary:
        embs_iso = embedder.encode([e["text"] for e in isolated_entries], **_EMB_KW)
        seed = embedder.encode([pre_summary[:1200]], **_EMB_KW)[0]
        sims = np.dot(embs_iso, seed)
        order = np.argsort(sims)[::-1][: min(5, len(sims))]
        for j in order:
            e = isolated_entries[int(j)]
            fused_blocks.append({"text": e["text"][:1500], "urls": [e["url"]], "avg_conf": round(float(e.get("confidence",0.6)),2)})

    confs_global = np.array(conf_all, dtype=float)
    w_global = confs_global / (confs_global.sum() + 1e-6)
    avg_conf_weighted = float((w_global * confs_global).sum())
    final_summary = summarize_with_llm(dimension, fused_blocks)

    high_blocks = [b for b in fused_blocks if b.get("avg_conf",0) >= 0.80]
    mid_blocks  = [b for b in fused_blocks if 0.60 <= b.get("avg_conf",0) < 0.80]
    low_blocks  = [b for b in fused_blocks if b.get("avg_conf",0) < 0.60]

    out_path = FUSION_DIR / f"{dimension}_fused.json"
    result = {
        "dimension": dimension, "fused_texts": fused_blocks, "summary": final_summary,
        "cluster_count": len(clusters), "doc_count": len(docs), "isolated_count": len(isolated_idx),
        "avg_confidence_weighted": round(avg_conf_weighted, 2),
        "confidence_distribution": {"high": len(high_blocks), "mid": len(mid_blocks), "low": len(low_blocks)},
        "avg_text_len": int(np.mean([len(b['text']) for b in fused_blocks])) if fused_blocks else 0,
        "threshold": threshold, "min_community_size": min_k,
        "embedding_model": EMBED_MODEL, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    top_domains = Counter(domains_all).most_common(10)
    print(f"{dimension} fusion complete -> {out_path}")
    print(f"clusters={len(clusters)} weighted_avg_confidence={result['avg_confidence_weighted']} confidence_tiers_H/M/L={len(high_blocks)}/{len(mid_blocks)}/{len(low_blocks)} top_domains={top_domains[:3]}")
    return result, top_domains

if __name__ == "__main__":
    print("Starting Fusion Search (safe mode) ...")
    dim_docs = load_evidence_files(proposal_dir)
    if not dim_docs:
        print("No evidence found."); sys.exit(0)

    fusion_index = {}; all_domains_global = Counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fuse_dimension, dim, docs): dim for dim, docs in dim_docs.items()}
        for f in as_completed(futures):
            dim = futures[f]
            try:
                result, top_domains = f.result()
                if result:
                    for d, c in (top_domains or []): all_domains_global[d] += c
                    fusion_index[dim] = {
                        "fused_file": str((FUSION_DIR / f"{dim}_fused.json").resolve()),
                        "cluster_count": result["cluster_count"], "isolated_count": result["isolated_count"],
                        "avg_confidence_weighted": result["avg_confidence_weighted"],
                        "confidence_distribution": result["confidence_distribution"],
                        "avg_text_len": result["avg_text_len"], "threshold": result["threshold"],
                        "min_community_size": result["min_community_size"], "embedding_model": result["embedding_model"],
                        "evidence_count": result["doc_count"], "top_domains": top_domains,
                        "summary_preview": (result["summary"] or "")[:150]
                    }
            except Exception as e:
                print(f"{dim} fusion failed: {e}")

    (FUSION_DIR / "fusion_report.json").write_text(json.dumps({
        "proposal_id": proposal_id, "fusion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dimension_stats": fusion_index, "top_domains_global": all_domains_global.most_common(15)
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (FUSION_DIR / "dimension_fusion_index.json").write_text(json.dumps(fusion_index, ensure_ascii=False, indent=2), encoding="utf-8")

    total_dims = len(fusion_index); total_docs = sum(v["evidence_count"] for v in fusion_index.values())
    print("\nFusion complete (safe mode)")
    print(f"output_dir={FUSION_DIR}")
    print(f"dimensions={total_dims} fused_evidence={total_docs}")
    print(f"global_top_domains={all_domains_global.most_common(10)}")
