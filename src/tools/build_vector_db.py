# -*- coding: utf-8 -*-
"""
Vector Knowledge Base Builder (v2025.12R; Fusion-Sync Edition)

Features:
- Compatible with fusion_search v2025.12R output, including confidence_distribution and embedding_model.
- Dynamically filters short text and low-confidence text.
- Reports high/mid/low confidence document counts and averages.
- Auto-detects GPU availability and adapts batch_size.
- Computes insert rate and record distribution for health checks.
- Writes an enhanced index with confidence_tiers and global average confidence.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb

# ========= Paths and environment =========
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_DIR = SRC_ROOT / "data"
FUSION_ROOT = DATA_DIR / "fused_evidence"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ========= Auto-detect latest proposal =========
subdirs = [d for d in FUSION_ROOT.iterdir() if d.is_dir()]
if not subdirs:
    raise FileNotFoundError("No fused_evidence directory found. Run fusion_search.py first.")
latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
proposal_id = latest_dir.name
print(f"Current proposal import target: {proposal_id}")

# ========= Model loading =========
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
print(f"Loading embedding model: {EMBED_MODEL} ...")
model = SentenceTransformer(EMBED_MODEL)

device = "cuda" if getattr(model, "device", None) and model.device.type == "cuda" else "cpu"
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "8"))
print(f"Inference device: {device.upper()} | batch_size={BATCH_SIZE}")

# ========= Initialize database =========
collection_name = f"fusion_{proposal_id}"
client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={
        "desc": "RAG Vector Knowledge Base (Fusion 2025.12R)",
        "proposal_id": proposal_id,
        "created_at": str(datetime.now())
    }
)
print(f"Loaded vector collection: {collection_name}")

# ========= Utilities =========
def _as_list(x):
    if isinstance(x, list): return x
    if not x: return []
    return [str(x)]

def _join_urls(urls: List[str], limit=8) -> str:
    urls = urls[:limit]
    return "; ".join(urls)

def _load_fused_jsons(fusion_dir: Path) -> List[Dict[str, Any]]:
    """
    Load fusion_search output files:
    - data["fused_texts"] : [{text, urls, avg_conf}]
    - skip short or low-confidence text
    """
    docs = []
    for f in fusion_dir.glob("*_fused.json"):
        try:
            dim = f.stem.replace("_fused", "")
            data = json.loads(f.read_text(encoding="utf-8"))
            fused_texts = data.get("fused_texts", [])
            threshold_lowconf = 0.45
            valid = 0
            for i, item in enumerate(fused_texts):
                if not isinstance(item, dict):
                    continue
                text = (item.get("text") or "").strip()
                urls = _as_list(item.get("urls") or [])
                avg_conf = float(item.get("avg_conf") or data.get("avg_confidence_weighted", 0.6))
                if len(text) < 100 or avg_conf < threshold_lowconf:
                    continue
                doc_id = f"{proposal_id}_{dim}_{i}"
                meta = {
                    "dimension": dim,
                    "proposal_id": proposal_id,
                    "source_file": f.name,
                    "urls": _join_urls(urls),
                    "avg_confidence": avg_conf,
                    "fusion_threshold": data.get("threshold", 0.65),
                    "embedding_model": data.get("embedding_model", EMBED_MODEL)
                }
                docs.append({"id": doc_id, "text": text, "meta": meta})
                valid += 1
            print(f"Loaded {dim:<12} -> {valid} valid texts")
        except Exception as e:
            print(f"Failed to parse {f.name}: {e}")
    return docs

def _dedup_ids(ids: List[str]) -> List[bool]:
    try:
        existing = set(collection.get(ids=ids).get("ids", []))
    except Exception:
        existing = set()
    return [(_id not in existing) for _id in ids]

# ========= Main flow =========
def build_vector_db():
    docs = _load_fused_jsons(latest_dir)
    if not docs:
        print("No fused data available for import.")
        return

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metas = [d["meta"] for d in docs]
    avg_len = sum(len(t) for t in texts) / max(len(texts), 1)

    print(f"\nText count: {len(texts)} | average_length_chars={avg_len:.1f}")

    # ===== Embedding generation =====
    print("Generating embeddings ...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        device=device
    )

    # ===== Deduplicate and write =====
    print("Writing to Chroma database ...")
    new_mask = _dedup_ids(ids)
    new_ids = [i for i, keep in zip(ids, new_mask) if keep]
    if not new_ids:
        print("No new records to add; this proposal may already be imported.")
    else:
        collection.add(
            ids=new_ids,
            documents=[t for t, keep in zip(texts, new_mask) if keep],
            embeddings=[e.tolist() for e, keep in zip(embeddings, new_mask) if keep],
            metadatas=[m for m, keep in zip(metas, new_mask) if keep]
        )
        print(f"Wrote {len(new_ids)} new records.")

    # ===== Distribution statistics =====
    dim_stats = {}
    conf_all = []
    for m in metas:
        dim = m["dimension"]
        conf = float(m.get("avg_confidence", 0.6))
        conf_all.append(conf)
        dim_stats.setdefault(dim, {"count": 0, "conf_sum": 0.0, "high": 0, "mid": 0, "low": 0})
        dim_stats[dim]["count"] += 1
        dim_stats[dim]["conf_sum"] += conf
        if conf >= 0.8:
            dim_stats[dim]["high"] += 1
        elif conf >= 0.6:
            dim_stats[dim]["mid"] += 1
        else:
            dim_stats[dim]["low"] += 1

    dim_distribution = {
        k: {
            "count": v["count"],
            "avg_conf": round(v["conf_sum"] / max(v["count"], 1), 2),
            "confidence_tiers": {
                "high": v["high"],
                "mid": v["mid"],
                "low": v["low"]
            }
        } for k, v in dim_stats.items()
    }

    global_avg_conf = round(np.mean(conf_all), 3)
    total_records = sum(v["count"] for v in dim_stats.values())
    stored_count = collection.count()
    insert_rate = round((stored_count / max(total_records, 1)) * 100, 1)

    # ===== Save index =====
    index_info = {
        "proposal_id": proposal_id,
        "collection_name": collection_name,
        "record_count": total_records,
        "stored_count": stored_count,
        "insert_rate_percent": insert_rate,
        "avg_text_length": round(avg_len, 1),
        "dimension_distribution": dim_distribution,
        "global_avg_confidence": global_avg_conf,
        "embedding_model": EMBED_MODEL,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "db_path": str(VECTOR_DB_DIR.resolve())
    }

    index_path = VECTOR_DB_DIR / f"{proposal_id}_vector_index.json"
    index_path.write_text(json.dumps(index_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nVector index saved: {index_path}")
    print(f"Current collection records: {stored_count} | global_avg_confidence={global_avg_conf}")
    print(f"Insert rate: {insert_rate:.1f}%")
    print("Vector knowledge base build complete.")

# ========= Entry point =========
if __name__ == "__main__":
    build_vector_db()
