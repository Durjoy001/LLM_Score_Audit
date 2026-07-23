from dotenv import load_dotenv
load_dotenv()

import os
import json
import argparse

from backend.chains.base_chain import BaseChain
from backend.chains.orchestrator import run_all, save_full_report

def parse_args():
    p = argparse.ArgumentParser(description="RAG-6View CLI")
    p.add_argument("--mode", choices=["single", "all"], default="all",
                   help="single=run one dimension; all=run all dimensions in parallel")
    p.add_argument("--dimension", default="team", help="Dimension to use in single mode")
    p.add_argument("--question", default=None, help="Custom question, optional")
    p.add_argument("--max_workers", type=int, default=3, help="Number of parallel worker threads")
    return p.parse_args()

def run_single(dim: str, question: str = None):
    print("Starting RAG Demo (Single)...")
    chain = BaseChain(dim)
    q = question or "Does the core team have strong research capability?"
    result = chain.run(q)
    os.makedirs("data/results", exist_ok=True)
    out = "data/results/single_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSingle-dimension analysis complete. Results saved to: {out}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

def run_all_dims(max_workers: int):
    print("Starting RAG Demo (ALL Dimensions, parallel)...")
    report = run_all(max_workers=max_workers)
    out_path = save_full_report(report)
    print(f"\nAll-dimension analysis complete. Report saved to: {out_path}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))

def main():
    args = parse_args()
    if args.mode == "single":
        run_single(args.dimension, args.question)
    else:
        run_all_dims(args.max_workers)

if __name__ == "__main__":
    main()
