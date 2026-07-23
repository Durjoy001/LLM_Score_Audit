# src/tools/run_pipeline.py
# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path

# Project root that contains src/.
BASE_DIR = Path(__file__).resolve().parents[2]


def run_cmd(cmd: list):
    """Run one child command from the project root and raise immediately on failure."""
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run_full_pipeline():
    """
    Run the full pipeline in automatic latest-proposal mode:

      1) prepare_proposal_text.py
      2) extract_facts_by_chunk.py
      3) build_dimensions_from_facts.py
      4) generate_questions.py
      5) llm_answering.py
      6) post_processing.py
      7) ai_expert_opinion.py
      8) generate_final_report.py
    9) generate_llm_scores.py

    Assumption: each script can detect the latest proposal when --file,
    --proposal_id, or --pid is omitted.
    """

    # 1) Prepare text.
    run_cmd(["python", "src/tools/prepare_proposal_text.py"])

    # 2) Extract facts.
    run_cmd(["python", "src/tools/extract_facts_by_chunk.py"])

    # 3) Build dimensions from facts.
    run_cmd(["python", "src/tools/build_dimensions_from_facts.py"])

    # 4) Generate questions.
    run_cmd(["python", "src/tools/generate_questions.py"])

    # 5) Generate LLM answers.
    run_cmd(["python", "src/tools/llm_answering.py"])

    # 6) post-processing
    run_cmd(["python", "src/tools/post_processing.py"])

    # 7) Generate AI expert opinion.
    run_cmd(["python", "src/tools/ai_expert_opinion.py"])

    # 8) Compile final report.
    run_cmd(["python", "src/tools/generate_final_report.py"])

    # 9) Generate LLM rubric scores.
    run_cmd(["python", "src/tools/generate_llm_scores.py"])

    print("Full pipeline finished.")


if __name__ == "__main__":
    # No CLI arguments are required; run once using the latest proposal.
    run_full_pipeline()
