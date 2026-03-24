"""
run_all.py
==========
Convenience script to execute the full CRISP-DM pipeline in order.

Usage
-----
    python run_all.py              # full pipeline (including BERT)
    python run_all.py --skip-bert  # skip BERT fine-tuning (CPU-only)
"""

import subprocess
import sys
import time

STAGES = [
    ("01_data_understanding.py",     "Data Understanding (EDA)"),
    ("02_data_preparation.py",       "Data Preparation"),
    ("03_modeling.py",               "Modeling"),
    ("04_evaluation.py",             "Evaluation"),
    ("05_knowledge_extraction.py",   "Knowledge Extraction (XAI)"),
]

skip_bert = "--skip-bert" in sys.argv


def run_stage(script: str, description: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  RUNNING: {description}")
    print(f"  Script : {script}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, script]
    if skip_bert and script in ("03_modeling.py", "04_evaluation.py"):
        cmd.append("--skip-bert")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    status = "✓ PASSED" if result.returncode == 0 else "✗ FAILED"
    print(f"\n  {status} ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Suicide Risk Classification – Full Pipeline Runner  ║")
    print("║   CRISP-DM Methodology                                ║")
    if skip_bert:
        print("║   Mode: CPU-only (--skip-bert)                        ║")
    else:
        print("║   Mode: Full (including BERT fine-tuning)              ║")
    print("╚════════════════════════════════════════════════════════╝")

    total_start = time.time()
    results = []

    for script, desc in STAGES:
        ok = run_stage(script, desc)
        results.append((desc, ok))
        if not ok:
            print(f"\n⚠ Pipeline stopped at '{desc}' due to error.")
            break

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for desc, ok in results:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {desc}")
    print(f"\n  Total time: {total_time:.1f}s")

    if all(ok for _, ok in results):
        print(f"\n  All stages passed! Now run:")
        print(f"    streamlit run 06_dashboard.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
