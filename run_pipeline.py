import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, desc):
    print(f"\n=======================================================")
    print(f" PIPELINE STAGE: {desc}")
    print(f" COMMAND: {' '.join(cmd)}")
    print(f"=======================================================\n")
    start_time = time.time()
    result = subprocess.run(cmd, check=True)
    elapsed = time.time() - start_time
    print(f"\nStage '{desc}' completed in {elapsed:.2f}s.\n")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run Full Retrieval Pipeline End-to-End")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of product images in dataset")
    parser.add_argument("--epochs", type=int, default=2, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and eval")
    args = parser.parse_args()

    python_exe = sys.executable

    total_start = time.time()
    timings = {}

    # Stage 1: Data Preparation
    timings["prepare_dataset"] = run_command([
        python_exe, "prepare_dataset.py", "--num_samples", str(args.num_samples)
    ], "1. Prepare Product Dataset")

    # Stage 2: LLaVA Caption Generation
    timings["caption_generation"] = run_command([
        python_exe, "caption_generation.py", "--manifest", "data/manifest.json", "--output_cache", "data/captions.json"
    ], "2. Resumable LLaVA Caption Generation")

    # Stage 3: CLIP LoRA Fine-Tuning
    timings["finetune_clip"] = run_command([
        python_exe, "finetune_clip.py", "--manifest", "data/manifest.json", "--captions", "data/captions.json", "--epochs", str(args.epochs), "--batch_size", str(args.batch_size)
    ], "3. Fine-Tune CLIP with LoRA")

    # Stage 4: Evaluation
    timings["evaluate_retrieval"] = run_command([
        python_exe, "evaluate_retrieval.py", "--manifest", "data/manifest.json", "--captions", "data/captions.json"
    ], "4. Zero-Shot vs Fine-Tuned Retrieval Evaluation")

    # Stage 5: Build FAISS Index
    timings["build_index"] = run_command([
        python_exe, "build_index.py", "--manifest", "data/manifest.json", "--captions", "data/captions.json"
    ], "5. Build FAISS Vector Index")

    total_elapsed = time.time() - total_start

    print("\n=======================================================")
    print("        FULL RETRIEVAL PIPELINE COMPLETE               ")
    print("=======================================================")
    for stage, sec in timings.items():
        print(f" - {stage:20s}: {sec:8.2f}s")
    print(f" -----------------------------------------------------")
    print(f"   Total Pipeline Time : {total_elapsed:8.2f}s ({total_elapsed/60.0:.2f} min)")
    print("=======================================================")
    print("\nTo launch search demo server and telemetry API, run:")
    print(f"  {python_exe} serve.py\n")


if __name__ == "__main__":
    main()
