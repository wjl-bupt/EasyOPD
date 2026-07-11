#!/usr/bin/env python3
"""
Prepare MBPP and LiveCodeBench-v6 datasets for code evaluation.

Downloads from HuggingFace and saves to the shared eval_data directory.

Usage:
    python prepare_code_data.py [--data_dir /path/to/eval_data]
"""

import os
import sys
import json
import argparse

DATA_DIR = "/path/to/EasyOPD/experiments/_shared/eval_data"


def prepare_mbpp(data_dir: str):
    """Download and save MBPP dataset."""
    output_dir = os.path.join(data_dir, "mbpp")
    
    # Check if already exists
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        # Verify it's a valid HF dataset
        if os.path.exists(os.path.join(output_dir, "dataset_info.json")) or \
           os.path.exists(os.path.join(output_dir, "state.json")):
            print(f"✓ MBPP dataset already exists: {output_dir}")
            return True
    
    print("=" * 60)
    print("Preparing MBPP dataset")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        return False
    
    print("Downloading MBPP dataset from HuggingFace...")
    try:
        ds = load_dataset("mbpp")
    except Exception as e:
        print(f"Error downloading MBPP: {e}")
        print("\nIf offline, you can manually download:")
        print("  python -c \"from datasets import load_dataset; "
              "ds = load_dataset('mbpp'); ds.save_to_disk('/path/to/mbpp')\"")
        return False
    
    print(f"MBPP splits: {list(ds.keys())}")
    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data)} samples, columns: {split_data.column_names}")
    
    os.makedirs(output_dir, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"✓ MBPP dataset saved to: {output_dir}")
    return True


def prepare_lcb(data_dir: str):
    """Download and save LiveCodeBench-v6 dataset.

    Modern `datasets` versions reject script-based loaders, so we bypass
    `load_dataset` entirely and download the raw jsonl files directly via
    huggingface_hub. release_v6 = test.jsonl + test2.jsonl + ... + test6.jsonl
    (see code_generation_lite.py in the upstream repo).
    """
    output_dir = os.path.join(data_dir, "live-code-bench-v6")
    problems_file = os.path.join(output_dir, "problems.jsonl")

    if os.path.exists(problems_file):
        with open(problems_file) as f:
            count = sum(1 for _ in f)
        if count > 0:
            print(f"✓ LiveCodeBench-v6 already exists: {problems_file} ({count} problems)")
            return True

    print("=" * 60)
    print("Preparing LiveCodeBench-v6 dataset")
    print("=" * 60)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: 'huggingface_hub' package not installed.")
        return False

    # release_v6 = all 6 jsonl files (matches upstream code_generation_lite.py)
    release_v6_files = [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ]

    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Downloading {len(release_v6_files)} jsonl files via hf_hub_download...")
    print("(LCB total ~4.5GB, this may take a while)")

    downloaded_paths = []
    try:
        for fname in release_v6_files:
            print(f"  Downloading {fname}...")
            path = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=fname,
                repo_type="dataset",
                local_dir=raw_dir,
            )
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"    ✓ {fname} ({size_mb:.1f} MB) -> {path}")
            downloaded_paths.append(path)
    except Exception as e:
        print(f"Error downloading LiveCodeBench files: {e}")
        return False

    # Merge all jsonl files into problems.jsonl, normalizing field names
    print(f"\nMerging {len(downloaded_paths)} files into {problems_file}...")
    count = 0
    with open(problems_file, 'w', encoding='utf-8') as fout:
        for path in downloaded_paths:
            with open(path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    problem = {
                        "question_id": sample.get(
                            "question_id",
                            sample.get("task_id", f"lcb_{count}"),
                        ),
                        "question_content": sample.get(
                            "question_content",
                            sample.get("description", ""),
                        ),
                        "test_cases": sample.get(
                            "test_cases",
                            sample.get("public_test_cases", "[]"),
                        ),
                    }
                    # Preserve every original field (public_test_cases,
                    # private_test_cases, starter_code, difficulty, ...).
                    for k, v in sample.items():
                        if k not in problem:
                            problem[k] = v
                    fout.write(json.dumps(problem, ensure_ascii=False) + "\n")
                    count += 1

    print(f"✓ Saved {count} problems to {problems_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare code evaluation datasets")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help=f"Output directory (default: {DATA_DIR})")
    parser.add_argument("--benchmarks", type=str, default="mbpp,lcb",
                        help="Comma-separated: mbpp, lcb (default: mbpp,lcb)")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    
    results = {}
    if "mbpp" in benchmarks:
        results["mbpp"] = prepare_mbpp(args.data_dir)
    if "lcb" in benchmarks:
        results["lcb"] = prepare_lcb(args.data_dir)
    
    print("\n" + "=" * 60)
    print("Summary:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
