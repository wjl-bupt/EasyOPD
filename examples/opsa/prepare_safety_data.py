"""
OPSA Data Preparation Script

Prepares safety alignment training data for the OPSA method.
Supports downloading and preprocessing the following datasets:
    - SafeChain (UWNSL/SafeChain): 40K samples of safe reasoning traces
    - ThinkSafe (Seanie-lee/ThinkSafe-*): Per-model safety datasets

Usage:
    python examples/opsa/prepare_safety_data.py \
        --dataset UWNSL/SafeChain \
        --output_dir data/opsa/ \
        --split train

    python examples/opsa/prepare_safety_data.py \
        --dataset Seanie-lee/ThinkSafe-Qwen3-1.7B \
        --output_dir data/opsa/ \
        --split train

Paper: https://arxiv.org/abs/2605.15239
"""

import argparse
from collections import Counter
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "Please install the `datasets` library: pip install datasets"
    )

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "Please install `pandas`: pip install pandas pyarrow"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare safety alignment data for OPSA training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="UWNSL/SafeChain",
        help="HuggingFace dataset name or path. "
             "Options: 'UWNSL/SafeChain', 'Seanie-lee/ThinkSafe-Qwen3-1.7B', etc."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/opsa/",
        help="Output directory for processed data files."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (train, test, validation)."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process. None for all."
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default=None,
        help="Key for the prompt/question field in the dataset. Auto-detected if None."
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default=None,
        help="Key for the safety label field (harmful/benign). Auto-detected if None."
    )
    return parser.parse_args()


def detect_keys(dataset_sample):
    """Auto-detect prompt and label keys from dataset sample."""
    keys = list(dataset_sample.keys())

    # Common prompt keys
    prompt_candidates = ["prompt", "question", "query", "input", "content", "instruction"]
    prompt_key = None
    for candidate in prompt_candidates:
        if candidate in keys:
            prompt_key = candidate
            break

    # Common label keys
    label_candidates = ["label", "safety_label", "category", "type", "is_harmful", "harmful"]
    label_key = None
    for candidate in label_candidates:
        if candidate in keys:
            label_key = candidate
            break

    return prompt_key, label_key


def format_sample(sample, prompt_key, label_key=None):
    """Format a single sample into EasyOPD-compatible format.

    Output format (parquet-compatible):
    {
        "content": [{"role": "user", "content": "..."}],
        "safety_label": "harmful" | "benign",  # if available
    }
    """
    prompt = sample[prompt_key]

    # Build chat-format content
    if isinstance(prompt, list):
        # Already in chat format
        content = prompt
    else:
        content = [{"role": "user", "content": str(prompt)}]

    result = {"content": content}

    # Add safety label if available
    if label_key and label_key in sample:
        label = sample[label_key]
        # Normalize label to harmful/benign
        if isinstance(label, (bool, int)):
            result["safety_label"] = "harmful" if label else "benign"
        elif isinstance(label, str):
            label_lower = label.lower()
            if label_lower in ("harmful", "unsafe", "dangerous", "toxic", "1", "true"):
                result["safety_label"] = "harmful"
            else:
                result["safety_label"] = "benign"
        else:
            result["safety_label"] = "unknown"

    return result


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} (split: {args.split})")

    try:
        dataset = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset '{args.dataset}': {e}")
        print("\nAvailable datasets for OPSA:")
        print("  - UWNSL/SafeChain (40K safe reasoning traces)")
        print("  - Seanie-lee/ThinkSafe-Qwen3-0.6B")
        print("  - Seanie-lee/ThinkSafe-Qwen3-1.7B")
        print("  - Seanie-lee/ThinkSafe-DeepSeek-R1-Distill-Qwen-1.5B")
        return

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")

    # Limit samples if requested
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples")

    # Detect keys
    sample = dataset[0]
    prompt_key = args.prompt_key or detect_keys(sample)[0]
    label_key = args.label_key or detect_keys(sample)[1]

    if prompt_key is None:
        print(f"ERROR: Could not auto-detect prompt key. Available keys: {list(sample.keys())}")
        print("Please specify --prompt_key explicitly.")
        return

    print(f"Using prompt_key='{prompt_key}', label_key='{label_key}'")

    # Process samples
    processed = []
    for i, sample in enumerate(dataset):
        formatted = format_sample(sample, prompt_key, label_key)
        processed.append(formatted)
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples...")

    print(f"Total processed: {len(processed)} samples")

    # Save as parquet (no train/val split: full data used for training,
    # matching the original OPSA setup)
    dataset_name = args.dataset.split("/")[-1].lower().replace("-", "_")

    # Store as 'prompt' column (list of chat messages) for verl RLHFDataset compatibility
    # Also include 'data_source' and 'reward_model' fields required by verl's reward pipeline.
    # OPSA is pure self-distillation and ignores rewards, but these fields must exist to
    # prevent KeyError in NaiveRewardManager.
    train_records = []
    for item in processed:
        safety_label = item.get("safety_label", "unknown")
        record = {
            "prompt": item["content"],  # list of dicts, e.g. [{"role": "user", "content": "..."}]
            "data_source": "opsa",
            "reward_model": {"ground_truth": safety_label, "style": "rule"},
        }
        if "safety_label" in item:
            record["safety_label"] = safety_label
        train_records.append(record)

    train_path = output_dir / f"{dataset_name}_train.parquet"
    pd.DataFrame(train_records).to_parquet(train_path, index=False)

    print("\nSaved:")
    print(f"  Train: {train_path} ({len(train_records)} samples)")

    # Print label distribution
    if label_key:
        labels = [r.get("safety_label", "unknown") for r in train_records]
        dist = Counter(labels)
        print("\nLabel distribution:")
        for label, count in sorted(dist.items()):
            print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")

    print("\nDone! Next steps:")
    print(f"  1. Update DATA_PATH in run_opsa.sh to: {train_path}")
    print("  2. Run: bash examples/opsa/run_opsa.sh")


if __name__ == "__main__":
    main()
