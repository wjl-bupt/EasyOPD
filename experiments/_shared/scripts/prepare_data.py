"""Prepare datasets for EasyOPD benchmark experiments.

Converts:
1. mixed_math_code_10k -> training parquet (with prompt, data_source, reward_model fields)
2. math500_test -> evaluation parquet
3. gsm8k test split -> evaluation parquet
"""

import os
import re
import json
import pandas as pd
from pathlib import Path

# Paths
DATASET_ROOT = "/path/to/workspace/workspace/dataset"
# Eval data is shared across all methods (math500/gsm8k/math_hard etc.)
EVAL_DATA_DIR = "/path/to/EasyOPD/experiments/_shared/eval_data"
# Training data is method-specific; this is just a fallback dump location for
# the legacy train/val/sft_train parquet (no longer used by the new SFT pipeline,
# kept for backward compatibility of older scripts).
TRAIN_DATA_DIR = EVAL_DATA_DIR  # same dir is fine; train.parquet & val.parquet just live alongside
MODEL_PATH = "/path/to/workspace/workspace/models/Qwen2.5-1.5B-Instruct"

os.makedirs(EVAL_DATA_DIR, exist_ok=True)


def prepare_training_data():
    """Prepare mixed_math_code_10k as training data in verl format."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading mixed_math_code_10k...")
    ds = load_from_disk(os.path.join(DATASET_ROOT, "mixed_math_code_10k"))

    records = []
    for i, item in enumerate(ds):
        messages = item["messages"]
        label = item["label"]

        # Build prompt from messages (user turn only)
        # messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        user_msg = messages[0]["content"] if messages else ""

        # Apply chat template for the prompt
        prompt_messages = [{"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Determine data source based on content
        data_source = "math" if any(kw in user_msg.lower() for kw in ["solve", "find", "calculate", "prove", "\\boxed"]) else "code"

        records.append({
            "prompt": prompt,
            "data_source": data_source,
            "reward_model": {"ground_truth": label},
        })

    # Split into train (9900) and val (50)
    train_records = records[:9900]
    val_records = records[9900:9950]

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    train_path = os.path.join(TRAIN_DATA_DIR, "train.parquet")
    val_path = os.path.join(TRAIN_DATA_DIR, "val.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    print(f"Training data: {len(train_df)} samples -> {train_path}")
    print(f"Validation data: {len(val_df)} samples -> {val_path}")


def prepare_math500_eval():
    """Prepare MATH-500 (actually MATH-Hard, 1324 problems) for evaluation."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading math500_test (MATH-Hard)...")
    ds = load_from_disk(os.path.join(DATASET_ROOT, "math500_test"))

    records = []
    for item in ds:
        problem = item["problem"]
        solution = item["solution"]

        # Extract ground truth from \boxed{}
        boxed_match = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
        ground_truth = boxed_match[-1] if boxed_match else ""

        prompt_messages = [{"role": "user", "content": f"Solve the following math problem. Put your final answer in \\boxed{{}}.\n\n{problem}"}]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        records.append({
            "prompt": prompt,
            "data_source": "math",
            "reward_model": {"ground_truth": ground_truth},
            "level": item["level"],
            "type": item["type"],
        })

    df = pd.DataFrame(records)
    # Take first 500 for MATH-500 benchmark
    df_500 = df.head(500)
    out_path = os.path.join(EVAL_DATA_DIR, "math500_eval.parquet")
    df_500.to_parquet(out_path)
    print(f"MATH-500 eval: {len(df_500)} samples -> {out_path}")

    # Also save full MATH-Hard
    out_path_full = os.path.join(EVAL_DATA_DIR, "math_hard_eval.parquet")
    df.to_parquet(out_path_full)
    print(f"MATH-Hard eval: {len(df)} samples -> {out_path_full}")


def prepare_gsm8k_eval():
    """Prepare GSM8K test split for evaluation."""
    from datasets import load_from_disk, DatasetDict
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading gsm8k test split...")
    ds = load_from_disk(os.path.join(DATASET_ROOT, "gsm8k_for_kdflow"))

    # The dataset has train and test splits
    if isinstance(ds, DatasetDict):
        test_ds = ds['test']
    elif hasattr(ds, 'keys') and callable(ds.keys) and 'test' in ds.keys():
        test_ds = ds['test']
    else:
        # Single split - take last 1319 samples (GSM8K test size)
        test_ds = ds.select(range(max(0, len(ds) - 1319), len(ds)))

    records = []
    for item in test_ds:
        messages = item["messages"]
        label = item["label"]

        # Extract the user question
        user_msg = messages[0]["content"] if messages else ""

        prompt_messages = [{"role": "user", "content": f"Solve the following math problem step by step. At the end, provide your answer in the format: #### <number>\n\n{user_msg}"}]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        records.append({
            "prompt": prompt,
            "data_source": "gsm8k",
            "reward_model": {"ground_truth": label},
        })

    df = pd.DataFrame(records)
    out_path = os.path.join(EVAL_DATA_DIR, "gsm8k_eval.parquet")
    df.to_parquet(out_path)
    print(f"GSM8K eval: {len(df)} samples -> {out_path}")


def prepare_sft_data():
    """Prepare SFT training data from mixed_math_code_10k sharegpt format."""
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Use the sharegpt JSON which has conversations with teacher responses
    sharegpt_path = os.path.join(DATASET_ROOT, "mixed_math_code_10k", "train_sharegpt.json")
    print(f"Loading SFT data from {sharegpt_path}...")

    with open(sharegpt_path) as f:
        data = json.load(f)

    records = []
    for item in data:
        convs = item.get("conversations", [])
        if len(convs) >= 2:
            user_msg = convs[0]["value"]
            assistant_msg = convs[1]["value"]

            # Extract ground truth from assistant response
            # For math: look for #### or \boxed{}
            label = ""
            import re as _re
            boxed = _re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', assistant_msg)
            if boxed:
                label = boxed[-1]
            else:
                # GSM8K style: #### <number>
                gsm_match = _re.findall(r'####\s*(\-?[0-9.,]+)', assistant_msg)
                if gsm_match:
                    label = gsm_match[-1].replace(',', '')

            prompt_messages = [{"role": "user", "content": user_msg}]
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            records.append({
                "prompt": prompt,
                "response": assistant_msg,
                "data_source": "math",
                "reward_model": {"ground_truth": label},
            })

    df = pd.DataFrame(records)
    out_path = os.path.join(TRAIN_DATA_DIR, "sft_train.parquet")
    df.to_parquet(out_path)
    print(f"SFT training data: {len(df)} samples -> {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        task = sys.argv[1]
        if task == "train":
            prepare_training_data()
        elif task == "math500":
            prepare_math500_eval()
        elif task == "gsm8k":
            prepare_gsm8k_eval()
        elif task == "sft":
            prepare_sft_data()
        else:
            print(f"Unknown task: {task}")
    else:
        prepare_training_data()
        prepare_math500_eval()
        prepare_gsm8k_eval()
        prepare_sft_data()
        print("\n=== All data prepared! ===")
