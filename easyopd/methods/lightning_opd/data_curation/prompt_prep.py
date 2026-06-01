"""Prepare SFT prompts from HuggingFace datasets for Lightning-OPD.

Extracts prompt-only data (user messages) from a HF dataset or local
file and writes JSONL for downstream SFT data generation.

Usage:
    python3 -m easyopd.methods.lightning_opd.data_curation.prompt_prep \\
        --output data/prompts/openthoughts3_300k.jsonl \\
        --num-samples 300000

    # Use a local parquet file
    python3 -m easyopd.methods.lightning_opd.data_curation.prompt_prep \\
        --input-parquet data/raw/local.parquet \\
        --output data/prompts/local_300k.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Extract prompts from HF datasets for SFT data generation."
    )
    parser.add_argument(
        "--input-parquet", type=str, default=None,
        help="Path to a local parquet file. If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--hf-dataset", type=str, default="open-thoughts/OpenThoughts3-1.2M",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=300000,
        help="Number of samples to keep (default: 300000). Set to 0 for all.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args(args)


def extract_prompt(sample: dict) -> dict | None:
    """Extract the prompt (non-assistant messages) from a sample."""
    if "conversations" in sample:
        messages = []
        for turn in sample["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                messages.append({"role": "user", "content": content})
            elif role == "system":
                messages.append({"role": "system", "content": content})
        if messages:
            return {"prompt": messages}

    if "prompt" in sample:
        if isinstance(sample["prompt"], list):
            return {"prompt": sample["prompt"]}
        elif isinstance(sample["prompt"], str):
            return {"prompt": [{"role": "user", "content": sample["prompt"]}]}

    if "messages" in sample:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in sample["messages"]
            if m["role"] != "assistant"
        ]
        if messages:
            return {"prompt": messages}

    return None


def main(args=None):
    parsed = parse_args(args)
    random.seed(parsed.seed)

    if parsed.input_parquet:
        import pandas as pd
        logger.info("Loading from local file: %s", parsed.input_parquet)
        df = pd.read_parquet(parsed.input_parquet)
        samples = df.to_dict("records")
    else:
        from datasets import load_dataset
        logger.info("Loading from HuggingFace: %s", parsed.hf_dataset)
        ds = load_dataset(parsed.hf_dataset, split="train")
        samples = list(ds)

    logger.info("Total samples: %d", len(samples))

    if parsed.num_samples > 0 and parsed.num_samples < len(samples):
        indices = random.sample(range(len(samples)), parsed.num_samples)
        indices.sort()
        samples = [samples[i] for i in indices]
        logger.info("Sampled %d samples", parsed.num_samples)

    written = 0
    skipped = 0
    Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
    with open(parsed.output, "w") as f:
        for sample in samples:
            prompt_item = extract_prompt(sample)
            if prompt_item and len(prompt_item["prompt"]) > 0:
                f.write(json.dumps(prompt_item) + "\n")
                written += 1
            else:
                skipped += 1

    logger.info("Written: %d, Skipped: %d", written, skipped)
    logger.info("Output: %s", parsed.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
