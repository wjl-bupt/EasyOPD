#!/usr/bin/env python3
"""Convert the KDFlow ``mixed_math_code_10k`` HuggingFace dataset into the
parquet layout that verl's ``RLHFDataset`` expects.

The source dataset (saved via ``datasets.Dataset.save_to_disk``) has columns:

* ``messages``: ``list[{"role": str, "content": str}]`` (single-turn user prompt)
* ``label``:    ``str`` (reference answer / ground truth)

verl's ``RLHFDataset`` reads ``prompt`` (a list of chat messages) and applies
the tokenizer's chat template internally. We therefore copy ``messages`` into
``prompt`` verbatim, surface ``label`` as ``reward_model.ground_truth`` so the
field can be picked up by reward implementations later, and tag every row with
``data_source`` plus an ``extra_info.index`` for traceability.

Usage::

    python prepare_data.py \
        --src /path/to/dataset/mixed_math_code_10k \
        --dst ~/data/mixed_math_code_10k \
        --val-size 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="/path/to/workspace/workspace/dataset/mixed_math_code_10k",
        help="Source HF dataset directory (output of save_to_disk).",
    )
    parser.add_argument(
        "--dst",
        default=os.path.expanduser("~/data/mixed_math_code_10k"),
        help="Destination directory for the generated parquet files.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Number of trailing rows to peel off as the validation split.",
    )
    parser.add_argument(
        "--data-source",
        default="mixed_math_code_10k",
        help="String stored in the ``data_source`` column.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed (set to a negative value to disable shuffling).",
    )
    return parser.parse_args()


def to_verl_row(example: dict, idx: int, data_source: str) -> dict:
    """Map a single KDFlow row to verl's ``RLHFDataset`` schema."""
    messages = example["messages"]
    # Defensive copy: ensure each turn is a plain dict[str, str].
    prompt = [
        {"role": str(turn["role"]), "content": str(turn["content"])}
        for turn in messages
    ]
    ground_truth = example.get("label", "")
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "mixed",
        "reward_model": {
            "style": "model",
            "ground_truth": "" if ground_truth is None else str(ground_truth),
        },
        "extra_info": {
            "split": "train",  # overwritten in the val split below
            "index": idx,
        },
    }


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_data] loading {src}")
    ds = load_from_disk(str(src))
    print(f"[prepare_data] rows={len(ds)} cols={ds.column_names}")

    if args.seed >= 0:
        ds = ds.shuffle(seed=args.seed)

    n = len(ds)
    val_size = max(0, min(args.val_size, n - 1))
    train_size = n - val_size
    print(f"[prepare_data] train={train_size} val={val_size}")

    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, n)) if val_size > 0 else None

    train_mapped = train_ds.map(
        lambda ex, idx: to_verl_row(ex, idx, args.data_source),
        with_indices=True,
        remove_columns=ds.column_names,
        desc="train -> verl",
    )

    train_path = dst / "train.parquet"
    train_mapped.to_parquet(str(train_path))
    print(f"[prepare_data] wrote {train_path} ({len(train_mapped)} rows)")

    if val_ds is not None:
        val_mapped = val_ds.map(
            lambda ex, idx: {
                **to_verl_row(ex, idx + train_size, args.data_source),
                "extra_info": {"split": "val", "index": idx + train_size},
            },
            with_indices=True,
            remove_columns=ds.column_names,
            desc="val -> verl",
        )
        val_path = dst / "test.parquet"
        val_mapped.to_parquet(str(val_path))
        print(f"[prepare_data] wrote {val_path} ({len(val_mapped)} rows)")


if __name__ == "__main__":
    main()
