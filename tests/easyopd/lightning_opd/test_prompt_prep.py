"""Tests for Lightning-OPD prompt preparation."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_prompt_prep_creates_output_parent():
    from easyopd.methods.lightning_opd.data_curation.prompt_prep import main

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.parquet"
        output_path = Path(tmpdir) / "nested" / "prompts.jsonl"
        pd.DataFrame({"prompt": ["What is 2+2?"]}).to_parquet(input_path, index=False)

        main([
            "--input-parquet", str(input_path),
            "--output", str(output_path),
            "--num-samples", "0",
        ])

        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert row["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
