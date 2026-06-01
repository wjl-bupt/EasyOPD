"""Tests for the Lightning-OPD response generation helper."""

import json
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_generate_responses_writes_messages_parquet():
    from easyopd.methods.lightning_opd.data_curation.generate_responses import (
        generate_responses,
        parse_args,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "prompts.jsonl")
        output_path = os.path.join(tmpdir, "responses.parquet")

        with open(input_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": [{"role": "user", "content": "What is 2+2?"}]}) + "\n")

        args = parse_args([
            "--input-prompts", input_path,
            "--output-parquet", output_path,
            "--model", "teacher-a",
            "--endpoint", "http://teacher/v1/chat/completions",
            "--max-tokens", "16",
            "--concurrency", "1",
        ])

        def fake_post_json(url, payload, timeout):
            assert url == "http://teacher/v1/chat/completions"
            assert payload["model"] == "teacher-a"
            return {"choices": [{"message": {"content": "The answer is 4."}}]}

        generate_responses(args, post_json_fn=fake_post_json)

        df = pd.read_parquet(output_path)
        assert len(df) == 1
        messages = df.iloc[0]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[-1] == {"role": "assistant", "content": "The answer is 4."}
        assert df.iloc[0]["metadata"]["model"] == "teacher-a"
