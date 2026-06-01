"""Tests for the prepare data pipeline (Phase 1 + Phase 2 mock)."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pandas as pd
import pytest


def test_phase1_tokenize_basic():
    """Phase 1 should produce intermediate parquet with expected columns."""
    from easyopd.methods.lightning_opd.data_curation.prepare import phase1_tokenize, parse_args

    # Create a fake input parquet
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "rollouts.parquet")
        output_dir = os.path.join(tmpdir, "output")

        df = pd.DataFrame({
            "messages": [
                [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "The answer is 4."},
                ],
                [
                    {"role": "user", "content": "What is 3+3?"},
                    {"role": "assistant", "content": "The answer is 6."},
                ],
            ]
        })
        df.to_parquet(input_path, index=False)

        # Use a real tokenizer path (we'll use a small one from HF or mock)
        # For CPU test, we mock the tokenizer
        from unittest.mock import MagicMock, patch

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "What is 2+2?"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        args = parse_args([
            "--tokenizer-path", "dummy",
            "--input-parquet", input_path,
            "--output-dir", output_dir,
            "--max-response-len", "10",
        ])

        with patch(
            "easyopd.methods.lightning_opd.data_curation.prepare.AutoTokenizer"
        ) as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            intermediate_path = os.path.join(output_dir, "rollouts-lightning_opd.parquet")
            phase1_tokenize(args, intermediate_path)

        # Verify output
        assert os.path.exists(intermediate_path)
        df_out = pd.read_parquet(intermediate_path)
        assert "prompt" in df_out.columns
        assert "response_tokens" in df_out.columns
        assert "response_length" in df_out.columns
        assert len(df_out) == 2


def test_phase1_truncation():
    """Phase 1 should truncate responses exceeding max_response_len."""
    from easyopd.methods.lightning_opd.data_curation.prepare import phase1_tokenize, parse_args

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "rollouts.parquet")
        output_dir = os.path.join(tmpdir, "output")

        df = pd.DataFrame({
            "messages": [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "A" * 100},
                ],
            ]
        })
        df.to_parquet(input_path, index=False)

        from unittest.mock import MagicMock, patch

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Hello"
        mock_tokenizer.encode.return_value = list(range(20))  # 20 tokens

        args = parse_args([
            "--tokenizer-path", "dummy",
            "--input-parquet", input_path,
            "--output-dir", output_dir,
            "--max-response-len", "10",
        ])

        with patch(
            "easyopd.methods.lightning_opd.data_curation.prepare.AutoTokenizer"
        ) as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            intermediate_path = os.path.join(output_dir, "rollouts-lightning_opd.parquet")
            phase1_tokenize(args, intermediate_path)

        df_out = pd.read_parquet(intermediate_path)
        assert df_out.iloc[0]["response_length"] == 10  # truncated


def test_phase1_skips_missing_assistant():
    """Phase 1 should skip rows without assistant messages."""
    from easyopd.methods.lightning_opd.data_curation.prepare import phase1_tokenize, parse_args

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "rollouts.parquet")
        output_dir = os.path.join(tmpdir, "output")

        df = pd.DataFrame({
            "messages": [
                [{"role": "user", "content": "Hello"}],  # no assistant
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ],
            ]
        })
        df.to_parquet(input_path, index=False)

        from unittest.mock import MagicMock, patch

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Hi"
        mock_tokenizer.encode.return_value = [1, 2, 3]

        args = parse_args([
            "--tokenizer-path", "dummy",
            "--input-parquet", input_path,
            "--output-dir", output_dir,
        ])

        with patch(
            "easyopd.methods.lightning_opd.data_curation.prepare.AutoTokenizer"
        ) as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            intermediate_path = os.path.join(output_dir, "rollouts-lightning_opd.parquet")
            phase1_tokenize(args, intermediate_path)

        df_out = pd.read_parquet(intermediate_path)
        assert len(df_out) == 1  # only the second row


def test_end_to_end_mock():
    """Phase 1 + Phase 2 (mocked) should produce complete parquet schema."""
    from easyopd.methods.lightning_opd.data_curation.prepare import (
        phase1_tokenize,
        parse_args,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "rollouts.parquet")
        output_dir = os.path.join(tmpdir, "output")

        df = pd.DataFrame({
            "messages": [
                [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "The answer is 4."},
                ],
            ]
        })
        df.to_parquet(input_path, index=False)

        from unittest.mock import MagicMock, patch

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "What is 2+2?"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        args = parse_args([
            "--tokenizer-path", "dummy",
            "--input-parquet", input_path,
            "--output-dir", output_dir,
            "--max-response-len", "10",
        ])

        with patch(
            "easyopd.methods.lightning_opd.data_curation.prepare.AutoTokenizer"
        ) as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            intermediate_path = os.path.join(output_dir, "rollouts-lightning_opd.parquet")
            phase1_tokenize(args, intermediate_path)

        # Verify schema
        df_out = pd.read_parquet(intermediate_path)
        required_cols = {"prompt", "response_tokens", "response_length", "metadata"}
        assert required_cols.issubset(set(df_out.columns))


def test_phase2_logprobs_mock_http():
    """Phase 2 should write teacher_log_probs using a mocked HTTP response."""
    from easyopd.methods.lightning_opd.data_curation.prepare import (
        phase2_logprobs,
        parse_args,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        intermediate_path = output_dir / "rollouts-lightning_opd.parquet"
        output_path = output_dir / "rollouts-lightning_opd-precomputed.parquet"

        df = pd.DataFrame({
            "prompt": ["What is 2+2?"],
            "label": ["0"],
            "response_tokens": [[10, 11, 12]],
            "response_length": [3],
            "metadata": [{"sft_teacher_id": "teacher-a", "opd_teacher_id": "teacher-a"}],
        })
        df.to_parquet(intermediate_path, index=False)

        from unittest.mock import MagicMock, patch

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2]

        args = parse_args([
            "--tokenizer-path", "dummy",
            "--input-parquet", str(Path(tmpdir) / "rollouts.parquet"),
            "--output-dir", str(output_dir),
            "--compute-teacher-logprobs",
            "--teacher-url", "http://teacher/v1/completions",
        ])

        response = {"prompt_logprobs": [{}, {}, {10: -0.1}, {11: -0.2}, {12: -0.3}]}
        with patch(
            "easyopd.methods.lightning_opd.data_curation.prepare.AutoTokenizer"
        ) as mock_auto, patch(
            "easyopd.methods.lightning_opd.data_curation.prepare.post_json",
            return_value=response,
        ):
            mock_auto.from_pretrained.return_value = mock_tokenizer
            phase2_logprobs(args, intermediate_path, output_path)

        df_out = pd.read_parquet(output_path)
        assert list(df_out.iloc[0]["teacher_log_probs"]) == [-0.1, -0.2, -0.3]
