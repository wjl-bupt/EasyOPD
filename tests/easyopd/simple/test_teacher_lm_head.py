# Copyright 2026 EasyOPD Contributors
#
# Tests for `load_teacher_lm_head`.
#
# Builds a tiny fake HF-style checkpoint on disk in three layouts
# (sharded safetensors / single safetensors / legacy pytorch_model.bin),
# verifies the loader picks the right tensor in each case, and exercises
# the tied-embedding fallback path.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _write_config(dirpath: Path, *, vocab_size: int, hidden_size: int, tie: bool):
    cfg = {
        "model_type": "llama",  # arbitrary; AutoConfig accepts known model_types
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "intermediate_size": 4,
        "tie_word_embeddings": tie,
    }
    (dirpath / "config.json").write_text(json.dumps(cfg))


@pytest.fixture
def vocab_hidden():
    return 64, 16


def test_load_from_sharded_safetensors(tmp_path: Path, vocab_hidden):
    safetensors = pytest.importorskip("safetensors.torch")
    from easyopd.methods.simple.teacher_lm_head import load_teacher_lm_head

    vocab, hidden = vocab_hidden
    ck = tmp_path / "ckpt"
    ck.mkdir()
    _write_config(ck, vocab_size=vocab, hidden_size=hidden, tie=False)

    head_w = torch.randn(vocab, hidden, dtype=torch.float32)
    other_w = torch.randn(hidden, hidden, dtype=torch.float32)
    safetensors.save_file({"lm_head.weight": head_w}, str(ck / "model-00001-of-00002.safetensors"))
    safetensors.save_file(
        {"model.layers.0.self_attn.q_proj.weight": other_w},
        str(ck / "model-00002-of-00002.safetensors"),
    )
    index = {
        "metadata": {},
        "weight_map": {
            "lm_head.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        },
    }
    (ck / "model.safetensors.index.json").write_text(json.dumps(index))

    head = load_teacher_lm_head(str(ck), dtype=torch.float32)

    assert isinstance(head, torch.nn.Linear)
    assert head.weight.shape == (vocab, hidden)
    assert head.bias is None
    assert head.weight.requires_grad is False
    assert torch.allclose(head.weight, head_w)


def test_load_from_single_safetensors(tmp_path: Path, vocab_hidden):
    safetensors = pytest.importorskip("safetensors.torch")
    from easyopd.methods.simple.teacher_lm_head import load_teacher_lm_head

    vocab, hidden = vocab_hidden
    ck = tmp_path / "ckpt"
    ck.mkdir()
    _write_config(ck, vocab_size=vocab, hidden_size=hidden, tie=False)

    head_w = torch.randn(vocab, hidden, dtype=torch.float32)
    safetensors.save_file({"lm_head.weight": head_w}, str(ck / "model.safetensors"))

    head = load_teacher_lm_head(str(ck), dtype=torch.float32)
    assert torch.allclose(head.weight, head_w)


def test_load_from_pytorch_bin(tmp_path: Path, vocab_hidden):
    from easyopd.methods.simple.teacher_lm_head import load_teacher_lm_head

    vocab, hidden = vocab_hidden
    ck = tmp_path / "ckpt"
    ck.mkdir()
    _write_config(ck, vocab_size=vocab, hidden_size=hidden, tie=False)

    head_w = torch.randn(vocab, hidden, dtype=torch.float32)
    torch.save({"lm_head.weight": head_w}, str(ck / "pytorch_model.bin"))

    head = load_teacher_lm_head(str(ck), dtype=torch.float32)
    assert torch.allclose(head.weight, head_w)


def test_tied_embeddings_fallback(tmp_path: Path, vocab_hidden):
    """When tie_word_embeddings=True and lm_head.weight is absent, fall back
    to model.embed_tokens.weight."""
    safetensors = pytest.importorskip("safetensors.torch")
    from easyopd.methods.simple.teacher_lm_head import load_teacher_lm_head

    vocab, hidden = vocab_hidden
    ck = tmp_path / "ckpt"
    ck.mkdir()
    _write_config(ck, vocab_size=vocab, hidden_size=hidden, tie=True)

    embed_w = torch.randn(vocab, hidden, dtype=torch.float32)
    safetensors.save_file({"model.embed_tokens.weight": embed_w}, str(ck / "model.safetensors"))

    head = load_teacher_lm_head(str(ck), dtype=torch.float32)
    assert torch.allclose(head.weight, embed_w)
    # Mutating the loaded head should not propagate back to the on-disk
    # tensor (sanity check the .clone() in the loader).
    with torch.no_grad():
        head.weight.add_(1.0)
    assert not torch.allclose(head.weight, embed_w)


def test_missing_checkpoint_raises(tmp_path: Path, vocab_hidden):
    from easyopd.methods.simple.teacher_lm_head import load_teacher_lm_head

    vocab, hidden = vocab_hidden
    ck = tmp_path / "ckpt"
    ck.mkdir()
    _write_config(ck, vocab_size=vocab, hidden_size=hidden, tie=False)
    # No weight files at all.
    with pytest.raises(FileNotFoundError):
        load_teacher_lm_head(str(ck))
