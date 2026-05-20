# Copyright 2026 EasyOPD Contributors
#
# Standalone loader for a teacher model's `lm_head` weight.
#
# Motivation: in `simple` cross-tokenizer KD the teacher's hidden states are
# computed on a remote SGLang actor pool (see `teacher_group.py`), but the
# projection from hidden states to logits (and the column-crop to the
# overlap sub-vocabulary) is done locally inside the student worker so the
# loss can be computed without shipping huge `[B, T, V_tea]` tensors over
# the wire. To do that, each student rank needs a frozen copy of the
# teacher's `lm_head.weight` (and possibly `embed_tokens.weight` when the
# teacher uses tied embeddings).
#
# This module loads JUST that one tensor from a HuggingFace checkpoint
# directory, handling the three relevant storage layouts:
#   1. Sharded safetensors with an `model.safetensors.index.json` map.
#   2. A single-file `model.safetensors`.
#   3. A single-file `pytorch_model.bin` (legacy torch checkpoint).
#
# It is intentionally lightweight: no full-model instantiation, so the
# memory footprint is exactly the lm_head matrix itself (~1 GB bf16 for
# Qwen2.5-7B's 152064 x 3584 vocab).

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


_LM_HEAD_CANDIDATE_KEYS = ("lm_head.weight",)
_TIED_FALLBACK_KEYS = (
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "embed_tokens.weight",
)


def _find_weight_in_index(index_json_path: str, candidate_keys) -> Optional[tuple[str, str]]:
    """Look up a tensor in a sharded safetensors index file.

    Returns:
        (matched_key, shard_filename) or None if no candidate key is in the index.
    """
    with open(index_json_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    for key in candidate_keys:
        if key in weight_map:
            return key, weight_map[key]
    return None


def _load_tensor_from_safetensors(shard_path: str, key: str, dtype: torch.dtype) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(shard_path, framework="pt") as f:
        if key not in f.keys():
            raise KeyError(f"key {key!r} not found in {shard_path}")
        tensor = f.get_tensor(key)
    return tensor.to(dtype=dtype)


def _load_tensor_from_torch_bin(bin_path: str, candidate_keys, dtype: torch.dtype) -> torch.Tensor:
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    for key in candidate_keys:
        if key in state:
            return state[key].to(dtype=dtype)
    raise KeyError(
        f"none of {candidate_keys!r} were found in legacy bin {bin_path}; "
        f"available top-level keys (sample): {list(state)[:8]}"
    )


def load_teacher_lm_head(
    teacher_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Linear:
    """Load only the `lm_head` (or tied embedding) of a HF teacher checkpoint.

    Args:
        teacher_path: path to a HuggingFace model directory containing
            `config.json` and one of:
              * `model.safetensors.index.json` + sharded `*.safetensors`,
              * a single `model.safetensors`,
              * a single `pytorch_model.bin`.
        dtype: target dtype of the loaded weight (bf16 recommended; fp32
            would double the memory footprint with no accuracy gain for
            inference-only projection).

    Returns:
        A frozen `nn.Linear(hidden_size, vocab_size, bias=False)` module
        whose weight equals the teacher's `lm_head.weight` (or, for tied
        models, a clone of `embed_tokens.weight`). Caller is responsible
        for moving the returned module to the desired device, e.g.
        `head.to(student_device)` after instantiation.

    Raises:
        FileNotFoundError: if no recognized checkpoint format is present.
        KeyError: if neither `lm_head.weight` nor any tied-fallback key
            can be located.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(teacher_path, trust_remote_code=True)
    tie_embeddings = bool(getattr(cfg, "tie_word_embeddings", False))

    # If embeddings are tied, lm_head.weight may be absent — fall back to
    # the embed_tokens key. We try lm_head first either way (some tied
    # models still serialize an `lm_head.weight`, in which case we use it).
    candidate_keys = list(_LM_HEAD_CANDIDATE_KEYS)
    if tie_embeddings:
        candidate_keys = list(_LM_HEAD_CANDIDATE_KEYS) + list(_TIED_FALLBACK_KEYS)

    weight: Optional[torch.Tensor] = None

    sharded_index = os.path.join(teacher_path, "model.safetensors.index.json")
    single_safetensors = os.path.join(teacher_path, "model.safetensors")
    single_pytorch_bin = os.path.join(teacher_path, "pytorch_model.bin")

    if os.path.isfile(sharded_index):
        match = _find_weight_in_index(sharded_index, candidate_keys)
        if match is None:
            raise KeyError(
                f"none of {candidate_keys!r} were found in "
                f"{sharded_index}; cannot load lm_head."
            )
        matched_key, shard_filename = match
        shard_path = os.path.join(teacher_path, shard_filename)
        logger.info(
            "[load_teacher_lm_head] loading %s from shard %s (dtype=%s)",
            matched_key,
            shard_filename,
            dtype,
        )
        weight = _load_tensor_from_safetensors(shard_path, matched_key, dtype)
    elif os.path.isfile(single_safetensors):
        from safetensors import safe_open

        with safe_open(single_safetensors, framework="pt") as f:
            available = set(f.keys())
            matched_key = next((k for k in candidate_keys if k in available), None)
            if matched_key is None:
                raise KeyError(
                    f"none of {candidate_keys!r} were found in "
                    f"{single_safetensors}; cannot load lm_head."
                )
            logger.info(
                "[load_teacher_lm_head] loading %s from %s (dtype=%s)",
                matched_key,
                single_safetensors,
                dtype,
            )
            weight = f.get_tensor(matched_key).to(dtype=dtype)
    elif os.path.isfile(single_pytorch_bin):
        logger.info(
            "[load_teacher_lm_head] loading lm_head from legacy %s (dtype=%s)",
            single_pytorch_bin,
            dtype,
        )
        weight = _load_tensor_from_torch_bin(single_pytorch_bin, candidate_keys, dtype)
    else:
        raise FileNotFoundError(
            f"no recognized checkpoint file under {teacher_path!r}; "
            f"expected one of: model.safetensors.index.json, "
            f"model.safetensors, pytorch_model.bin"
        )

    assert weight is not None
    if weight.dim() != 2:
        raise RuntimeError(
            f"loaded lm_head weight has unexpected shape {tuple(weight.shape)} "
            f"(expected 2D [vocab_size, hidden_size])."
        )

    vocab_size, hidden_size = weight.shape
    expected_vocab = int(getattr(cfg, "vocab_size", vocab_size))
    expected_hidden = int(getattr(cfg, "hidden_size", hidden_size))
    if vocab_size != expected_vocab or hidden_size != expected_hidden:
        logger.warning(
            "[load_teacher_lm_head] config says (vocab=%d, hidden=%d) but "
            "weight shape is (vocab=%d, hidden=%d); using weight shape.",
            expected_vocab,
            expected_hidden,
            vocab_size,
            hidden_size,
        )

    head = nn.Linear(hidden_size, vocab_size, bias=False)
    with torch.no_grad():
        # Clone so a tied-embedding fallback does not share storage with any
        # caller-held tensor; freeze for safety.
        head.weight.copy_(weight.detach().clone())
    head.weight.requires_grad_(False)
    return head


__all__ = ["load_teacher_lm_head"]
