"""Shared fixtures for GAD CPU contract tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch


@pytest.fixture
def tiny_micro_batch():
    """A dict-of-tensors mimicking a single micro-batch on CPU."""
    bsz, t_s, t_t = 2, 4, 5
    return {
        "input_ids": torch.arange(bsz * t_s).reshape(bsz, t_s).long(),
        "attention_mask": torch.ones(bsz, t_s, dtype=torch.long),
        "position_ids": torch.arange(t_s).unsqueeze(0).expand(bsz, t_s).long(),
        "responses": torch.arange(bsz * 2).reshape(bsz, 2).long(),  # response_length = 2
        "teacher_input_ids": (torch.arange(bsz * t_t) + 100).reshape(bsz, t_t).long(),
        "teacher_attention_mask": torch.ones(bsz, t_t, dtype=torch.long),
        "teacher_position_ids": torch.arange(t_t).unsqueeze(0).expand(bsz, t_t).long(),
        "teacher_response": (torch.arange(bsz * 3) + 200).reshape(bsz, 3).long(),  # teacher_response_length = 3
    }


@pytest.fixture
def constant_logits_module():
    """A torch.nn.Module returning a fixed-shape logits/values tensor.

    Mimics the minimal interface that dp_critic._forward_micro_batch
    uses: a callable that returns either a value head output or a model
    with `.logits`.
    """

    class _Out:
        def __init__(self, logits: torch.Tensor):
            self.logits = logits

    class _Mod(torch.nn.Module):
        def __init__(self, response_length: int):
            super().__init__()
            self.response_length = response_length
            self.calls: list[dict[str, Any]] = []

        def forward(self, **kwargs):
            self.calls.append({k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)})
            ids = kwargs["input_ids"]
            bsz, seqlen = ids.shape[0], ids.shape[-1]
            # constant per-token scalar (use 1.0 for student inputs, 2.0 if input shifted by 100, marker for teacher)
            base = 2.0 if ids.float().mean() >= 50 else 1.0
            return _Out(logits=torch.full((bsz, seqlen, 1), base))

    return _Mod
