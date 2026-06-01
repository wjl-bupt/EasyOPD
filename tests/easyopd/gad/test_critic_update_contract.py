"""Contract tests for easyopd.methods.gad.critic_update.update_critic_step.

We do not test numerical equivalence to the verl PPO critic update.
We test that the function:
  (a) runs student and teacher forwards via the injected critic worker,
  (b) calls loss.backward and an optimizer step,
  (c) returns a flat dict of metrics containing the expected keys
      (matching verl's own update_critic return contract).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf


class _FakeDataProto:
    """A minimal stand-in for verl.protocol.DataProto in CPU tests."""

    def __init__(self, batch: dict, meta_info: dict | None = None):
        self.batch = dict(batch)
        self.non_tensor_batch: dict = {}
        self.meta_info = meta_info or {}

    def select(self, batch_keys=None, non_tensor_batch_keys=None):
        if batch_keys is None:
            return _FakeDataProto(self.batch, self.meta_info)
        return _FakeDataProto({k: self.batch[k] for k in batch_keys if k in self.batch}, self.meta_info)

    def split(self, micro_batch_size: int):
        # Single split for the contract test.
        return [self]


def _make_worker(monkeypatch):
    """Build a mock critic worker the way dp_critic uses self."""

    optimizer = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()

    # Make _forward_micro_batch return a tensor where the LAST token has 1.0
    # (student) or 2.0 (teacher), as if last_token_only has already been applied.
    def fake_forward(self, micro_batch, *, compute_teacher: bool = False):
        bsz, t = micro_batch["input_ids"].shape[0], micro_batch["input_ids"].shape[-1]
        out = torch.zeros(bsz, t, requires_grad=True)
        with torch.no_grad():
            scale = 2.0 if compute_teacher else 1.0
            out_l = out.clone()
            out_l[:, -1] = scale
        out_l.requires_grad_(True)
        return out_l

    worker = SimpleNamespace(
        config=OmegaConf.create(
            {
                "gad": {"enable": True, "discriminator_init_path": "/tmp/x"},
                "ppo_mini_batch_size": 2,
                "use_dynamic_bsz": False,
                "ppo_micro_batch_size_per_gpu": 2,
            }
        ),
        critic_optimizer=optimizer,
        critic_module=MagicMock(),  # GradientAccumulator-like; train/eval will be called
        gradient_accumulation=1,
        ulysses_sequence_parallel_size=1,
        device_name="cpu",
        _forward_micro_batch=lambda mb, **kw: fake_forward(None, mb, **kw),
        _optimizer_step=lambda: torch.tensor(1.0),
    )
    return worker, optimizer


def _make_data():
    return _FakeDataProto(
        batch={
            "input_ids": torch.arange(8).reshape(2, 4).long(),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "position_ids": torch.zeros(2, 4, dtype=torch.long),
            "responses": torch.zeros(2, 2, dtype=torch.long),
            "response_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            "teacher_input_ids": torch.arange(10).reshape(2, 5).long() + 100,
            "teacher_attention_mask": torch.ones(2, 5, dtype=torch.long),
            "teacher_position_ids": torch.zeros(2, 5, dtype=torch.long),
            "teacher_response": torch.arange(6).reshape(2, 3).long() + 200,
        },
        meta_info={
            "micro_batch_size": 2,
            "use_dynamic_bsz": False,
            "max_token_len": 1024,
        },
    )


def test_update_step_returns_required_metrics(monkeypatch):
    from easyopd.methods.gad.critic_update import update_critic_step

    worker, optimizer = _make_worker(monkeypatch)
    metrics = update_critic_step(worker, _make_data())

    for key in (
        "critic/d_loss",
        "critic/d_acc",
        "critic/student_value_mean",
        "critic/teacher_value_mean",
        "critic/grad_norm",
    ):
        assert key in metrics, f"missing metric {key} in {metrics}"
    assert isinstance(metrics, dict), f"expected dict return, got {type(metrics)}"


def test_update_step_validates_data_contract(monkeypatch):
    from easyopd.methods.gad.critic_update import update_critic_step
    from easyopd.methods.gad.data_contract import GADBatchContractError

    worker, _ = _make_worker(monkeypatch)
    bad = _make_data()
    del bad.batch["teacher_input_ids"]
    with pytest.raises(GADBatchContractError):
        update_critic_step(worker, bad)
