"""Reward function for the EasyOPD simple cross-tokenizer KD example.

The simple recipe optimizes the cross-tokenizer distillation loss and sets
``distillation.distillation_loss.use_task_rewards=False``. Verl's PPO/GRPO
training loop still requires a scalar reward for every rollout, so this module
provides a neutral placeholder reward for dry-runs and KD-only training.
"""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Return a neutral task reward for KD-only training."""
    del data_source, solution_str, ground_truth, extra_info, kwargs
    return 0.0
