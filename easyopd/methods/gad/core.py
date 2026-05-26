"""Pure tensor primitives for GAD.

No verl, no DataProto, no I/O. Everything in this module is unit-
testable on CPU with small tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def summed_reward(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Sum masked per-token scores into a per-sequence scalar.

    Args:
        values: shape (B, T), per-token discriminator output.
        response_mask: shape (B, T), 1.0 on valid positions, 0.0 elsewhere.

    Returns:
        Tensor of shape (B,).
    """
    return (values * response_mask).sum(dim=-1)


def compute_discriminator_loss(
    student_vpreds: torch.Tensor,
    teacher_vpreds: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise loss for discriminator.

    Loss = -mean(log_sigmoid(teacher_reward - student_reward))
    where each reward is the masked sum over its sequence.

    Lifted from `YTianZHU/verl@gad:verl/trainer/ppo/core_algos.py:832-836`.
    """
    teacher_reward = summed_reward(teacher_vpreds, teacher_response_mask)
    student_reward = summed_reward(student_vpreds, response_mask)
    return -F.logsigmoid(teacher_reward - student_reward).mean()


def discriminator_accuracy(
    student_vpreds: torch.Tensor,
    teacher_vpreds: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor,
) -> float:
    """Fraction of (s, t) pairs where teacher_sum > student_sum."""
    teacher_reward = summed_reward(teacher_vpreds, teacher_response_mask)
    student_reward = summed_reward(student_vpreds, response_mask)
    return (teacher_reward > student_reward).float().mean().item()


def last_token_only(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Zero out every position except the last valid one in each row.

    Used to turn the critic's per-token output into a sequence-level
    score localized at the final response token. Rows with an all-zero
    mask become all-zero rows (no spurious score).
    """
    response_lengths = response_mask.sum(dim=1).long()  # (B,)
    has_valid = response_lengths > 0  # (B,)
    last_token_indices = (response_lengths - 1).clamp(min=0)  # (B,)

    last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
    last_token_mask[batch_indices, last_token_indices] = True
    last_token_mask[~has_valid] = False  # rows with no valid token: keep all zeros

    return values * last_token_mask.type_as(values)
