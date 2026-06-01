"""On-policy distillation advantage estimator for Lightning-OPD.

Computes per-token advantage = teacher_log_prob - student_log_prob.
This is the exact KL gradient surrogate for the distillation objective
given the offline teacher consistency condition (arXiv:2604.13010 §3).
"""

from __future__ import annotations

import torch

from verl.trainer.ppo.core_algos import register_adv_est


class LightningOPDMissingTeacherLogprobs(RuntimeError):
    """Raised when teacher_log_probs is missing from the batch."""


@register_adv_est("on_policy_distillation")
def compute_on_policy_distillation_advantages(
    *,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_log_probs: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token advantage = teacher_log_prob - student_log_prob.

    Lightning-OPD paper §3: this signal is the exact KL gradient surrogate
    for the distillation objective, given the offline teacher consistency
    condition.

    Args:
        token_level_rewards: Shape ``(B, T)``. For pure distillation this is
            all-zeros (no task reward). Included for interface compatibility.
        response_mask: Shape ``(B, T)``. 1 for valid response tokens, 0 for
            padding / prompt.
        teacher_log_probs: Shape ``(B, T)``. Precomputed offline teacher
            log-probabilities aligned with the response tokens.
        old_log_probs: Shape ``(B, T)``. Student log-probabilities from the
            actor rollout path. Preferred over ``token_level_rewards`` when
            available because Lightning-OPD does not use task rewards to define
            its advantage signal.
        config: Algorithm config (unused, kept for interface compatibility).
        **kwargs: Ignored; absorbs extra args from the generic adv dispatcher.

    Returns:
        ``(advantages, returns)`` both of shape ``(B, T)``.

    Raises:
        LightningOPDMissingTeacherLogprobs: If ``teacher_log_probs`` is None.
    """
    if teacher_log_probs is None:
        raise LightningOPDMissingTeacherLogprobs(
            "teacher_log_probs not found in batch. For Lightning-OPD the training "
            "data parquet must contain a 'teacher_log_probs' column produced by "
            "the prepare_data pipeline."
        )

    student_log_probs = old_log_probs if old_log_probs is not None else token_level_rewards

    advantages = (teacher_log_probs - student_log_probs) * response_mask
    returns = advantages.clone()
    return advantages, returns
