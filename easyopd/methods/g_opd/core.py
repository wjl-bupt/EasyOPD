"""
G-OPD: Generalized On-Policy Distillation with Reward Extrapolation - Core Algorithm

This module implements the core G-OPD algorithm from:
    "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation"
    (https://arxiv.org/abs/2602.12125)

The key idea is to introduce a reward scaling factor (lambda) and a flexible reference model
to generalize standard OPD. Building on G-OPD, ExOPD (On-Policy Distillation with Reward
Extrapolation) outperforms standard OPD in both same-size and strong-to-weak distillation.

Key functions:
    - compute_g_opd_advantages: Computes G-OPD advantages with reward scaling.
    - compute_multi_teacher_advantages: Routes samples to different teachers.
    - compute_standard_opd_advantages: Standard OPD (lambda=1.0) as special case.
"""

import torch
from typing import Optional, Union


def compute_g_opd_advantages(
    old_log_probs: torch.Tensor,
    ref_log_prob: torch.Tensor,
    base_log_prob: torch.Tensor,
    lambda_vals: float = 1.0,
) -> torch.Tensor:
    """Compute G-OPD advantages with reward scaling factor.

    Implements the generalized on-policy distillation objective:
        When lambda_vals == 1.0 (standard OPD):
            reverse_kl = old_log_probs - ref_log_prob
        When lambda_vals != 1.0 (G-OPD / ExOPD):
            reverse_kl = (old_log_probs - base_log_prob) - (ref_log_prob - base_log_prob) * lambda_vals

    The advantage is then: A = -reverse_kl

    Args:
        old_log_probs: (batch_size, seq_len) student (actor) log-probs.
        ref_log_prob: (batch_size, seq_len) teacher (ref) log-probs.
        base_log_prob: (batch_size, seq_len) base/reference model log-probs for normalization.
        lambda_vals: Reward scaling factor.
            - lambda_vals=1.0: standard OPD
            - lambda_vals>1.0: ExOPD (reward extrapolation, amplifies teacher signal)
            - lambda_vals<1.0: G-OPD (dampens teacher signal)

    Returns:
        advantages: (batch_size, seq_len) computed advantages (negated reverse KL).
    """
    if lambda_vals == 1.0:
        # Standard OPD: directly use reverse KL
        reverse_kl = old_log_probs - ref_log_prob
    else:
        # G-OPD / ExOPD: apply reward scaling with base model normalization
        # reverse_kl = (pi_student / pi_base) - lambda * (pi_teacher / pi_base)
        reverse_kl = (old_log_probs - base_log_prob) - (ref_log_prob - base_log_prob) * lambda_vals

    # Advantage is the negation of reverse KL (minimizing KL = maximizing advantage)
    advantages = -reverse_kl
    return advantages


def compute_multi_teacher_advantages(
    old_log_probs: torch.Tensor,
    ref_log_prob: torch.Tensor,
    base_ref_log_prob: torch.Tensor,
    base_log_prob: torch.Tensor,
    opd_teacher: Union[list, tuple],
    lambda_vals: float = 1.0,
) -> torch.Tensor:
    """Compute advantages for multi-teacher distillation.

    In multi-teacher distillation, different samples are routed to different teachers
    based on the `opd_teacher` field in the data. Currently supports two teachers:
    - "math" teacher: uses ref_log_prob
    - "code" teacher: uses base_ref_log_prob

    Args:
        old_log_probs: (batch_size, seq_len) student log-probs.
        ref_log_prob: (batch_size, seq_len) math teacher log-probs.
        base_ref_log_prob: (batch_size, seq_len) code teacher log-probs.
        base_log_prob: (batch_size, seq_len) base model log-probs.
        opd_teacher: List/tuple of teacher type strings ("math" or "code") per sample.
        lambda_vals: Reward scaling factor.

    Returns:
        advantages: (batch_size, seq_len) computed advantages.
    """
    batch_size = old_log_probs.shape[0]
    reverse_kl = torch.zeros_like(old_log_probs)

    for i in range(batch_size):
        teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher

        if teacher_type == "math":
            if lambda_vals == 1.0:
                reverse_kl[i] = old_log_probs[i] - ref_log_prob[i]
            else:
                reverse_kl[i] = (old_log_probs[i] - base_log_prob[i]) - \
                    (ref_log_prob[i] - base_log_prob[i]) * lambda_vals
        elif teacher_type == "code":
            if lambda_vals == 1.0:
                reverse_kl[i] = old_log_probs[i] - base_ref_log_prob[i]
            else:
                reverse_kl[i] = (old_log_probs[i] - base_log_prob[i]) - \
                    (base_ref_log_prob[i] - base_log_prob[i]) * lambda_vals
        else:
            # Default: use math teacher
            reverse_kl[i] = old_log_probs[i] - ref_log_prob[i]

    advantages = -reverse_kl
    return advantages


def compute_standard_opd_advantages(
    old_log_probs: torch.Tensor,
    ref_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Compute standard OPD advantages (special case of G-OPD with lambda=1.0).

    This is the simplest form: A = -(old_log_probs - ref_log_prob) = ref_log_prob - old_log_probs

    Args:
        old_log_probs: (batch_size, seq_len) student log-probs.
        ref_log_prob: (batch_size, seq_len) teacher log-probs.

    Returns:
        advantages: (batch_size, seq_len) computed advantages.
    """
    return -(old_log_probs - ref_log_prob)
