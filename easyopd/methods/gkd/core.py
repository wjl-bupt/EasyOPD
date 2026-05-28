"""
GKD: Generalized Knowledge Distillation (On-Policy Distillation)

Core algorithm implementation from:
    "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
    Agarwal et al., ICLR 2024
    https://arxiv.org/abs/2306.13649

The key idea is to combine on-policy sampling (student generates) with dense
teacher feedback (per-token KL divergence), using a Generalized Jensen-Shannon
Divergence (JSD) that interpolates between forward and reverse KL.

Key functions:
    - generalized_jsd: Computes the generalized JSD loss between student and teacher.
    - gkd_loss: Full GKD loss with temperature scaling and masking.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def generalized_jsd(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute Generalized Jensen-Shannon Divergence.

    Following the paper (Agarwal et al., ICLR 2024, Eq. 1):

        D_beta[pi_T, pi_S] = beta * KL(pi_T || pi_S) + (1 - beta) * KL(pi_S || pi_T)

    where pi_T = teacher, pi_S = student.

    Special cases:
        - beta = 0.0: Pure forward KL = KL(student || teacher)
                      (mean-seeking, student covers all teacher modes)
        - beta = 0.5: Symmetric JSD
        - beta = 1.0: Pure reverse KL = KL(teacher || student)
                      (mode-seeking, student concentrates on teacher's high-prob regions)

    Args:
        student_log_probs: Student model log-probabilities over vocabulary.
            Shape: (batch_size, seq_len, vocab_size) or (batch_size, vocab_size).
        teacher_log_probs: Teacher model log-probabilities over vocabulary.
            Shape: same as student_log_probs.
        beta: Interpolation parameter (paper convention).
            beta=0 -> forward KL, beta=1 -> reverse KL, beta=0.5 -> symmetric JSD.

    Returns:
        Per-token JSD values. Shape: (batch_size, seq_len) or (batch_size,).
    """
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()

    # Forward KL: KL(student || teacher) = sum_i p_s(i) * (log p_s(i) - log p_t(i))
    forward_kl = torch.sum(
        student_probs * (student_log_probs - teacher_log_probs),
        dim=-1,
    )

    # Reverse KL: KL(teacher || student) = sum_i p_t(i) * (log p_t(i) - log p_s(i))
    reverse_kl = torch.sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        dim=-1,
    )

    # Generalized JSD (paper Eq. 1): beta * RKL + (1-beta) * FKL
    jsd = beta * reverse_kl + (1.0 - beta) * forward_kl

    return jsd


def generalized_jsd_from_estimator(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute Generalized JSD using single-sample KL estimators.

    Following the paper convention (Eq. 1):
        D_beta = beta * KL(teacher || student) + (1-beta) * KL(student || teacher)

    When only per-token log-probabilities (not full distributions) are available,
    we use single-sample estimators:
        - Forward KL estimator (k1): log p_s - log p_t
        - Reverse KL low-variance estimator (k3): exp(log p_t - log p_s) - (log p_t - log p_s) - 1

    Args:
        student_log_prob: Per-token student log-probability.
            Shape: (batch_size, seq_len).
        teacher_log_prob: Per-token teacher log-probability.
            Shape: (batch_size, seq_len).
        beta: Interpolation parameter (paper convention).
            beta=0 -> forward KL, beta=1 -> reverse KL.

    Returns:
        Per-token JSD estimates. Shape: (batch_size, seq_len).
    """
    # Forward KL single-sample estimator (k1): log p_s - log p_t
    forward_kl_est = student_log_prob - teacher_log_prob

    # Reverse KL low-variance estimator (k3): exp(log_ratio) - log_ratio - 1
    log_ratio = teacher_log_prob - student_log_prob
    log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)
    reverse_kl_est = torch.exp(log_ratio_clamped) - log_ratio_clamped - 1.0
    reverse_kl_est = torch.clamp(reverse_kl_est, min=-10.0, max=10.0)

    # Generalized JSD (paper Eq. 1): beta * RKL + (1-beta) * FKL
    jsd = beta * reverse_kl_est + (1.0 - beta) * forward_kl_est

    return jsd


def gkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, dict]:
    """Compute the full GKD loss with temperature scaling and masking.

    This is the main entry point for computing GKD loss from raw logits.
    Follows the paper convention (Eq. 1):
        D_beta = beta * KL(teacher || student) + (1-beta) * KL(student || teacher)

    Args:
        student_logits: Student model logits. Shape: (batch_size, seq_len, vocab_size).
        teacher_logits: Teacher model logits. Shape: (batch_size, seq_len, vocab_size).
        response_mask: Mask for valid response tokens. Shape: (batch_size, seq_len).
        beta: JSD interpolation parameter (paper convention).
            beta=0 -> forward KL (mean-seeking),
            beta=0.5 -> symmetric JSD,
            beta=1 -> reverse KL (mode-seeking).
        temperature: Temperature for softmax scaling.
        loss_agg_mode: Loss aggregation mode. Options: "token-mean", "seq-mean-token-sum".

    Returns:
        Tuple of (loss, metrics_dict).
    """
    # Temperature scaling
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature

    # Compute log-probabilities
    student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)

    # Compute generalized JSD per token
    jsd_per_token = generalized_jsd(student_log_probs, teacher_log_probs, beta=beta)

    # Apply mask and aggregate
    mask = response_mask.float()
    if loss_agg_mode == "token-mean":
        valid_tokens = mask.sum().clamp(min=1.0)
        loss = (jsd_per_token * mask).sum() / valid_tokens
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(jsd_per_token * mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(jsd_per_token * mask, dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        loss = torch.mean(seq_losses)
    else:
        valid_tokens = mask.sum().clamp(min=1.0)
        loss = (jsd_per_token * mask).sum() / valid_tokens

    # Compute individual KL components for metrics
    with torch.no_grad():
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()

        forward_kl_per_token = torch.sum(
            student_probs * (student_log_probs - teacher_log_probs), dim=-1
        )
        reverse_kl_per_token = torch.sum(
            teacher_probs * (teacher_log_probs - student_log_probs), dim=-1
        )

        valid_tokens = mask.sum().clamp(min=1.0)
        forward_kl_mean = (forward_kl_per_token * mask).sum() / valid_tokens
        reverse_kl_mean = (reverse_kl_per_token * mask).sum() / valid_tokens
        jsd_mean = (jsd_per_token * mask).sum() / valid_tokens

    metrics = {
        "gkd/loss": loss.detach().item(),
        "gkd/forward_kl": forward_kl_mean.item(),
        "gkd/reverse_kl": reverse_kl_mean.item(),
        "gkd/jsd": jsd_mean.item(),
    }

    return loss, metrics


def compute_on_policy_ratio(
    lambda_param: float,
    step: int = 0,
    schedule: str = "constant",
    total_steps: int = 1,
) -> float:
    """Compute the on-policy sampling ratio for the current step.

    GKD uses lambda to control the mixture of on-policy (student-generated)
    and off-policy (dataset/teacher-generated) data:
        - lambda = 0.0: Pure off-policy (traditional KD)
        - lambda = 1.0: Pure on-policy (student generates all sequences)
        - 0 < lambda < 1: Mixed mode (each batch randomly chooses)

    Args:
        lambda_param: Base on-policy ratio.
        step: Current training step.
        schedule: Schedule type. Options: "constant", "linear_warmup".
        total_steps: Total number of training steps (for scheduling).

    Returns:
        Current on-policy ratio (float between 0 and 1).
    """
    if schedule == "constant":
        return lambda_param
    elif schedule == "linear_warmup":
        # Linearly increase from 0 to lambda_param over training
        progress = min(step / max(total_steps, 1), 1.0)
        return lambda_param * progress
    else:
        return lambda_param
