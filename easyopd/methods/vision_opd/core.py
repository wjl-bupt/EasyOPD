"""
Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs
via On-Policy Self-Distillation - Core Algorithm

This module implements the core Vision-OPD algorithm from:
    "Vision-OPD: Learning to See Fine Details for Multimodal LLMs
     via On-Policy Self-Distillation"
    (https://arxiv.org/abs/2605.18740)

The key idea is to use on-policy self-distillation with a teacher model
(maintained via EMA of the student) that receives fine-grained visual inputs
(e.g., bounding-box cropped images) to guide the student model which only
sees the original image.

Key functions:
    - compute_self_distillation_loss: Computes the VOPD distillation loss
      (forward KL, reverse KL, or JSD between student and teacher).
    - ema_update_teacher: Performs EMA update of teacher model parameters.
    - add_tail_bucket: Adds a tail probability bucket for top-k distillation.
    - renorm_topk_log_probs: Renormalizes top-k log probabilities.
"""

import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple


def add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Add a tail probability bucket for top-k distillation.

    Computes log(1 - sum(p_i)) and appends it as an additional dimension.
    This ensures the distribution sums to 1 even when using top-k logits.

    Args:
        log_probs: (batch_size, seq_len, k) top-k log probabilities.

    Returns:
        (batch_size, seq_len, k+1) log probabilities with tail bucket appended.
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)  # Avoid log_s >= 0
    # log(1 - exp(log_s)) = log(-(exp(log_s) - 1))
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
    """Renormalize top-k log probabilities to sum to 1.

    Args:
        logp: (batch_size, seq_len, k) top-k log probabilities.

    Returns:
        (batch_size, seq_len, k) renormalized log probabilities.
    """
    logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
    return logp - logZ


def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.5,
    full_logit_distillation: bool = True,
    distillation_topk: Optional[int] = None,
    distillation_add_tail: bool = True,
    is_clip: Optional[float] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute the Vision-OPD self-distillation loss.

    Supports three distillation modes:
    - Forward KL (alpha=0.0): KL(teacher || student)
    - Reverse KL (alpha=1.0): KL(student || teacher)
    - Generalized JSD (0 < alpha < 1): Interpolated divergence

    Also supports:
    - Full-logit distillation (all vocabulary logits)
    - Top-k logit distillation (memory efficient)
    - IS ratio clipping for stability

    Args:
        student_log_probs: (batch_size, seq_len) student token log-probs.
        teacher_log_probs: (batch_size, seq_len) teacher token log-probs.
        response_mask: (batch_size, seq_len) mask for valid response tokens.
        alpha: KL interpolation coefficient.
            0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        full_logit_distillation: Whether to use full-logit KL distillation.
        distillation_topk: If set, use top-k logits for distillation.
        distillation_add_tail: Whether to add a tail bucket for top-k.
        is_clip: Clip value for distillation IS ratio; None disables IS weighting.
        old_log_probs: (batch_size, seq_len) old policy log-probs for IS ratio.
        student_all_log_probs: (batch_size, seq_len, vocab_size) full student logits.
        teacher_all_log_probs: (batch_size, seq_len, vocab_size) full teacher logits.
        student_topk_log_probs: (batch_size, seq_len, k) top-k student logits.
        teacher_topk_log_probs: (batch_size, seq_len, k) top-k teacher logits.
        self_distillation_mask: (batch_size,) mask indicating which samples have teacher.
        rollout_is_weights: (batch_size, seq_len) rollout correction IS weights.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    metrics = {}

    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    if full_logit_distillation:
        use_topk = distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError(
                    "top-k distillation requires student_topk_log_probs and teacher_topk_log_probs."
                )
            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if distillation_add_tail:
                student_distill_log_probs = add_tail_bucket(student_distill_log_probs)
                teacher_distill_log_probs = add_tail_bucket(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError(
                    "full_logit_distillation requires student_all_log_probs and teacher_all_log_probs."
                )
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        if alpha == 0.0:
            # Forward KL: KL(teacher || student)
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs,
                reduction="none", log_target=True
            )
        elif alpha == 1.0:
            # Reverse KL: KL(student || teacher)
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs,
                reduction="none", log_target=True
            )
        else:
            # Generalized Jensen-Shannon Divergence
            alpha_t = torch.tensor(
                alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([
                    student_distill_log_probs + torch.log(1 - alpha_t),
                    teacher_distill_log_probs + torch.log(alpha_t),
                ]),
                dim=0,
            )
            kl_teacher = F.kl_div(
                mixture_log_probs, teacher_distill_log_probs,
                reduction="none", log_target=True
            )
            kl_student = F.kl_div(
                mixture_log_probs, student_distill_log_probs,
                reduction="none", log_target=True
            )
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)

        raw_per_token_loss = kl_loss.sum(-1)
    else:
        # Non-full-logit: only reverse KL supported
        assert alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs
        raw_per_token_loss = log_ratio.detach() * student_log_probs

    weighted_per_token_loss = raw_per_token_loss

    # Apply IS ratio clipping
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        weighted_per_token_loss = weighted_per_token_loss * ratio

    # Apply rollout correction weights
    if rollout_is_weights is not None:
        weighted_per_token_loss = weighted_per_token_loss * rollout_is_weights

    valid_token_count = loss_mask.sum().clamp(min=1.0)
    metrics["self_distillation/raw_jsd_token_mean"] = (
        (raw_per_token_loss * loss_mask).sum() / valid_token_count
    ).detach().item()
    metrics["self_distillation/weighted_jsd_token_mean"] = (
        (weighted_per_token_loss * loss_mask).sum() / valid_token_count
    ).detach().item()
    if self_distillation_mask is not None:
        metrics["self_distillation/mask_mean"] = self_distillation_mask.float().mean().detach().item()
    metrics["self_distillation/num_distill_tokens"] = loss_mask.sum().detach().item()

    # Compute final loss (token-mean over valid tokens)
    loss = (weighted_per_token_loss * loss_mask).sum() / valid_token_count

    return loss, metrics


def ema_update_teacher(
    teacher_module: torch.nn.Module,
    student_module: torch.nn.Module,
    update_rate: float = 0.05,
) -> None:
    """Perform EMA update of teacher model parameters.

    teacher_param = (1 - update_rate) * teacher_param + update_rate * student_param

    Args:
        teacher_module: The teacher model to update.
        student_module: The student model to copy from.
        update_rate: EMA update rate (0.0 = no update, 1.0 = full copy).
    """
    if update_rate == 0.0:
        return
    with torch.no_grad():
        for teacher_param, student_param in zip(
            teacher_module.parameters(),
            student_module.parameters(),
        ):
            student_data = student_param.data.to(device=teacher_param.device)
            teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)


def progressive_update_teacher(
    teacher_module: torch.nn.Module,
    student_module: torch.nn.Module,
) -> None:
    """Hard-sync teacher to student (used in progressive mode).

    Args:
        teacher_module: The teacher model to update.
        student_module: The student model to copy from.
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(
            teacher_module.parameters(),
            student_module.parameters(),
        ):
            teacher_param.data.copy_(student_param.data.to(device=teacher_param.device))
        for teacher_buffer, student_buffer in zip(
            teacher_module.buffers(),
            student_module.buffers(),
        ):
            teacher_buffer.data.copy_(student_buffer.data.to(device=teacher_buffer.device))
