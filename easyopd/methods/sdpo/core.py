"""
SDPO: Self-Distilled Policy Optimization

Core algorithm implementation from:
    "Reinforcement Learning via Self-Distillation"
    Hübotter et al., 2026
    https://arxiv.org/abs/2601.20802

The key idea is to augment on-policy optimization (e.g., GRPO) with
self-distillation from the model's own high-reward trajectories. The model
conditioned on feedback acts as a self-teacher, and its next-token predictions
are distilled back into the policy using a generalized JSD loss.

Key components:
    - compute_sdpo_self_distillation_loss: Main distillation loss (JSD/KL with IS correction)
    - build_reprompt_text: Constructs teacher prompts from demonstrations and feedback
    - select_demonstration: Selects successful demonstrations for reprompting
    - compute_ema_update: EMA teacher weight update
"""

import re
import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple


def compute_sdpo_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.5,
    full_logit_distillation: bool = True,
    distillation_topk: Optional[int] = None,
    distillation_add_tail: bool = True,
    is_clip: Optional[float] = 2.0,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute the SDPO self-distillation loss.

    This implements the core SDPO loss: a generalized JSD between the student
    (current policy) and the self-teacher (model conditioned on successful
    demonstrations/feedback).

    The loss supports:
        - Full-logit distillation (KL over entire vocabulary)
        - Top-k distillation (KL over top-k tokens + tail bucket)
        - Importance sampling correction (IS clipping)
        - Per-sample masking (only distill samples with demonstrations)

    Args:
        student_log_probs: Per-token student log-probabilities.
            Shape: (batch_size, response_length).
        teacher_log_probs: Per-token teacher log-probabilities.
            Shape: (batch_size, response_length).
        response_mask: Mask for valid response tokens.
            Shape: (batch_size, response_length).
        alpha: KL interpolation coefficient.
            0.0 = forward KL, 1.0 = reverse KL, 0.5 = symmetric JSD.
        full_logit_distillation: Whether to use full-logit KL distillation.
        distillation_topk: If set, use top-k logits for distillation.
        distillation_add_tail: Whether to add a tail bucket for top-k.
        is_clip: Clip value for IS ratio. None disables IS weighting.
        old_log_probs: Log-probs from the old policy (for IS correction).
            Shape: (batch_size, response_length).
        student_all_log_probs: Full vocabulary student log-probs.
            Shape: (batch_size, response_length, vocab_size).
        teacher_all_log_probs: Full vocabulary teacher log-probs.
            Shape: (batch_size, response_length, vocab_size).
        student_topk_log_probs: Top-k student log-probs.
            Shape: (batch_size, response_length, k).
        teacher_topk_log_probs: Top-k teacher log-probs.
            Shape: (batch_size, response_length, k).
        self_distillation_mask: Per-sample mask (1 = has demonstration).
            Shape: (batch_size,).
        loss_agg_mode: Loss aggregation mode.
        rollout_is_weights: Pre-computed rollout IS weights.
            Shape: (batch_size, response_length).

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
                student_distill_log_probs = _add_tail(student_distill_log_probs)
                teacher_distill_log_probs = _add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = _renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = _renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError(
                    "full_logit_distillation requires student_all_log_probs and teacher_all_log_probs."
                )
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        # Compute generalized JSD
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
            # Generalized JSD
            alpha_tensor = torch.tensor(
                alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            # Mixture distribution: M = alpha * teacher + (1-alpha) * student
            mixture_log_probs = torch.logsumexp(
                torch.stack([
                    student_distill_log_probs + torch.log(1 - alpha_tensor),
                    teacher_distill_log_probs + torch.log(alpha_tensor),
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
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha_tensor)

        per_token_loss = kl_loss.sum(-1)
    else:
        # Non-full-logit: reverse KL only
        assert alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    # Apply IS correction
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    # Aggregate loss
    loss = _agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
    )

    return loss, metrics


def build_reprompt_text(
    prompt_text: str,
    solution: Optional[str],
    feedback: Optional[str],
    reprompt_template: str,
    solution_template: str,
    feedback_template: str,
    feedback_only_without_solution: bool = True,
) -> str:
    """Build the reprompted text for the self-teacher.

    SDPO constructs a teacher prompt by combining:
    - The original question
    - A successful demonstration (if available)
    - Environment feedback from failed attempts (if available)

    Args:
        prompt_text: Original user prompt text.
        solution: A successful solution string (or None).
        feedback: Environment feedback string (or None).
        reprompt_template: Template with {prompt}, {solution}, {feedback} placeholders.
        solution_template: Template with {successful_previous_attempt} placeholder.
        feedback_template: Template with {feedback_raw} placeholder.
        feedback_only_without_solution: If True, only use feedback when no solution exists.

    Returns:
        The reprompted text string.
    """
    has_solution = solution is not None
    has_feedback = feedback is not None

    # Determine whether to use feedback
    use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

    # Build solution section
    solution_section = ""
    if has_solution:
        solution_section = solution_template.format(
            successful_previous_attempt=solution
        )

    # Build feedback section
    feedback_section = ""
    if use_feedback:
        feedback_section = feedback_template.format(
            feedback_raw=feedback
        )

    # Combine sections
    if use_feedback or has_solution:
        reprompt_text = reprompt_template.format(
            prompt=prompt_text,
            solution=solution_section,
            feedback=feedback_section,
        )
    else:
        reprompt_text = prompt_text

    return reprompt_text


def select_demonstration(
    idx: int,
    success_by_uid: dict,
    uids: list,
    response_texts: list,
    dont_reprompt_on_self_success: bool = False,
    remove_thinking_from_demonstration: bool = False,
) -> Optional[str]:
    """Select a successful demonstration for reprompting.

    SDPO uses the model's own successful responses as demonstrations.
    For a given sample, it finds other successful responses from the same
    prompt group and uses one as a demonstration.

    Args:
        idx: Index of the current sample.
        success_by_uid: Dict mapping uid -> list of successful sample indices.
        uids: List of uids for each sample.
        response_texts: List of decoded response texts.
        dont_reprompt_on_self_success: If True, exclude the sample's own response.
        remove_thinking_from_demonstration: If True, remove <think>...</think> tags.

    Returns:
        A successful demonstration string, or None if no demonstration available.
    """
    uid = uids[idx]
    solution_idxs = success_by_uid.get(uid, [])

    if dont_reprompt_on_self_success:
        solution_idxs = [j for j in solution_idxs if j != idx]

    if len(solution_idxs) == 0:
        return None

    # Take the first successful demonstration (effectively random)
    solution_idx = solution_idxs[0]
    solution_str = response_texts[solution_idx]

    if remove_thinking_from_demonstration:
        solution_str = _remove_thinking_trace(solution_str)

    return solution_str


def compute_ema_update(
    student_params: dict,
    teacher_params: dict,
    update_rate: float = 0.05,
) -> dict:
    """Compute EMA update for teacher model weights.

    SDPO uses an EMA of the student model as the self-teacher:
        theta_teacher = (1 - rate) * theta_teacher + rate * theta_student

    Args:
        student_params: Dictionary of student model parameters.
        teacher_params: Dictionary of teacher model parameters.
        update_rate: EMA update rate (0 = no update, 1 = copy student).

    Returns:
        Updated teacher parameters dictionary.
    """
    updated = {}
    for key in teacher_params:
        if key in student_params:
            updated[key] = (
                (1.0 - update_rate) * teacher_params[key]
                + update_rate * student_params[key]
            )
        else:
            updated[key] = teacher_params[key]
    return updated


# ============ Internal helper functions ============


def _add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Add a tail bucket to top-k log-probabilities.

    Computes log(1 - sum(p_i)) for the remaining probability mass
    and appends it as an additional "tail" token.

    Args:
        log_probs: Top-k log-probabilities. Shape: (..., k).

    Returns:
        Log-probabilities with tail bucket. Shape: (..., k+1).
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    # log(1 - exp(log_s)) = log(-(exp(log_s) - 1)) = log(-expm1(log_s))
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def _renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
    """Renormalize top-k log-probabilities to sum to 1.

    Args:
        logp: Top-k log-probabilities. Shape: (..., k).

    Returns:
        Renormalized log-probabilities. Shape: (..., k).
    """
    logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
    return logp - logZ


def _remove_thinking_trace(text: str) -> str:
    """Remove <think>...</think> tags and their content from text."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)


def _agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """Aggregate per-token loss into a scalar.

    Args:
        loss_mat: Per-token loss values. Shape: (batch_size, seq_len).
        loss_mask: Mask for valid tokens. Shape: (batch_size, seq_len).
        loss_agg_mode: Aggregation mode.

    Returns:
        Scalar loss tensor.
    """
    if loss_agg_mode == "token-mean":
        valid_tokens = loss_mask.sum().clamp(min=1.0)
        loss = (loss_mat * loss_mask).sum() / valid_tokens
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        valid_seqs = seq_mask.sum().clamp(min=1.0)
        loss = (seq_losses * seq_mask).sum() / valid_seqs
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)
        seq_valid = (seq_mask > 0).float()
        valid_seqs = seq_valid.sum().clamp(min=1.0)
        loss = (seq_losses * seq_valid).sum() / valid_seqs
    else:
        valid_tokens = loss_mask.sum().clamp(min=1.0)
        loss = (loss_mat * loss_mask).sum() / valid_tokens

    return loss
