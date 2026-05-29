"""
OPSA: On-Policy Self-Distillation for Safety Alignment

Core algorithm implementation from:
    "Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation"
    Fu et al., arXiv 2026
    https://arxiv.org/abs/2605.15239

The key idea is to use the same base model as both teacher and student in an
on-policy self-distillation loop, with type-conditional privileged contexts that
activate latent safety reasoning. Dense per-token KL supervision is applied
within the "refusal-decision window" — the narrow early-response region where
safety behavior is determined.

Key functions:
    - compute_teacher_flip_rate: Training-free TFR metric for context quality.
    - compute_early_window_weights: Per-token weights for the refusal-decision window.
    - compute_opsa_kl_loss: Per-token forward KL D_KL(p_T || p_S) with masking.
    - opsa_loss: Full OPSA loss with temperature scaling, window weighting, and aggregation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_teacher_flip_rate(
    teacher_safe_flags_with_context: torch.Tensor,
    teacher_safe_flags_without_context: torch.Tensor,
) -> torch.Tensor:
    """Compute the Teacher Flip Rate (TFR) for privileged context evaluation.

    TFR measures how often a privileged context converts an unsafe response
    (from the frozen base model without context) into a safe response
    (from the frozen base model with context). This is a training-free signal
    to select effective privileged contexts.

    Paper reference: [Fu et al., arXiv 2605.15239, Section 3.3]

        TFR = |{x : unsafe(T(x)) AND safe(T(x | I))}| / |{x : unsafe(T(x))}|

    where T(x) is the teacher response without context and T(x|I) is with context I.

    Args:
        teacher_safe_flags_with_context: Binary flags indicating whether teacher
            responses WITH privileged context are safe (1) or unsafe (0).
            Shape: (num_samples,).
        teacher_safe_flags_without_context: Binary flags indicating whether teacher
            responses WITHOUT privileged context are safe (1) or unsafe (0).
            Shape: (num_samples,).

    Returns:
        Scalar tensor with TFR value in [0, 1]. Returns 0 if no unsafe base responses.
    """
    # Identify samples where base model (without context) is unsafe
    base_unsafe_mask = (teacher_safe_flags_without_context == 0)
    num_base_unsafe = base_unsafe_mask.sum().float()

    if num_base_unsafe == 0:
        return torch.tensor(0.0, device=teacher_safe_flags_with_context.device)

    # Count how many of those become safe with privileged context
    flipped_to_safe = (base_unsafe_mask & (teacher_safe_flags_with_context == 1)).sum().float()

    tfr = flipped_to_safe / num_base_unsafe
    return tfr


def compute_early_window_weights(
    response_mask: torch.Tensor,
    window_size: int = 32,
    decay_type: str = "linear",
    min_weight: float = 0.1,
) -> torch.Tensor:
    """Compute per-token weights emphasizing the early refusal-decision window.

    OPSA concentrates gradient updates within the safety-critical token window
    at the beginning of the response, where refusal/compliance decisions are made.
    Tokens beyond this window receive reduced (but non-zero) weight.

    Paper reference: [Fu et al., arXiv 2605.15239, Section 3.2]

    Args:
        response_mask: Binary mask for valid response tokens.
            Shape: (batch_size, seq_len).
        window_size: Number of tokens at the start of each response that form
            the "refusal-decision window". Default: 32 tokens.
        decay_type: How weights decay beyond the window.
            Options: "linear" (gradual decay), "step" (hard cutoff to min_weight),
                     "exponential" (exponential decay).
        min_weight: Minimum weight for tokens outside the decision window.
            Ensures some gradient signal even for later tokens.

    Returns:
        Per-token weight tensor. Shape: (batch_size, seq_len).
        Values in [min_weight, 1.0], with 1.0 for tokens in the decision window.
    """
    batch_size, seq_len = response_mask.shape
    device = response_mask.device

    # Find the start of each response (first 1 in response_mask per row)
    # Use cumsum to identify position within the response
    response_positions = torch.cumsum(response_mask, dim=-1) * response_mask  # (B, L)

    # Tokens within window_size of response start get weight 1.0
    in_window = (response_positions > 0) & (response_positions <= window_size)

    # Per-sample actual response length (number of valid response tokens).
    # Using this for normalization avoids contamination from prompt/padding
    # length (seq_len includes prompt + padding, which is irrelevant to the
    # response-side decay). Shape: (B,).
    resp_len = response_positions.max(dim=-1).values  # (B,)
    # Per-sample max position beyond the early window; clamp to 1 to avoid
    # division by zero when resp_len <= window_size (in that case all tokens
    # are inside the window and decay_factor is unused for them).
    # Shape: (B, 1) for broadcasting against (B, L).
    max_beyond = (resp_len - window_size).clamp(min=1).float().unsqueeze(-1)  # (B, 1)

    if decay_type == "step":
        # Hard cutoff: full weight in window, min_weight outside
        weights = torch.where(in_window, torch.ones_like(response_mask, dtype=torch.float32),
                              torch.full_like(response_mask, min_weight, dtype=torch.float32))
    elif decay_type == "linear":
        # Linear decay from 1.0 at window boundary to min_weight at each
        # sample's own response end. Normalized per-sample so that different
        # prompt/response lengths within the same batch get independent decay
        # rates rather than sharing a global seq_len-based slope.
        beyond_window_pos = (response_positions - window_size).clamp(min=0).float()
        decay_factor = 1.0 - (1.0 - min_weight) * (beyond_window_pos / max_beyond)
        weights = torch.where(in_window, torch.ones_like(decay_factor), decay_factor)
    elif decay_type == "exponential":
        # Exponential decay: weight = exp(-alpha * (pos - window_size))
        # Calibrated per-sample so the weight at each sample's own response
        # end equals min_weight, independent of padded seq_len.
        beyond_window_pos = (response_positions - window_size).clamp(min=0).float()
        alpha = -torch.log(torch.tensor(min_weight, device=device)) / max_beyond  # (B, 1)
        decay_factor = torch.exp(-alpha * beyond_window_pos)
        weights = torch.where(in_window, torch.ones_like(decay_factor), decay_factor)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}. Options: 'linear', 'step', 'exponential'.")

    # Apply response_mask: non-response tokens get weight 0
    weights = weights * response_mask.float()

    return weights


def compute_opsa_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token forward KL divergence D_KL(p_T || p_S).

    This is the core supervision signal in OPSA: the teacher (frozen base with
    privileged context) provides dense per-token supervision to the student
    (on-policy model without context).

    Paper reference: [Fu et al., arXiv 2605.15239, Eq. 3]

        L_OPSA = sum_t D_KL(p_T(·|x, I, y_{<t}) || p_S(·|x, y_{<t}))

    where p_T is teacher distribution (with privileged context I) and
    p_S is student distribution (without context).

    Note: Forward KL (teacher || student) is used because it encourages the
    student to cover all modes of the teacher's distribution, which is important
    for safety (ensures student learns to refuse when teacher refuses).

    Args:
        student_log_probs: Student model log-probabilities over vocabulary.
            Shape: (batch_size, seq_len, vocab_size).
        teacher_log_probs: Teacher model log-probabilities over vocabulary
            (computed with privileged context).
            Shape: (batch_size, seq_len, vocab_size).

    Returns:
        Per-token KL divergence values.
            Shape: (batch_size, seq_len).
    """
    # Forward KL: D_KL(p_T || p_S) = sum_v p_T(v) * (log p_T(v) - log p_S(v))
    teacher_probs = teacher_log_probs.exp()
    kl_per_token = torch.sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        dim=-1,
    )
    # Clamp to avoid negative values from numerical issues
    kl_per_token = kl_per_token.clamp(min=0.0)
    return kl_per_token


def opsa_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
    window_size: int = 32,
    decay_type: str = "linear",
    min_weight: float = 0.1,
    use_window_weighting: bool = True,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, dict]:
    """Compute the full OPSA loss with temperature scaling, window weighting, and masking.

    This is the main entry point for computing the OPSA distillation loss.
    It combines:
      1. Temperature-scaled softmax for both teacher and student
      2. Per-token forward KL divergence D_KL(p_T || p_S)
      3. Early-window weighting to concentrate updates on safety-critical tokens
      4. Masked aggregation over valid response tokens

    Paper reference: [Fu et al., arXiv 2605.15239, Section 3]

    Args:
        student_logits: Student model logits (on-policy, without privileged context).
            Shape: (batch_size, seq_len, vocab_size).
        teacher_logits: Teacher model logits (frozen copy, with privileged context).
            Shape: (batch_size, seq_len, vocab_size).
        response_mask: Binary mask for valid response tokens.
            Shape: (batch_size, seq_len).
        temperature: Temperature for softmax scaling. Higher values produce
            softer distributions, transferring more "dark knowledge".
        window_size: Number of tokens in the refusal-decision window.
        decay_type: Weight decay type beyond the window.
            Options: "linear", "step", "exponential".
        min_weight: Minimum weight for tokens outside the decision window.
        use_window_weighting: Whether to apply early-window weighting.
            If False, all response tokens receive equal weight.
        loss_agg_mode: Loss aggregation mode.
            Options: "token-mean", "seq-mean-token-sum", "seq-mean-token-mean".

    Returns:
        Tuple of (loss, metrics_dict).
            loss: Scalar loss tensor.
            metrics_dict: Dictionary with monitoring metrics.
    """
    # Temperature scaling
    student_logits_scaled = student_logits / temperature
    teacher_logits_scaled = teacher_logits / temperature

    # Compute log-probabilities
    student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits_scaled, dim=-1)

    # Compute per-token forward KL
    kl_per_token = compute_opsa_kl_loss(student_log_probs, teacher_log_probs)

    # Apply early-window weighting
    mask = response_mask.float()
    if use_window_weighting:
        window_weights = compute_early_window_weights(
            response_mask, window_size=window_size,
            decay_type=decay_type, min_weight=min_weight,
        )
        weighted_kl = kl_per_token * window_weights
    else:
        weighted_kl = kl_per_token * mask

    # Aggregate loss
    if loss_agg_mode == "token-mean":
        valid_tokens = mask.sum().clamp(min=1.0)
        loss = (weighted_kl * mask).sum() / valid_tokens
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(weighted_kl * mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
        seq_losses = torch.sum(weighted_kl * mask, dim=-1) / seq_lengths
        loss = torch.mean(seq_losses)
    else:
        valid_tokens = mask.sum().clamp(min=1.0)
        loss = (weighted_kl * mask).sum() / valid_tokens

    # Compute metrics (no_grad for efficiency)
    with torch.no_grad():
        valid_tokens = mask.sum().clamp(min=1.0)
        kl_mean = (kl_per_token * mask).sum() / valid_tokens

        # Window vs non-window KL breakdown
        if use_window_weighting:
            window_mask = (window_weights >= 1.0 - 1e-6) & (mask > 0)
            non_window_mask = (~window_mask) & (mask > 0)
            window_tokens = window_mask.sum().clamp(min=1.0).float()
            non_window_tokens = non_window_mask.sum().clamp(min=1.0).float()
            kl_in_window = (kl_per_token * window_mask.float()).sum() / window_tokens
            kl_outside_window = (kl_per_token * non_window_mask.float()).sum() / non_window_tokens
        else:
            kl_in_window = kl_mean
            kl_outside_window = torch.tensor(0.0, device=loss.device)

    metrics = {
        "opsa/loss": loss.detach().item(),
        "opsa/kl_mean": kl_mean.item(),
        "opsa/kl_in_window": kl_in_window.item(),
        "opsa/kl_outside_window": kl_outside_window.item(),
        "opsa/temperature": temperature,
        "opsa/window_size": window_size,
    }

    return loss, metrics
