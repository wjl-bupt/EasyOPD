"""
SOD: Step-wise On-policy Distillation - Core Algorithm

This module implements the core step-wise OPD weighting algorithm from:
    "SOD: Step-wise On-policy Distillation for Small Language Model Agents"
    (https://arxiv.org/abs/2605.07725)

The key idea is to adaptively re-weight the OPD signal at each reasoning step
based on the student-teacher divergence trajectory, preventing harmful supervision
when cascade failures occur in tool-integrated reasoning (TIR).

Key functions:
    - _extract_step_boundaries: Identifies assistant turns from response_mask.
    - compute_stepwise_opd_weights: Computes per-token w_k weights (Eq. 6, 7).
    - apply_stepwise_opd: Applies weighted OPD to advantages (Eq. 10).
"""

import torch


def _extract_step_boundaries(response_mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    """Extract per-sample step (assistant turn) boundaries from response_mask.

    In multi-turn agent loops, response_mask is 1 for assistant tokens and 0 for
    tool-response / padding tokens. Each contiguous run of 1s is one "step".

    Args:
        response_mask: (batch_size, seq_len), binary mask.

    Returns:
        List (length = batch_size) of lists of (start, end) index pairs.
        Each (start, end) pair satisfies: response_mask[i, start:end] are all 1.
    """
    batch_boundaries = []
    bsz, seq_len = response_mask.shape
    for i in range(bsz):
        mask_i = response_mask[i]  # (seq_len,)
        boundaries = []
        in_segment = False
        seg_start = 0
        for t in range(seq_len):
            if mask_i[t].item() == 1 and not in_segment:
                seg_start = t
                in_segment = True
            elif mask_i[t].item() == 0 and in_segment:
                boundaries.append((seg_start, t))
                in_segment = False
        if in_segment:
            boundaries.append((seg_start, seq_len))
        batch_boundaries.append(boundaries)
    return batch_boundaries


def compute_stepwise_opd_weights(
    old_log_probs: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    epsilon: float = 1e-6,
    delta: float = 0.5,
) -> tuple[torch.Tensor, list[dict]]:
    """Compute per-token step-wise OPD weights w_k for each sample.

    Implements Equations 6 and 7 from the SOD paper:
        d_k = (1/|I_k|) * sum_{t in I_k} |log pi_theta(y_t) - log pi_teacher(y_t)|  (Eq. 6)
        w_k = min(prod_{u=1}^{k-1} (d_u + eps)/(d_{u+1} + eps), 1 + delta)          (Eq. 7)

    For each sample, we:
      1. Identify step boundaries from response_mask (contiguous 1-runs).
      2. Compute d_k = mean(|log pi_theta - log pi_teacher|) for each step k.
      3. Compute w_1 = 1, w_k = min(prod_{u=1}^{k-1} (d_u+eps)/(d_{u+1}+eps), 1+delta).
      4. Broadcast w_k to all tokens in step k.

    Args:
        old_log_probs: (batch_size, seq_len) student log-probs on sampled tokens.
        ref_log_prob:  (batch_size, seq_len) teacher log-probs on sampled tokens.
        response_mask: (batch_size, seq_len) binary mask (1=assistant, 0=tool/pad).
        epsilon: small constant for numerical stability.
        delta: upper-bound offset, w_k <= 1 + delta.

    Returns:
        weight_mask: (batch_size, seq_len) per-token OPD weights.
        log_info: list of dicts, one per sample, containing d_values, w_values, etc.
    """
    bsz, seq_len = response_mask.shape
    # |log pi_theta - log pi_teacher| per token
    abs_logprob_diff = (old_log_probs - ref_log_prob).abs()  # (bsz, seq_len)

    batch_boundaries = _extract_step_boundaries(response_mask)
    weight_mask = torch.zeros_like(response_mask, dtype=torch.float32)
    log_info = []

    for i in range(bsz):
        boundaries = batch_boundaries[i]
        if len(boundaries) == 0:
            log_info.append({"n_steps": 0, "d_values": [], "w_values": [], "n_tokens_per_step": [], "boundaries": []})
            continue

        # Compute d_k for each step
        d_values = []
        n_tokens_per_step = []
        for start, end in boundaries:
            step_diff = abs_logprob_diff[i, start:end]
            step_mask = response_mask[i, start:end]
            n_tokens = step_mask.sum().item()
            if n_tokens > 0:
                d_k = (step_diff * step_mask).sum().item() / n_tokens
            else:
                d_k = 0.0
            d_values.append(d_k)
            n_tokens_per_step.append(int(n_tokens))

        # Compute w_k via cumulative product of adjacent ratios
        K = len(d_values)
        w_values = [1.0]  # w_1 = 1
        if K > 1:
            cum_prod = 1.0
            for u in range(K - 1):
                ratio = (d_values[u] + epsilon) / (d_values[u + 1] + epsilon)
                cum_prod *= ratio
                w_k = min(cum_prod, 1.0 + delta)
                w_values.append(w_k)

        # Broadcast w_k to all tokens in step k
        for k, (start, end) in enumerate(boundaries):
            weight_mask[i, start:end] = w_values[k]

        log_info.append({
            "n_steps": K,
            "d_values": d_values,
            "w_values": w_values,
            "n_tokens_per_step": n_tokens_per_step,
            "boundaries": [(s, e) for s, e in boundaries],
        })

    return weight_mask, log_info


def apply_stepwise_opd(
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    epsilon: float = 1e-6,
    delta: float = 0.5,
    opd_coef: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Apply step-wise weighted OPD to the GRPO advantages (Eq. 10).

    Computes:
        A_total = A_GRPO + opd_coef * w_k * (log pi_teacher - log pi_theta)

    Args:
        advantages: (batch_size, seq_len) or (batch_size,) GRPO advantages.
        old_log_probs: (batch_size, seq_len) student log-probs.
        ref_log_prob: (batch_size, seq_len) teacher log-probs.
        response_mask: (batch_size, seq_len) binary mask.
        epsilon: numerical stability constant.
        delta: upper-bound offset for w_k.
        opd_coef: global coefficient for the OPD term.

    Returns:
        A_total: (batch_size, seq_len) modified advantages.
        stepwise_weights: (batch_size, seq_len) the computed w_k weights.
        log_info: per-sample logging information.
    """
    # OPD signal: ref_log_prob - old_log_probs (reverse KL direction for distillation)
    raw_local_adv = (ref_log_prob - old_log_probs) * response_mask

    # Compute per-token step-wise weights w_k
    stepwise_weights, log_info = compute_stepwise_opd_weights(
        old_log_probs=old_log_probs,
        ref_log_prob=ref_log_prob,
        response_mask=response_mask,
        epsilon=epsilon,
        delta=delta,
    )
    stepwise_weights = stepwise_weights.to(raw_local_adv.device)

    # Weighted OPD signal: w_k * (ref_log_prob - old_log_probs) per token
    weighted_opd = opd_coef * stepwise_weights * raw_local_adv

    # Total advantage: A_GRPO (broadcast to token dim if needed) + weighted OPD
    if advantages.dim() == 1:
        A_total = advantages.unsqueeze(1) + weighted_opd
    else:
        A_total = advantages + weighted_opd

    return A_total, stepwise_weights, log_info
