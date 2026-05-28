"""
OPCD: On-Policy Context Distillation for Language Models - Core Algorithm

This module implements the core OPCD algorithm from:
    "On-Policy Context Distillation for Language Models"
    (https://arxiv.org/abs/2602.12275)

The key idea is to distill contextual knowledge (e.g., experiential knowledge,
system prompts) into a language model by using on-policy KL divergence minimization.
The teacher model receives the context (experience) in its prompt while the student
generates responses without the context, then the student is trained to match the
teacher's output distribution via KL loss.

Key functions:
    - kl_penalty: Computes various forms of KL divergence between student and teacher.
    - compute_opcd_loss: Computes the OPCD context distillation loss.
    - build_experience_prompt: Injects experience into prompt messages.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


# Template for injecting experience into prompts
EXPERIENCE_SOLVE_PROMPT_TEMPLATE = (
    "You have previously learned the following experience that may help you solve this problem:\n\n"
    "<experience>\n{experience}\n</experience>\n\n"
    "Now, please solve the following problem. "
    "You can use the experience above if it is helpful, but you should also think independently.\n\n"
    "{prompt}"
)


def kl_penalty(
    logprob: torch.FloatTensor,
    ref_logprob: torch.FloatTensor,
    kl_penalty_type: str,
    kl_renorm_topk: bool = False,
) -> torch.FloatTensor:
    """Compute KL divergence penalty between student and teacher log-probabilities.

    Supports multiple KL divergence formulations:
    - "seqkd": Sequence-level KD loss (-ref_logprob, i.e., cross-entropy with teacher)
    - "kl" / "k1": Standard forward KL (logprob - ref_logprob)
    - "abs": Absolute difference |logprob - ref_logprob|
    - "mse" / "k2": Mean squared error 0.5 * (logprob - ref_logprob)^2
    - "low_var_kl" / "k3": Low-variance KL estimator (ratio - log_ratio - 1)
    - "full": Full KL divergence sum_i p_i * (log p_i - log q_i) over vocabulary

    Args:
        logprob: Student log-probabilities.
            For non-"full" modes: shape (batch_size, seq_len) — per-token log-probs.
            For "full" mode: shape (batch_size, seq_len, vocab_size) or (batch_size, seq_len, topk).
        ref_logprob: Teacher/reference log-probabilities (same shape as logprob).
        kl_penalty_type: Type of KL penalty to compute.
        kl_renorm_topk: Whether to renormalize top-k log-probs before computing full KL.

    Returns:
        KL penalty tensor. For non-"full" modes: (batch_size, seq_len).
        For "full" mode: (batch_size, seq_len) — summed over vocabulary dimension.
    """
    if kl_penalty_type == "seqkd":
        return -ref_logprob

    if kl_penalty_type in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty_type in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty_type in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty_type == "full":
        if kl_renorm_topk:
            # Padded positions are set to -1e20
            # Use a mask to ensure they don't affect logsumexp
            mask = logprob > -1e15  # use -1e15 as threshold to be safe

            logprob = logprob - torch.logsumexp(
                torch.where(mask, logprob, torch.full_like(logprob, -1e20)),
                dim=-1, keepdim=True,
            )
            ref_logprob = ref_logprob - torch.logsumexp(
                torch.where(mask, ref_logprob, torch.full_like(ref_logprob, -1e20)),
                dim=-1, keepdim=True,
            )

        # Compute KL: sum over vocabulary dimension
        # KL(p || q) = sum_i p_i * (log p_i - log q_i)
        kl_density = logprob.exp() * (logprob - ref_logprob)

        # Mask out padded positions (logprob = -1e20 -> exp = 0)
        kl_density = torch.where(logprob > -1e15, kl_density, torch.zeros_like(kl_density))
        kl = kl_density.sum(dim=-1)

        return kl

    raise NotImplementedError(f"Unknown kl_penalty type: {kl_penalty_type}")


def compute_opcd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_loss_type: str = "full",
    kl_renorm_topk: bool = False,
    on_policy_merge: bool = True,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, dict]:
    """Compute the OPCD context distillation loss.

    In on-policy mode (on_policy_merge=True):
        - Student generates responses without context
        - Teacher provides logits with context in prompt
        - Loss = KL(student || teacher) — student learns to match teacher

    In off-policy mode (on_policy_merge=False):
        - Teacher generates responses with context
        - Loss = KL(teacher || student) — student learns from teacher's generations

    Args:
        student_log_probs: Student model log-probabilities.
        teacher_log_probs: Teacher model (with context) log-probabilities.
        response_mask: Mask for valid response tokens.
        kl_loss_type: Type of KL divergence. Options: "full", "kl", "abs", "mse", "low_var_kl", "seqkd".
        kl_renorm_topk: Whether to renormalize top-k log-probs.
        on_policy_merge: If True, use on-policy KL direction (student || teacher).
        loss_agg_mode: Loss aggregation mode. Options: "token-mean", "seq-mean-token-sum".

    Returns:
        Tuple of (loss, metrics_dict).
    """
    metrics = {}

    if on_policy_merge:
        # On-policy: KL(student || teacher) — student learns to match teacher
        kld = kl_penalty(
            logprob=student_log_probs,
            ref_logprob=teacher_log_probs,
            kl_penalty_type=kl_loss_type,
            kl_renorm_topk=kl_renorm_topk,
        )
    else:
        # Off-policy: KL(teacher || student) — student learns from teacher's generations
        kld = kl_penalty(
            logprob=teacher_log_probs,
            ref_logprob=student_log_probs,
            kl_penalty_type=kl_loss_type,
            kl_renorm_topk=kl_renorm_topk,
        )

    # Aggregate loss
    if loss_agg_mode == "token-mean":
        valid_tokens = response_mask.sum().clamp(min=1.0)
        loss = (kld * response_mask).sum() / valid_tokens
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(kld * response_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(kld * response_mask, dim=-1) / torch.sum(response_mask, dim=-1).clamp(min=1.0)
        loss = torch.mean(seq_losses)
    else:
        valid_tokens = response_mask.sum().clamp(min=1.0)
        loss = (kld * response_mask).sum() / valid_tokens

    metrics["opcd/kl_loss"] = loss.detach().item()
    metrics["opcd/kl_mean_per_token"] = (
        (kld * response_mask).sum() / response_mask.sum().clamp(min=1.0)
    ).detach().item()

    return loss, metrics


def build_experience_prompt(
    raw_prompt_messages: list,
    experience: str,
    mode: str = "user_content",
    train_system_prompt: bool = False,
) -> list:
    """Inject experience knowledge into prompt messages for the teacher model.

    Supports two modes:
    1. "user_content": Append experience to the last user message content.
    2. "system_prompt": Insert experience as a system message at the beginning.

    Args:
        raw_prompt_messages: Original prompt messages (list of dicts).
        experience: Experience text to inject.
        mode: Injection mode. Options: "user_content", "system_prompt".
        train_system_prompt: If True, use system prompt mode regardless of `mode`.

    Returns:
        Modified messages with experience injected.
    """
    from copy import deepcopy
    import numpy as np

    msgs = deepcopy(raw_prompt_messages)
    if isinstance(msgs, np.ndarray):
        msgs = msgs.tolist()

    if not experience or not experience.strip():
        return msgs

    if train_system_prompt or mode == "system_prompt":
        # Insert experience as system message
        msgs.insert(0, {"role": "system", "content": experience})
    else:
        # Append experience to last user message
        content = msgs[-1]["content"]
        updated_content = EXPERIENCE_SOLVE_PROMPT_TEMPLATE.format(
            experience=experience, prompt=content
        )
        msgs[-1]["content"] = updated_content

    return msgs


def truncate_experience(
    experience: str,
    max_tokens: int,
    tokenizer=None,
) -> str:
    """Truncate experience text to fit within max_tokens.

    If tokenizer is provided, truncates by token count.
    Otherwise, truncates by character count (approximate).

    Args:
        experience: Experience text to truncate.
        max_tokens: Maximum number of tokens allowed.
        tokenizer: Optional tokenizer for accurate token counting.

    Returns:
        Truncated experience string.
    """
    if not experience:
        return experience

    if tokenizer is not None:
        tokens = tokenizer.encode(experience, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            experience = tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        # Approximate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        if len(experience) > max_chars:
            experience = experience[:max_chars]

    return experience
