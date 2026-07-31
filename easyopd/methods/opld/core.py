"""Core math for OPLD (on-policy listwise distillation).

The student samples K rollouts per prompt. Teacher and student each turn their
K per-sequence logprobs into a group-wise categorical distribution via softmax,
and we distill the teacher distribution into the student one with a KL.

The exact gradient of that group KL is a per-sequence policy gradient, so the
whole method reduces to an *advantage*: no bespoke loss is needed, the standard
PPO/PG path in the actor consumes it. See ``advantage_estimator.py`` for how the
advantage is wired into verl, and ``README.md`` for the derivation.

Everything in this module is pure tensor math with no verl imports so it stays
unit-testable on CPU.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

# Default hyper-parameters. Overridable via ``algorithm.easyopd.listwise`` in the
# training yaml (see ``read_listwise_config``).
_DEFAULTS = {
    "beta": 1.0,
    "eta": 0.0,
    "length_norm": True,
    "kl_direction": "qT_to_qS",
    "std_norm": False,
}

_EPS = 1e-6


def read_listwise_config(config: Any = None) -> dict:
    """Pull the OPLD hyper-parameters out of an ``AlgoConfig``.

    Reads ``algorithm.easyopd.listwise`` (falling back to the legacy key
    ``algorithm.easyopd.listwise_kl``). ``AlgoConfig.easyopd`` is a real declared
    field, so unlike a made-up ``algorithm.listwise_kl`` this actually round-trips
    from yaml instead of being silently swallowed by ``BaseConfig.get``.

    Unknown keys raise, so a typo in the yaml fails loudly instead of silently
    training with defaults.
    """
    resolved = dict(_DEFAULTS)
    if config is None:
        return resolved

    easyopd_cfg = config.get("easyopd", None) or {}
    if not isinstance(easyopd_cfg, dict):
        # OmegaConf DictConfig supports mapping access; anything else we ignore.
        try:
            easyopd_cfg = dict(easyopd_cfg)
        except (TypeError, ValueError):
            return resolved

    user_cfg = easyopd_cfg.get("listwise", None)
    if user_cfg is None:
        user_cfg = easyopd_cfg.get("listwise_kl", None)
    if user_cfg is None:
        return resolved

    user_cfg = dict(user_cfg)
    unknown = set(user_cfg) - set(_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown OPLD config key(s) {sorted(unknown)} under algorithm.easyopd.listwise. "
            f"Valid keys: {sorted(_DEFAULTS)}."
        )

    resolved.update(user_cfg)
    resolved["beta"] = float(resolved["beta"])
    resolved["eta"] = float(resolved["eta"])
    resolved["length_norm"] = bool(resolved["length_norm"])
    resolved["std_norm"] = bool(resolved["std_norm"])
    resolved["kl_direction"] = str(resolved["kl_direction"])

    if resolved["kl_direction"] not in ("qT_to_qS", "qS_to_qT"):
        raise ValueError(
            f"kl_direction must be 'qT_to_qS' or 'qS_to_qT', got {resolved['kl_direction']!r}."
        )
    return resolved


def _group_advantage(
    q_teacher: torch.Tensor,
    q_student: torch.Tensor,
    kl_direction: str,
) -> torch.Tensor:
    """Per-sequence advantage for one prompt group.

    With ``v_i = beta * s_i`` the softmax logits and ``s_i`` the (optionally
    length-normalized) student sequence logprob, the PG convention
    ``loss = -sum_i A_i * s_i`` means we need ``A_i = -dD/dv_i``.

    forward, ``D = KL(q_T || q_S)``:
        dD/dv_i = q_S(i) - q_T(i)      =>  A_i = q_T(i) - q_S(i)

    reverse, ``D = KL(q_S || q_T)``:
        let w_i = log q_S(i) - log q_T(i)
        dD/dv_i = q_S(i) * (w_i - sum_j q_S(j) w_j)
                =>  A_i = -q_S(i) * (w_i - D)

    Both are mean-zero over the group, so no extra baseline is required.
    """
    if kl_direction == "qS_to_qT":
        w = torch.log(q_student + _EPS) - torch.log(q_teacher + _EPS)
        kl = (q_student * w).sum()
        return -q_student * (w - kl)
    # "qT_to_qS" (default)
    return q_teacher - q_student


def compute_listwise_advantage(
    teacher_seq_logprobs: torch.Tensor,
    student_seq_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    response_lengths: Optional[torch.Tensor] = None,
    token_level_rewards: Optional[torch.Tensor] = None,
    config: Any = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Listwise on-policy distillation advantage.

    On-policy, ``student_seq_logprobs`` are the rollout-time (old) logprobs, so
    q_S is the student's own group-softmax at the update start; the PPO ratio in
    the downstream PG loss absorbs within-update drift.

    Args:
        teacher_seq_logprobs: (bsz,) masked sum of teacher token logprobs.
        student_seq_logprobs: (bsz,) masked sum of student (old) token logprobs.
        response_mask: (bsz, response_length).
        index: (bsz,) prompt-group ids (uid) grouping the K candidates.
        response_lengths: (bsz,) response token counts. Defaults to the row sums
            of ``response_mask``. Only used when ``length_norm`` is on.
        token_level_rewards: (bsz, response_length). Only used when ``eta != 0``.
        config: AlgoConfig, read via :func:`read_listwise_config`.

    Returns:
        ``(advantages, metrics)`` where advantages is (bsz, response_length),
        the per-sequence A_i broadcast over response tokens and masked.
    """
    cfg = read_listwise_config(config)
    beta = cfg["beta"]
    eta = cfg["eta"]

    device = response_mask.device
    teacher_seq_logprobs = teacher_seq_logprobs.to(device=device, dtype=torch.float32)
    student_seq_logprobs = student_seq_logprobs.to(device=device, dtype=torch.float32)

    if response_lengths is None:
        response_lengths = response_mask.sum(dim=-1)
    response_lengths = response_lengths.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Per-sequence scores that feed the group softmax (softmax logits).
        s_teacher = teacher_seq_logprobs
        s_student = student_seq_logprobs
        if cfg["length_norm"]:
            # NOTE: this normalizes the softmax logits only. The downstream PG term
            # differentiates log(pi) rather than log(pi)/L, so strictly the advantage
            # should also carry a 1/L_i factor. We keep the (standard) approximation
            # that response lengths within a group are comparable.
            denom = response_lengths.clamp_min(1.0)
            s_teacher = s_teacher / denom
            s_student = s_student / denom

        u = beta * s_teacher  # teacher-target logits
        if eta != 0.0:
            if token_level_rewards is None:
                raise ValueError("eta != 0 requires token_level_rewards to be provided.")
            u = u + eta * (token_level_rewards * response_mask).sum(dim=-1)
        v = beta * s_student  # student logits

        bsz = u.shape[0]
        # Group logits by prompt id.
        id2pos: dict[Any, list[int]] = defaultdict(list)
        for i in range(bsz):
            id2pos[index[i]].append(i)

        seq_adv = torch.zeros(bsz, dtype=torch.float32, device=device)
        n_degenerate = 0
        for pos in id2pos.values():
            if len(pos) == 1:
                # Single-candidate group: q_T = q_S = 1 -> A = 0. Nothing to learn
                # from a listwise comparison of one item (matches GRPO's degenerate
                # group). Counted so the caller can warn about rollout.n == 1.
                n_degenerate += 1
                continue
            idx = torch.tensor(pos, device=device)
            q_teacher = torch.softmax(u[idx], dim=0)
            q_student = torch.softmax(v[idx], dim=0)
            a = _group_advantage(q_teacher, q_student, cfg["kl_direction"])
            if cfg["std_norm"]:
                a = a / (a.std(unbiased=False) + _EPS)
            seq_adv[idx] = a

        advantages = seq_adv.unsqueeze(-1) * response_mask

    n_groups = len(id2pos)
    metrics = {
        "opld/num_groups": float(n_groups),
        "opld/degenerate_group_frac": float(n_degenerate) / max(n_groups, 1),
        "opld/seq_adv_mean": seq_adv.mean().item(),
        "opld/seq_adv_std": seq_adv.std(unbiased=False).item(),
        "opld/seq_adv_absmax": seq_adv.abs().max().item() if seq_adv.numel() else 0.0,
        "opld/teacher_seq_logprob_mean": teacher_seq_logprobs.mean().item(),
        "opld/student_seq_logprob_mean": student_seq_logprobs.mean().item(),
    }
    return advantages, metrics
