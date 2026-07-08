# Copyright 2026 EasyOPD Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SDPO: Self-Distillation Policy Optimization — core algorithm.

Paper: "Reinforcement Learning via Self-Distillation"
       Hübotter et al., 2026 (https://arxiv.org/abs/2601.20802)
       Code:  https://github.com/lasgroup/SDPO

This module is a *faithful* reimplementation of the lasgroup/SDPO reference
algorithm inside EasyOPD's verl fork. It mirrors, line-for-line where it
matters, the reference behaviour:

    L_SDPO(theta) = sum_t  D_JSD^alpha( pi_theta(.|x, y_<t)
                                        || stopgrad(pi_tilde(.|x, f, y_<t)) )

where the *student* is the policy conditioned only on the question ``x`` and the
*self-teacher* ``pi_tilde`` is an **EMA copy** of the policy conditioned on rich
feedback ``f`` (a correct demonstration from the rollout group and/or
environment feedback). Faithful to the reference, the teacher:
    * is a SEPARATE module (initialised from the reference / base model) that is
      EMA-updated towards the policy each optimizer step
      (``teacher_regularization="ema"``, ``teacher_update_rate=0.05``); see
      ``compute_ema_update`` and ``DataParallelPPOActor._update_teacher``;
    * is fed the *reprompted* context (reprompt prompt + the ORIGINAL response)
      so it re-scores the student's own tokens under hindsight feedback;
    * is stop-gradient'd.

Faithful behaviour vs. the reference (lasgroup/SDPO):
    * distillation is **logit-level top-K generalized JSD** over the *student's*
      top-K vocab subset (the reference gathers the teacher at the student's
      top-K indices), plus an optional tail bucket;
    * samples without a usable self-teacher (``self_distillation_mask == 0``)
      contribute **zero** gradient (no GRPO fallback);
    * the per-token loss is multiplied by token-level rollout-correction IS
      weights (``rollout_is_weights``) when present.

Public API:
    * ``compute_sdpo_self_distillation_loss`` — the self-distillation loss
      (full-logit / top-K / token-level), matching the reference.
    * ``build_reprompt_text`` — render the reprompt text from templates.
    * ``select_demonstration`` — pick a successful demonstration from the group.
    * ``compute_ema_update`` — EMA teacher parameter update.
    * ``build_sdpo_teacher_inputs`` — build the reprompted self-teacher batch.
    * ``compute_sdpo_actor_loss`` — actor-side orchestration (EMA-teacher
      forward + loss), called from ``dp_actor``.
    * ``compute_sdpo_loss`` — backward-compatible thin wrapper (used by the
      EasyOPD hook adapter; not on the training hot path).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "compute_sdpo_self_distillation_loss",
    "compute_sdpo_loss",
    "build_reprompt_text",
    "select_demonstration",
    "compute_ema_update",
    "build_sdpo_teacher_inputs",
    "compute_sdpo_actor_loss",
    "remove_thinking_from_text",
    "sequence_rewards",
]

_LOG_RATIO_CLAMP = 20.0
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)

# Optional one-time debug dump of the actual self-teacher reprompt input (set
# SDPO_DEBUG_TEACHER_INPUT=1 to enable; off by default). Verifies byte-for-byte
# what the teacher re-scores: system prompt present? demonstration injected?
# response aligned at the tail? — useful when wiring up new reprompt datasets.
_SDPO_TEACHER_DUMPED = False


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a tail bucket ``log(1 - sum(exp(log_probs)))`` so a top-K slice
    forms a proper distribution (captures the residual mass outside top-K)."""
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)  # avoid log(0) when top-K already covers ~1
    tail = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail], dim=-1)


def _renorm_topk(log_probs: torch.Tensor) -> torch.Tensor:
    """Renormalize top-K log-probs to sum to 1 (no tail bucket)."""
    return log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)


def _distribution_entropy(distill_log_probs: torch.Tensor) -> torch.Tensor:
    """Per-token Shannon entropy (nats) of a distribution given as log-probs
    ``[B, T, K(+1)]`` (must already sum to 1 over the last dim, e.g. the top-K +
    tail-bucket distributions used for distillation).

    SDPO health diagnostic: a *teacher* entropy that climbs over training is the
    signature of an EMA self-teacher drifting toward high-entropy / rambling
    targets (the failure mode where the student is distilled into ever longer,
    format-violating responses)."""
    probs = distill_log_probs.exp()
    return -(probs * distill_log_probs).sum(dim=-1)


def _aggregate(per_token_loss: torch.Tensor, loss_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Mask + aggregate a per-token loss (matches verl ``agg_loss`` with
    ``batch_num_tokens = loss_mask.sum()`` for ``token-mean``)."""
    if mode == "seq-mean-token-sum":
        seq = (per_token_loss * loss_mask).sum(dim=-1)
        return seq.mean()
    if mode == "seq-mean-token-mean":
        seq = (per_token_loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1.0)
        return seq.mean()
    # default: token-mean
    return (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Loss (reference: core_algos.compute_self_distillation_loss)
# ---------------------------------------------------------------------------

def compute_sdpo_self_distillation_loss(
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
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """SDPO self-distillation loss — faithful to lasgroup/SDPO.

    Args:
        student_log_probs: ``[B, T]`` per-token log-probs of the chosen response
            tokens under the student context ``pi(y_t | x, y_<t)`` (carries grad;
            used for the IS ratio and the token-level estimator).
        teacher_log_probs: ``[B, T]`` per-token log-probs of the SAME tokens under
            the feedback-informed self-teacher (only used by the token-level,
            non-full-logit estimator).
        response_mask: ``[B, T]`` mask of valid response tokens.
        alpha: generalized-JSD interpolation. ``0.0`` -> ``KL(teacher||student)``,
            ``1.0`` -> ``KL(student||teacher)``, ``0.5`` -> symmetric JSD.
        full_logit_distillation: distill whole (top-K) distributions vs. the
            token-level estimator.
        distillation_topk: if set (with ``full_logit_distillation``), distill over
            the top-K vocab subset using ``student_topk_log_probs`` /
            ``teacher_topk_log_probs``.
        distillation_add_tail: add a tail bucket for the residual top-K mass.
        is_clip: clip the IS ratio ``exp(student-old)`` (off-policy correction).
        old_log_probs: behaviour-policy log-probs (required if ``is_clip`` set).
        student_all_log_probs / teacher_all_log_probs: ``[B, T, V]`` full-vocab
            log-softmax (full-logit path without top-K).
        student_topk_log_probs / teacher_topk_log_probs: ``[B, T, K]`` top-K
            log-softmax over the SAME vocab indices (top-K path).
        self_distillation_mask: ``[B]`` mask selecting samples that have a
            self-teacher; others are excluded (zero gradient).
        loss_agg_mode: ``token-mean`` | ``seq-mean-token-sum`` |
            ``seq-mean-token-mean``.
        rollout_is_weights: ``[B, T]`` token-level rollout-correction IS weights.

    Returns:
        ``(loss, metrics)``.
    """
    metrics: dict = {}
    # [EasyOPD:SDPO diag] per-token entropy of the distillation distributions
    # (only computed on the full-logit / top-K path; None for the token-level
    # fallback). Reported below as sdpo/{student,teacher}_entropy.
    student_entropy_tok: Optional[torch.Tensor] = None
    teacher_entropy_tok: Optional[torch.Tensor] = None

    loss_mask = response_mask.float()
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.to(loss_mask.dtype).unsqueeze(1)

    if full_logit_distillation:
        use_topk = distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError(
                    "top-k distillation requires student_topk_log_probs and teacher_topk_log_probs."
                )
            student_distill = student_topk_log_probs
            teacher_distill = teacher_topk_log_probs
            if distillation_add_tail:
                student_distill = _add_tail_bucket(student_distill)
                teacher_distill = _add_tail_bucket(teacher_distill)
            else:
                student_distill = _renorm_topk(student_distill)
                teacher_distill = _renorm_topk(teacher_distill)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError(
                    "full_logit_distillation requires student_all_log_probs and teacher_all_log_probs."
                )
            student_distill = student_all_log_probs
            teacher_distill = teacher_all_log_probs

        # [EasyOPD:SDPO diag] entropy of the student vs. (stop-grad) self-teacher
        # distillation distributions. Detached: diagnostic only, no grad.
        with torch.no_grad():
            student_entropy_tok = _distribution_entropy(student_distill)
            teacher_entropy_tok = _distribution_entropy(teacher_distill)

        if alpha == 0.0:
            kl_loss = F.kl_div(student_distill, teacher_distill, reduction="none", log_target=True)
        elif alpha == 1.0:
            kl_loss = F.kl_div(teacher_distill, student_distill, reduction="none", log_target=True)
        else:
            a = torch.tensor(alpha, dtype=student_distill.dtype, device=student_distill.device)
            # log of the mixture m = (1-a) * student + a * teacher
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_distill + torch.log(1 - a), teacher_distill + torch.log(a)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, a)  # generalized Jensen-Shannon divergence

        per_token_loss = kl_loss.sum(-1)
    else:
        assert alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-_LOG_RATIO_CLAMP, max=_LOG_RATIO_CLAMP)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    # Rollout-correction (training-vs-rollout policy mismatch) IS weights.
    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = _aggregate(per_token_loss, loss_mask, loss_agg_mode)

    with torch.no_grad():
        valid = loss_mask.sum().clamp(min=1.0)
        metrics["sdpo/loss"] = loss.detach().item()
        metrics["sdpo/num_distill_tokens"] = loss_mask.sum().item()
        metrics["sdpo/per_token_mean"] = (per_token_loss.detach() * loss_mask).sum().item() / valid.item()
        if self_distillation_mask is not None:
            metrics["sdpo/teacher_fraction"] = self_distillation_mask.float().mean().item()
        # [EasyOPD:SDPO diag] mean entropy of the distillation distributions over
        # the distilled tokens. A rising sdpo/teacher_entropy (especially when it
        # leads sdpo/student_entropy) is direct evidence that the EMA self-teacher
        # is the source of the collapse (drifting to high-entropy targets).
        if student_entropy_tok is not None:
            metrics["sdpo/student_entropy"] = (student_entropy_tok * loss_mask).sum().item() / valid.item()
        if teacher_entropy_tok is not None:
            metrics["sdpo/teacher_entropy"] = (teacher_entropy_tok * loss_mask).sum().item() / valid.item()
        if student_entropy_tok is not None and teacher_entropy_tok is not None:
            metrics["sdpo/teacher_minus_student_entropy"] = (
                ((teacher_entropy_tok - student_entropy_tok) * loss_mask).sum().item() / valid.item()
            )
    return loss, metrics


def compute_sdpo_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.5,
    is_clip: Optional[float] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    distillation_add_tail: bool = True,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Backward-compatible thin wrapper around
    :func:`compute_sdpo_self_distillation_loss` for the EasyOPD hook adapter.

    NOTE: this is NOT on the SDPO training hot path (the ``dp_actor`` SDPO branch
    calls :func:`compute_sdpo_actor_loss` directly). It exists so that
    ``easyopd.methods.sdpo.hooks.SDPOLossHook`` keeps working.
    """
    logit_level = student_topk_log_probs is not None and teacher_topk_log_probs is not None
    if logit_level:
        return compute_sdpo_self_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            alpha=alpha,
            full_logit_distillation=True,
            distillation_topk=int(student_topk_log_probs.shape[-1]),
            distillation_add_tail=distillation_add_tail,
            is_clip=is_clip,
            old_log_probs=old_log_probs,
            student_topk_log_probs=student_topk_log_probs,
            teacher_topk_log_probs=teacher_topk_log_probs,
            self_distillation_mask=self_distillation_mask,
            loss_agg_mode=loss_agg_mode,
            rollout_is_weights=rollout_is_weights,
        )
    # Token-level reverse-KL estimator (reference only supports reverse KL here).
    return compute_sdpo_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        alpha=1.0,
        full_logit_distillation=False,
        is_clip=is_clip,
        old_log_probs=old_log_probs,
        self_distillation_mask=self_distillation_mask,
        loss_agg_mode=loss_agg_mode,
        rollout_is_weights=rollout_is_weights,
    )


# ---------------------------------------------------------------------------
# EMA teacher (reference: dp_actor._update_teacher)
# ---------------------------------------------------------------------------

def compute_ema_update(
    student_params: dict,
    teacher_params: dict,
    update_rate: float,
) -> dict:
    """EMA update of the teacher parameters towards the student.

    ``teacher <- (1 - update_rate) * teacher + update_rate * student``

    Args:
        student_params: name -> tensor (current policy parameters).
        teacher_params: name -> tensor (self-teacher parameters).
        update_rate: EMA mixing coefficient in ``[0, 1]``. ``0`` leaves the
            teacher unchanged; ``1`` copies the student.

    Returns:
        A new dict of updated teacher parameters. Keys present only in the
        teacher are preserved unchanged.
    """
    updated: dict = {}
    for name, teacher_val in teacher_params.items():
        if name in student_params:
            student_val = student_params[name].to(device=teacher_val.device, dtype=teacher_val.dtype)
            updated[name] = (1.0 - update_rate) * teacher_val + update_rate * student_val
        else:
            updated[name] = teacher_val
    return updated


# ---------------------------------------------------------------------------
# Reprompt construction (reference: ray_trainer._maybe_build_self_distillation_batch)
# ---------------------------------------------------------------------------

def remove_thinking_from_text(text: str) -> str:
    """Strip ``<think>...</think>`` spans from a demonstration."""
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


def sequence_rewards(reward_tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a token-level reward tensor ``[B, T]`` to per-sequence ``[B]``."""
    if reward_tensor is None:
        raise ValueError("[EasyOPD:sdpo] reward_tensor is required to build SDPO reprompts.")
    if reward_tensor.dim() == 1:
        return reward_tensor
    return reward_tensor.sum(dim=-1)


def build_reprompt_text(
    prompt_text: str,
    solution: Optional[str],
    feedback: Optional[str],
    reprompt_template: str,
    solution_template: str,
    feedback_template: str,
    feedback_only_without_solution: bool = False,
) -> str:
    """Render the self-teacher reprompt text from templates.

    Mirrors the reference ``_build_teacher_message``: when neither a solution
    nor (usable) feedback is available the ORIGINAL prompt is returned verbatim.
    """
    has_solution = solution is not None
    has_feedback = feedback is not None
    # If feedback_only_without_solution is True, only use feedback when no solution exists.
    use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

    solution_section = solution_template.format(successful_previous_attempt=solution) if has_solution else ""
    feedback_section = feedback_template.format(feedback_raw=feedback) if use_feedback else ""

    if use_feedback or has_solution:
        return reprompt_template.format(
            prompt=prompt_text,
            solution=solution_section,
            feedback=feedback_section,
        )
    return prompt_text


def select_demonstration(
    idx: int,
    success_by_uid: dict,
    uids: Any,
    response_texts: list,
    dont_reprompt_on_self_success: bool = False,
    remove_thinking_from_demonstration: bool = False,
) -> Optional[str]:
    """Pick a successful demonstration for sample ``idx`` from its rollout group.

    Mirrors the reference ``_get_solution``: takes the first successful response
    in the same ``uid`` group (optionally excluding the sample itself), and
    optionally strips its thinking trace.
    """
    uid = uids[idx]
    solution_idxs = list(success_by_uid.get(uid, []))
    if dont_reprompt_on_self_success:
        solution_idxs = [j for j in solution_idxs if j != idx]
    if len(solution_idxs) == 0:
        return None
    solution_idx = solution_idxs[0]  # first success in the group (effectively random)
    solution_str = response_texts[solution_idx]
    if remove_thinking_from_demonstration:
        solution_str = remove_thinking_from_text(solution_str)
    return solution_str


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return cfg.get(key, default)
        except Exception:  # noqa: BLE001
            pass
    return getattr(cfg, key, default)


def _question_messages_from_raw_prompt(raw_prompt: Any) -> Tuple[list, str]:
    """Return ``(system_messages, user_question)`` from a verl ``raw_prompt``.

    ``system_messages`` is everything before the final turn (system / history),
    preserved verbatim — matching the reference ``raw_prompt[:-1]``.
    """
    if raw_prompt is None:
        return [], ""
    if isinstance(raw_prompt, str):
        return [], raw_prompt
    try:
        messages = [dict(m) if isinstance(m, dict) else m for m in list(raw_prompt)]
    except TypeError:
        return [], str(raw_prompt)
    if not messages:
        return [], ""
    last = messages[-1]
    user_text = str(last.get("content", "")) if isinstance(last, dict) else str(last)
    return messages[:-1], user_text


def _collect_solutions_by_uid(
    uids: Any,
    seq_rewards: torch.Tensor,
    success_reward_threshold: float,
) -> dict:
    """Group indices of successful samples by their ``uid``."""
    success_by_uid: dict = defaultdict(list)
    seq_scores = seq_rewards.detach().to("cpu").tolist()
    for idx, uid in enumerate(uids):
        if seq_scores[idx] >= success_reward_threshold:
            success_by_uid[uid].append(idx)
    return success_by_uid


def build_sdpo_teacher_inputs(
    batch: Any,
    reward_tensor: torch.Tensor,
    cfg: Any,
    tokenizer: Any,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> Optional[Tuple[dict, dict]]:
    """Build the SDPO self-teacher batch by reprompting rollouts.

    For every sample the self-teacher prompt injects (a) a correct demonstration
    drawn from a successful rollout in the *same* prompt group (``uid``) and/or
    (b) textual environment feedback, then appends the sample's ORIGINAL response
    so the teacher re-scores the student's own tokens in hindsight (reference
    ``_maybe_build_self_distillation_batch``).

    Returns ``(tensors, metrics)`` where ``tensors`` holds ``teacher_input_ids``,
    ``teacher_attention_mask``, ``teacher_position_ids``,
    ``teacher_response_start_idx`` and ``self_distillation_mask``; or ``None`` if
    the batch lacks the fields needed to build reprompts.
    """
    import numpy as np

    if "responses" not in batch.batch.keys() or "response_mask" not in batch.batch.keys():
        return None

    device = batch.batch["responses"].device
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]
    batch_size = responses.shape[0]

    uids = batch.non_tensor_batch.get("uid", None)
    if uids is None:
        # Without group ids we cannot find demonstrations; fall back to feedback-only.
        uids = np.arange(batch_size)

    # The self-teacher MUST see the original question (and system prompt) to give
    # useful hindsight supervision. verl only populates ``raw_prompt`` when
    # ``data.return_raw_chat=True``; without it the reprompt silently drops the
    # question + system prompt and SDPO collapses (the teacher re-scores responses
    # for an unknown question). Fail fast instead of training on a broken target.
    raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)
    if raw_prompts is None or all(rp is None for rp in raw_prompts):
        raise ValueError(
            "[EasyOPD:sdpo] batch is missing 'raw_prompt' (the original chat messages); "
            "the self-teacher reprompt would drop the question + system prompt and SDPO "
            "would collapse. Set 'data.return_raw_chat=True' in the launch command."
        )

    success_threshold = float(_cfg_get(cfg, "success_reward_threshold", 1.0))
    dont_reprompt_on_self_success = bool(_cfg_get(cfg, "dont_reprompt_on_self_success", True))
    remove_thinking = bool(_cfg_get(cfg, "remove_thinking_from_demonstration", True))
    include_feedback = bool(_cfg_get(cfg, "include_environment_feedback", False))
    feedback_only_without_solution = bool(
        _cfg_get(cfg, "environment_feedback_only_without_solution", False)
    )
    max_prompt_len = int(_cfg_get(cfg, "max_reprompt_len", 10240))
    reprompt_truncation = str(_cfg_get(cfg, "reprompt_truncation", "right"))
    # Defaults mirror lasgroup/SDPO actor.yaml exactly (no trailing newline),
    # so the self-teacher reprompt text is byte-faithful to the reference.
    reprompt_template = _cfg_get(
        cfg, "reprompt_template",
        "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.",
    )
    solution_template = _cfg_get(
        cfg, "solution_template",
        "\nCorrect solution:\n\n{successful_previous_attempt}",
    )
    feedback_template = _cfg_get(
        cfg, "feedback_template",
        "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}",
    )

    # Per-sample decoded responses (used as demonstrations) — decode the FULL
    # response row with special tokens skipped (reference parity).
    response_texts: list[str] = [
        tokenizer.decode(responses[i].tolist(), skip_special_tokens=True) for i in range(batch_size)
    ]

    seq_rewards = sequence_rewards(reward_tensor)
    success_by_uid = _collect_solutions_by_uid(uids, seq_rewards, success_threshold)
    solution_strs = [
        select_demonstration(
            i, success_by_uid, uids, response_texts,
            dont_reprompt_on_self_success=dont_reprompt_on_self_success,
            remove_thinking_from_demonstration=remove_thinking,
        )
        for i in range(batch_size)
    ]

    optional_feedback = batch.non_tensor_batch.get("privileged_context", None)
    if optional_feedback is None:
        optional_feedback = batch.non_tensor_batch.get("feedback", None)

    def _feedback_for(i: int) -> Optional[str]:
        if not include_feedback or optional_feedback is None:
            return None
        try:
            raw_fb = optional_feedback[i]
        except (IndexError, KeyError, TypeError):
            return None
        if raw_fb is None or not str(raw_fb).strip():
            return None
        return str(raw_fb)

    apply_kwargs = dict(apply_chat_template_kwargs or {})
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    resp_lens = response_mask.bool().sum(dim=1).to("cpu").tolist()

    teacher_prompt_ids_list: list[torch.Tensor] = []
    teacher_prompt_mask_list: list[torch.Tensor] = []
    teacher_present_mask_list: list[float] = []

    for i in range(batch_size):
        solution = solution_strs[i]
        feedback = _feedback_for(i)
        # use_feedback mirrors build_reprompt_text's gating (for the present mask).
        use_feedback = feedback is not None and (
            not feedback_only_without_solution or solution is None
        )
        has_teacher = (solution is not None) or use_feedback

        system_messages, question = _question_messages_from_raw_prompt(raw_prompts[i])

        if has_teacher and int(resp_lens[i]) > 0:
            teacher_present_mask_list.append(1.0)
            reprompt_text = build_reprompt_text(
                prompt_text=question,
                solution=solution,
                feedback=feedback,
                reprompt_template=reprompt_template,
                solution_template=solution_template,
                feedback_template=feedback_template,
                feedback_only_without_solution=feedback_only_without_solution,
            )
        else:
            # No usable self-teacher: placeholder (original prompt) with mask 0.
            teacher_present_mask_list.append(0.0)
            reprompt_text = question

        teacher_messages = system_messages + [{"role": "user", "content": reprompt_text}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                teacher_messages, tokenize=False, add_generation_prompt=True, **apply_kwargs,
            )
        except Exception:  # noqa: BLE001 — tokenizer without chat template
            prompt_text = teacher_messages[-1]["content"]

        side = "left" if reprompt_truncation == "left" else "right"
        prev_side = getattr(tokenizer, "truncation_side", "right")
        tokenizer.truncation_side = side
        try:
            enc = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=max_prompt_len,
            )
        finally:
            tokenizer.truncation_side = prev_side

        teacher_prompt_ids_list.append(enc["input_ids"].squeeze(0).to("cpu"))
        teacher_prompt_mask_list.append(enc["attention_mask"].squeeze(0).to("cpu"))

    # Assemble each teacher sequence as: LEFT-PAD + reprompt prompt + FULL response.
    # The response block (length == response_length) stays at the TAIL so verl's
    # `_forward_micro_batch` (which slices the last `responses.size(-1)` positions)
    # returns teacher log-probs aligned 1:1 with the student's.
    max_tp_len = max((int(p.numel()) for p in teacher_prompt_ids_list), default=1)
    teacher_input_ids_list: list[torch.Tensor] = []
    teacher_attention_mask_list: list[torch.Tensor] = []
    teacher_response_start_idx_list: list[torch.Tensor] = []
    for i in range(batch_size):
        p_ids = teacher_prompt_ids_list[i]
        p_mask = teacher_prompt_mask_list[i]
        left = int(max_tp_len - p_ids.numel())
        left_ids = torch.full((left,), pad_id, dtype=p_ids.dtype)
        left_mask = torch.zeros(left, dtype=p_mask.dtype)
        resp_ids = responses[i].detach().to("cpu").to(p_ids.dtype)
        resp_mask = response_mask[i].detach().to("cpu").to(p_mask.dtype)
        teacher_input_ids_list.append(torch.cat([left_ids, p_ids, resp_ids], dim=0))
        teacher_attention_mask_list.append(torch.cat([left_mask, p_mask, resp_mask], dim=0))
        teacher_response_start_idx_list.append(torch.tensor([max_tp_len], dtype=torch.long))

    teacher_input_ids = torch.stack(teacher_input_ids_list).to(device)
    teacher_attention_mask = torch.stack(teacher_attention_mask_list).to(device)
    teacher_position_ids = torch.clip(teacher_attention_mask.cumsum(dim=-1) - 1, min=0).to(device)
    teacher_present_mask = torch.tensor(teacher_present_mask_list, dtype=torch.float32, device=device)

    tensors = {
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attention_mask,
        "teacher_position_ids": teacher_position_ids,
        "teacher_response_start_idx": torch.stack(teacher_response_start_idx_list).to(device),
        "self_distillation_mask": teacher_present_mask,
    }
    num_with_solution = sum(1 for s in solution_strs if s is not None)
    metrics = {
        "self_distillation/reprompt_sample_fraction": teacher_present_mask.mean().item(),
        "self_distillation/success_sample_fraction": float(num_with_solution) / max(batch_size, 1),
        "self_distillation/policy_fallback_fraction": (1.0 - teacher_present_mask.mean()).item(),
    }

    # ---- one-time teacher-input dump (byte-level reprompt verification) ----
    global _SDPO_TEACHER_DUMPED
    import os as _os
    if not _SDPO_TEACHER_DUMPED and _os.environ.get("SDPO_DEBUG_TEACHER_INPUT", "0") == "1":
        _SDPO_TEACHER_DUMPED = True
        try:
            j = next((i for i in range(batch_size) if teacher_present_mask_list[i] > 0.5), 0)
            R = int(responses.shape[1])
            full = teacher_input_ids[j][teacher_attention_mask[j].bool()]
            tail = teacher_input_ids[j][-R:]
            tail_m = teacher_attention_mask[j][-R:]
            logger.warning(
                "[EasyOPD:sdpo][teacher-dump] sample=%d has_teacher=%.0f resp_start_idx=%d "
                "teacher_seq_len=%d demo_present=%s\n"
                "===== FULL teacher input (decoded) =====\n%s\n"
                "===== appended RESPONSE tail (what teacher re-scores) =====\n%s\n"
                "========================================",
                j, teacher_present_mask_list[j], int(max_tp_len), int(full.numel()),
                solution_strs[j] is not None,
                tokenizer.decode(full.tolist()),
                tokenizer.decode(tail[tail_m.bool()].tolist()),
            )
        except Exception as _e:  # noqa: BLE001 - never let a debug dump crash training
            logger.warning("[EasyOPD:sdpo][teacher-dump] failed: %r", _e)

    return tensors, metrics


# ---------------------------------------------------------------------------
# Actor-side orchestration (called from dp_actor's SDPO branch)
# ---------------------------------------------------------------------------

def compute_sdpo_actor_loss(
    *,
    model_inputs: dict,
    student_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_cfg: Any,
    loss_agg_mode: str,
    temperature: float,
    forward_fn: Callable,
    teacher_module: Any = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    student_topk_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute the SDPO self-distillation loss for one micro-batch — faithful.

    Pipeline (reference ``dp_actor`` SDPO branch):
        1. Student top-K log-probs + indices over the original context. Faithful
           to lasgroup/SDPO these are extracted in the SAME forward that produces
           ``log_prob``; ``dp_actor`` passes them in via ``student_topk_log_probs``
           / ``student_topk_indices`` so no second student forward is needed. When
           they are not supplied (e.g. the hook adapter path), a dedicated student
           forward (WITH grad) is run here as a fallback.
        2. Self-teacher forward (NO grad, on the EMA ``teacher_module``) over the
           reprompted context, gathered at the *student's* top-K indices.
        3. Logit-level generalized-JSD distillation loss, masked to samples that
           actually have a self-teacher (``self_distillation_mask``); samples
           without a teacher contribute zero gradient (no GRPO fallback).

    Args:
        model_inputs: micro-batch tensors; must contain ``teacher_input_ids``,
            ``teacher_attention_mask``, ``teacher_position_ids``, ``responses``
            and (optionally) ``self_distillation_mask``.
        student_log_prob: ``[B, T]`` student chosen-token log-probs (grad).
        old_log_prob: ``[B, T]`` behaviour-policy log-probs (for IS clip).
        response_mask: ``[B, T]`` response token mask.
        self_distillation_cfg: ``SelfDistillationConfig`` (dict-like).
        loss_agg_mode: loss aggregation mode.
        temperature: sampling temperature used for both forwards.
        forward_fn: ``actor._forward_micro_batch`` (supports ``module=``,
            ``opsa_topk_k=``, ``opsa_gather_indices=``).
        teacher_module: the EMA self-teacher module (``None`` -> live policy).
        rollout_is_weights: optional ``[B, T]`` rollout-correction IS weights.
        student_topk_log_probs: ``[B, T, K]`` student top-K log-softmax already
            extracted by the caller's main forward (grad). When provided together
            with ``student_topk_indices`` the in-function student forward is skipped.
        student_topk_indices: ``[B, T, K]`` vocab indices for the student top-K.

    Returns:
        ``(policy_loss, metrics)``.
    """
    if self_distillation_cfg is None:
        raise ValueError("[EasyOPD:sdpo] loss_mode='sdpo' requires actor.self_distillation config.")

    self_distillation_mask = model_inputs.get("self_distillation_mask", None)

    teacher_inputs = {
        "responses": model_inputs["responses"],
        "input_ids": model_inputs["teacher_input_ids"],
        "attention_mask": model_inputs["teacher_attention_mask"],
        "position_ids": model_inputs["teacher_position_ids"],
    }
    if "teacher_multi_modal_inputs" in model_inputs:
        teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]

    alpha = float(_cfg_get(self_distillation_cfg, "alpha", 0.5))
    is_clip = _cfg_get(self_distillation_cfg, "is_clip", None)
    full_logit = bool(_cfg_get(self_distillation_cfg, "full_logit_distillation", True))
    distillation_topk = _cfg_get(self_distillation_cfg, "distillation_topk", None)
    add_tail = bool(_cfg_get(self_distillation_cfg, "distillation_add_tail", True))
    use_topk = full_logit and distillation_topk is not None and int(distillation_topk) > 0

    student_topk_lp = None
    teacher_topk_lp = None
    teacher_log_prob = None

    if use_topk:
        # 1) Student top-K log-probs + indices. Prefer the values the caller already
        #    extracted in its MAIN forward (faithful to lasgroup/SDPO, which fuses
        #    top-K extraction into the same pass as log_prob); only run a dedicated
        #    student forward (WITH grad) here when they were not supplied.
        if student_topk_log_probs is not None and student_topk_indices is not None:
            student_topk_lp = student_topk_log_probs
            student_topk_idx = student_topk_indices
        else:
            student_out = forward_fn(
                model_inputs, temperature=temperature, calculate_entropy=False,
                opsa_topk_k=int(distillation_topk),
            )
            student_extra = student_out[2] if isinstance(student_out, tuple) and len(student_out) >= 3 else {}
            student_topk_lp = student_extra.get("topk_log_probs") if isinstance(student_extra, dict) else None
            student_topk_idx = student_extra.get("topk_indices") if isinstance(student_extra, dict) else None

        if student_topk_lp is not None and student_topk_idx is not None:
            # 2) Self-teacher forward (no grad, EMA module): gathered at the
            #    student's top-K indices (so both sides share the same support).
            with torch.no_grad():
                teacher_out = forward_fn(
                    teacher_inputs, temperature=temperature, calculate_entropy=False,
                    opsa_gather_indices=student_topk_idx, module=teacher_module,
                )
            teacher_extra = teacher_out[2] if isinstance(teacher_out, tuple) and len(teacher_out) >= 3 else {}
            teacher_topk_lp = teacher_extra.get("gathered_log_probs") if isinstance(teacher_extra, dict) else None

        if student_topk_lp is None or teacher_topk_lp is None:
            # top-K extraction unavailable (e.g. ulysses SP / fused kernels).
            use_topk = False
            student_topk_lp = None
            teacher_topk_lp = None

    if not use_topk:
        # Token-level fallback (reference supports reverse KL only here).
        with torch.no_grad():
            teacher_out = forward_fn(
                teacher_inputs, temperature=temperature, calculate_entropy=False,
                module=teacher_module,
            )
        teacher_log_prob = teacher_out[1] if isinstance(teacher_out, tuple) else teacher_out

    loss, metrics = compute_sdpo_self_distillation_loss(
        student_log_probs=student_log_prob,
        teacher_log_probs=teacher_log_prob if teacher_log_prob is not None else student_log_prob.detach(),
        response_mask=response_mask,
        alpha=alpha if use_topk else 1.0,
        full_logit_distillation=use_topk,
        distillation_topk=int(distillation_topk) if use_topk else None,
        distillation_add_tail=add_tail,
        is_clip=is_clip,
        old_log_probs=old_log_prob,
        student_topk_log_probs=student_topk_lp,
        teacher_topk_log_probs=teacher_topk_lp,
        self_distillation_mask=self_distillation_mask,
        loss_agg_mode=loss_agg_mode,
        rollout_is_weights=rollout_is_weights,
    )

    metrics["sdpo/logit_level"] = 1.0 if use_topk else 0.0
    return loss, metrics
