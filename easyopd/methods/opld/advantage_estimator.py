"""Advantage-estimator integration for OPLD.

OPLD is a *driver-side* method. The group softmax needs all K rollouts of a
prompt at once, and the actor's per-microbatch loss cannot guarantee that the K
candidates of a uid co-locate on one rank / in one microbatch. So the listwise
KL is turned into an advantage on the driver (where the full batch and its
``uid`` grouping are available) and the standard PG path consumes it.

Two things are registered here:

1. ``@register_adv_est("listwise")`` -- the estimator itself.
2. A wrapper around ``verl.trainer.ppo.ray_trainer.compute_advantage`` that
   feeds the estimator the tensors it needs. The wrapper exists because
   ``compute_advantage`` only forwards ``old_log_probs`` / ``teacher_log_probs``
   for the hardcoded name ``on_policy_distillation`` (owned by lightning_opd);
   for any other estimator the generic branch passes only ``token_level_rewards``,
   ``response_mask``, ``config`` and ``index``. Patching from inside the method
   package keeps ``verl/`` untouched.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import register_adv_est

from easyopd.methods.opld.core import compute_listwise_advantage

logger = logging.getLogger(__name__)

ADV_ESTIMATOR_NAME = "listwise"

# Metrics from the most recent estimator call. The verl estimator signature has
# no channel for metrics, so we stash them here for optional logging.
_LAST_METRICS: dict[str, float] = {}


def get_last_metrics() -> dict[str, float]:
    """Return metrics recorded by the most recent advantage computation."""
    return dict(_LAST_METRICS)


# Teacher per-token logprob keys we know how to consume, in priority order.
# All are (bsz, response_length), already aligned with ``response_mask``.
#
# ``ref_log_prob`` is the online route: point the ref worker at the teacher via
# ``actor_rollout_ref.ref.model.path`` (fsdp_workers.py:673-680) and the frozen
# "reference" forward *is* the teacher scoring the student's rollouts. It lands
# on the batch at ray_trainer.py:2392, i.e. before compute_advantage (2484).
# Listed last so an explicit teacher key always wins if both are present.
_TEACHER_LOGPROB_KEYS = (
    "teacher_log_probs",       # lightning_opd offline parquet path
    "opsa_teacher_log_probs",  # OPSA online frozen-ref forward
    "ref_log_prob",            # generic online teacher-as-ref路线
)


class OPLDMissingTeacherLogprobs(RuntimeError):
    """Raised when no teacher log-probabilities are present on the batch."""


class OPLDMissingUID(RuntimeError):
    """Raised when the batch has no ``uid`` grouping to build lists from."""


def _find_teacher_logprobs(batch) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """Return the first available teacher per-token logprob tensor and its key."""
    keys = batch.batch.keys()
    for key in _TEACHER_LOGPROB_KEYS:
        if key in keys:
            return batch.batch[key], key
    return None, None


@register_adv_est(ADV_ESTIMATOR_NAME)
def compute_listwise_advantage_estimator(
    *,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index=None,
    teacher_seq_logprobs: torch.Tensor = None,
    student_seq_logprobs: torch.Tensor = None,
    response_lengths: torch.Tensor = None,
    config=None,
    **kwargs,  # absorbs gamma/lam/num_repeat/etc. from the generic dispatcher
) -> tuple[torch.Tensor, torch.Tensor]:
    """Listwise on-policy distillation advantage (verl estimator entry point).

    The sequence-level logprobs are pre-reduced by :func:`_compute_advantage_patched`
    (which has access to the full DataProto) and injected as kwargs.

    Returns:
        ``(advantages, returns)`` both (bsz, response_length). ``returns`` is a
        copy of ``advantages``; the PG path ignores it and no critic is used.
    """
    if index is None:
        raise OPLDMissingUID(
            "adv_estimator=listwise requires 'uid' in non_tensor_batch to group "
            "the K rollouts of each prompt."
        )
    if teacher_seq_logprobs is None or student_seq_logprobs is None:
        raise OPLDMissingTeacherLogprobs(
            "adv_estimator=listwise requires teacher/student sequence logprobs. "
            "These are injected by easyopd.methods.opld's compute_advantage patch; "
            "if you see this, the patch was not installed (is the method registered "
            "via easyopd.methods.opld.register()?)."
        )

    advantages, metrics = compute_listwise_advantage(
        teacher_seq_logprobs=teacher_seq_logprobs,
        student_seq_logprobs=student_seq_logprobs,
        response_mask=response_mask,
        index=index,
        response_lengths=response_lengths,
        token_level_rewards=token_level_rewards,
        config=config,
    )

    if metrics["opld/degenerate_group_frac"] > 0.5:
        logger.warning(
            "[EasyOPD:opld] %.0f%% of prompt groups have a single rollout, whose "
            "listwise advantage is identically zero. Set actor_rollout_ref.rollout.n > 1.",
            100.0 * metrics["opld/degenerate_group_frac"],
        )

    _LAST_METRICS.clear()
    _LAST_METRICS.update(metrics)
    return advantages, advantages.clone()


# ---------------------------------------------------------------------------
# compute_advantage patch
# ---------------------------------------------------------------------------

_PATCHED = False


class OPLDConflictingKLTerm(RuntimeError):
    """Raised when another KL objective would corrupt the listwise advantage."""


def _assert_no_conflicting_kl_term(algo_cfg) -> None:
    """Fail loudly if some other KL objective is also active.

    OPLD's advantage ``A_i = q_T(i) - q_S(i)`` *is* the gradient of the listwise
    KL, so no additional KL term belongs in the objective. In particular, launch
    scripts enable ``algorithm.token_kl_reg`` purely as a side channel to make
    ``main_ppo.add_ref_policy_worker`` spawn the ref worker (which we repurpose as
    the teacher). That is only safe while the regularizer stays inert -- if
    ``beta_max`` is ever raised above ``beta_min`` it starts blending a token-level
    ``ref_log_prob - old_log_probs`` signal into ``batch["advantages"]``
    (ray_trainer.py:1714-1731) and silently corrupts our advantage.
    """
    if algo_cfg is None:
        return

    cfg = algo_cfg.get("token_kl_reg", None)
    if cfg is None:
        return

    beta_max = cfg.get("beta_max", None) if hasattr(cfg, "get") else getattr(cfg, "beta_max", None)
    beta_min = cfg.get("beta_min", 0.0) if hasattr(cfg, "get") else getattr(cfg, "beta_min", 0.0)
    stepwise = (
        cfg.get("stepwise_enable", False) if hasattr(cfg, "get")
        else getattr(cfg, "stepwise_enable", False)
    )

    if stepwise:
        raise OPLDConflictingKLTerm(
            "algorithm.token_kl_reg.stepwise_enable=True is incompatible with "
            "adv_estimator=listwise: the SOD branch overwrites batch['advantages'] "
            "(ray_trainer.py:1640-1704). Set stepwise_enable=False."
        )

    if beta_max is not None and float(beta_max) > float(beta_min):
        raise OPLDConflictingKLTerm(
            f"algorithm.token_kl_reg is active (beta_max={beta_max} > beta_min={beta_min}) "
            "alongside adv_estimator=listwise. It would blend a token-level KL signal "
            "into the listwise advantage (ray_trainer.py:1714-1731). OPLD's advantage is "
            "already the exact gradient of the listwise KL, so no extra KL term is "
            "wanted. Leave beta_max unset (null) -- token_kl_reg.enable=True is only "
            "used to make the ref worker spawn."
        )


def _compute_advantage_patched(original):
    """Wrap ``compute_advantage`` to inject OPLD's sequence-level tensors."""

    def wrapper(data, adv_estimator, *args, **kwargs):
        name = getattr(adv_estimator, "value", adv_estimator)
        if name != ADV_ESTIMATOR_NAME:
            return original(data, adv_estimator, *args, **kwargs)

        _assert_no_conflicting_kl_term(kwargs.get("config"))

        # verl's generic branch builds adv_kwargs itself and would drop anything
        # we add here, so reduce to sequence level and pass through meta_info-free
        # keyword injection by calling the estimator ourselves.
        from verl.trainer.ppo.ray_trainer import compute_response_mask

        if "response_mask" not in data.batch.keys():
            data.batch["response_mask"] = compute_response_mask(data)
        response_mask = data.batch["response_mask"]

        if "uid" not in data.non_tensor_batch:
            raise OPLDMissingUID(
                "adv_estimator=listwise requires 'uid' in non_tensor_batch."
            )

        teacher_token_logprobs, key = _find_teacher_logprobs(data)
        if teacher_token_logprobs is None:
            raise OPLDMissingTeacherLogprobs(
                "adv_estimator=listwise found no teacher log-probabilities on the "
                f"batch (looked for {list(_TEACHER_LOGPROB_KEYS)}).\n"
                "Online route:  point the ref worker at the teacher with\n"
                "  actor_rollout_ref.ref.model.path=<TEACHER>\n"
                "so 'ref_log_prob' is computed before compute_advantage.\n"
                "Offline route: supply a parquet with a 'teacher_log_probs' column "
                "(see easyopd.methods.lightning_opd.data_curation.prepare).\n"
                "Note 'opsa_teacher_log_probs' is attached *after* compute_advantage "
                "in the current fit loop and is therefore not usable here."
            )
        logger.info("[EasyOPD:opld] using '%s' as the teacher logprob source", key)

        if teacher_token_logprobs.shape != response_mask.shape:
            raise ValueError(
                f"teacher logprobs '{key}' shape {tuple(teacher_token_logprobs.shape)} "
                f"does not match response_mask {tuple(response_mask.shape)}."
            )

        student_token_logprobs = data.batch["old_log_probs"]

        # Mask before summing: padding positions carry arbitrary logprobs.
        mask_f = response_mask.to(torch.float32)
        teacher_seq = (teacher_token_logprobs.to(torch.float32) * mask_f).sum(dim=-1)
        student_seq = (student_token_logprobs.to(torch.float32) * mask_f).sum(dim=-1)

        advantages, returns = compute_listwise_advantage_estimator(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=response_mask,
            index=data.non_tensor_batch["uid"],
            teacher_seq_logprobs=teacher_seq,
            student_seq_logprobs=student_seq,
            response_lengths=mask_f.sum(dim=-1),
            config=kwargs.get("config"),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    return wrapper


def install_compute_advantage_patch() -> None:
    """Install the ``compute_advantage`` wrapper (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return

    from verl.trainer.ppo import ray_trainer

    ray_trainer.compute_advantage = _compute_advantage_patched(ray_trainer.compute_advantage)
    _PATCHED = True
    logger.info("[EasyOPD:opld] installed compute_advantage patch for adv_estimator=listwise")
