"""GAD critic update loop.

Drop-in replacement for the body of `DataParallelPPOCritic.update_critic`
when `gad.enable=true`. The body runs the critic forward twice per
micro-batch (once on student, once on teacher), computes the Bradley-
Terry pairwise loss, accumulates gradients, takes an optimizer step,
and returns a flat dict of per-step metrics (matching verl's own
`update_critic` return contract, which the CriticWorker mutates with
`metrics["perf/mfu/critic"] = ...`).

Validation: on every call we run `validate_gad_batch` against the
data's batch keys. This is cheap (a dict membership check) and the
single place where the GAD data contract is enforced at runtime.
"""

from __future__ import annotations

from typing import Any

from easyopd.methods.gad.core import (
    compute_discriminator_loss,
    discriminator_accuracy,
    summed_reward,
)
from easyopd.methods.gad.data_contract import validate_gad_batch


def _append(metrics: dict, **kvs) -> None:
    for k, v in kvs.items():
        metrics.setdefault(k, []).append(v)


def _reduce(metrics: dict) -> dict:
    return {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in metrics.items()}


def update_critic_step(worker: Any, data: Any):
    """Run one critic update step under GAD's discriminator semantics.

    Args:
        worker: a verl DataParallelPPOCritic instance. We use:
            * `_forward_micro_batch(micro_batch, compute_teacher=...)`
            * `_optimizer_step()`
            * `gradient_accumulation`, `config`, `critic_module`, `critic_optimizer`
        data: a DataProto whose `.batch` includes both student and teacher tensors.

    Returns:
        A flat ``dict[str, float]`` of metrics with keys
        ``critic/d_loss``, ``critic/d_acc``, ``critic/student_value_mean``,
        ``critic/teacher_value_mean``, ``critic/grad_norm``. The caller
        (verl's ``CriticWorker``) mutates this dict to attach
        ``perf/mfu/critic`` before wrapping it into a DataProto.
    """
    validate_gad_batch(data.batch)

    worker.critic_module.train()
    worker.critic_optimizer.zero_grad()

    use_dynamic_bsz = bool(getattr(worker.config, "use_dynamic_bsz", False))
    ppo_mini = int(getattr(worker.config, "ppo_mini_batch_size", 1))

    metrics: dict[str, list[float]] = {}

    micro_batches = data.split(int(getattr(worker.config, "ppo_micro_batch_size_per_gpu", ppo_mini)))

    for micro in micro_batches:
        # Re-use the micro-batch dict directly; the forward adapter will copy.
        mb = {**micro.batch, **micro.non_tensor_batch}

        student_vpreds = worker._forward_micro_batch(mb, compute_teacher=False)
        teacher_vpreds = worker._forward_micro_batch(mb, compute_teacher=True)

        # response_mask is sliced to the last `response_length` tokens of the relevant attention_mask.
        # Student: from student attention_mask. Teacher: from teacher_attention_mask.
        s_len = student_vpreds.shape[-1]
        t_len = teacher_vpreds.shape[-1]
        response_mask = mb["attention_mask"][:, -s_len:].to(student_vpreds.dtype)
        teacher_mask = mb["teacher_attention_mask"][:, -t_len:].to(teacher_vpreds.dtype)

        d_loss = compute_discriminator_loss(
            student_vpreds=student_vpreds,
            teacher_vpreds=teacher_vpreds,
            response_mask=response_mask,
            teacher_response_mask=teacher_mask,
        )
        d_acc = discriminator_accuracy(
            student_vpreds=student_vpreds,
            teacher_vpreds=teacher_vpreds,
            response_mask=response_mask,
            teacher_response_mask=teacher_mask,
        )
        s_mean = summed_reward(student_vpreds, response_mask).mean().item()
        t_mean = summed_reward(teacher_vpreds, teacher_mask).mean().item()

        if use_dynamic_bsz:
            loss = d_loss * (student_vpreds.shape[0] / max(ppo_mini, 1))
        else:
            loss = d_loss / max(int(getattr(worker, "gradient_accumulation", 1)), 1)

        loss.backward()

        _append(
            metrics,
            **{
                "critic/d_loss": d_loss.detach().item(),
                "critic/d_acc": float(d_acc),
                "critic/student_value_mean": s_mean,
                "critic/teacher_value_mean": t_mean,
            },
        )

    grad_norm = worker._optimizer_step()
    _append(metrics, **{"critic/grad_norm": float(grad_norm) if not hasattr(grad_norm, "item") else grad_norm.item()})

    return _reduce(metrics)
