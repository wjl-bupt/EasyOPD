# Copyright 2026 EasyOPD Contributors

"""Hook adapter for echo_kd: trivial MSE between student/teacher log-probs."""

import torch


class EchoKDLossHook:
    """Minimal LossHook: returns mean((s - t) ** 2) over masked positions."""

    def compute_loss(self, student_logits, teacher_logits, mask, config=None, **kwargs):
        diff = (student_logits - teacher_logits) ** 2
        if diff.dim() == mask.dim() + 1:
            diff = diff.sum(dim=-1)
        denom = mask.sum().clamp(min=1)
        loss = (diff * mask).sum() / denom
        return loss, {"echo_kd/mean_sq_diff": float(loss.detach().item())}
