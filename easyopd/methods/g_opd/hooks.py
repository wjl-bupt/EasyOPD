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

"""Hook adapters for the `g_opd` (Generalized On-Policy Distillation) method.

Wraps the existing G-OPD core functions into the EasyOPD hook interfaces.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, LossHook, Metrics, RewardHook, TeacherSidecarHook


class GOPDLossHook:
    """LossHook adapter for the G-OPD method.

    Wraps `compute_g_opd_advantages` into the standard LossHook interface.
    G-OPD modifies the advantage computation with reward scaling and
    flexible reference models.
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute G-OPD loss with reward extrapolation.

        Args:
            student_logits: Student log-probs.
            teacher_logits: Teacher/reference log-probs.
            mask: Response mask.
            config: G-OPD configuration (reward_scale, ref_mode, etc.).
            **kwargs: Must contain 'rewards', 'old_log_probs', 'ref_log_probs'.

        Returns:
            (loss, metrics) with G-OPD advantage statistics.
        """
        from easyopd.methods.g_opd.core import compute_g_opd_advantages

        old_log_probs = kwargs.get("old_log_probs", student_logits)
        ref_log_probs = kwargs.get("ref_log_probs", teacher_logits)
        base_log_probs = kwargs.get("base_log_probs", old_log_probs)

        # G-OPD expects per-token log-probs [batch, seq_len], not full logits [batch, seq_len, vocab]
        # If 3D tensors are passed, reduce to 2D by taking max log-prob
        import torch.nn.functional as F
        if old_log_probs.dim() == 3:
            old_log_probs = F.log_softmax(old_log_probs, dim=-1).max(dim=-1).values
        if ref_log_probs.dim() == 3:
            ref_log_probs = F.log_softmax(ref_log_probs, dim=-1).max(dim=-1).values
        if base_log_probs.dim() == 3:
            base_log_probs = F.log_softmax(base_log_probs, dim=-1).max(dim=-1).values

        s_log_probs = student_logits
        if s_log_probs.dim() == 3:
            s_log_probs = F.log_softmax(s_log_probs, dim=-1).max(dim=-1).values

        lambda_vals = config.get("lambda_vals", 1.0) if isinstance(config, dict) else getattr(config, "lambda_vals", 1.0)

        advantages = compute_g_opd_advantages(
            old_log_probs=old_log_probs,
            ref_log_prob=ref_log_probs,
            base_log_prob=base_log_probs,
            lambda_vals=lambda_vals,
        )

        # Compute policy gradient loss with G-OPD advantages
        ratio = torch.exp(s_log_probs - old_log_probs)
        loss = -(ratio * advantages * mask).sum() / mask.sum().clamp(min=1)

        metrics = {
            "g_opd/mean_advantage": advantages[mask.bool()].mean().item() if mask.any() else 0.0,
            "g_opd/lambda_vals": lambda_vals,
        }

        return loss, metrics


class GOPDRewardHook:
    """RewardHook adapter for the G-OPD method.

    Handles the corrected reward computation that G-OPD uses
    for reward extrapolation.
    """

    def compute_reward(
        self,
        batch: Any,
        config: Config,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute G-OPD corrected rewards.

        G-OPD uses reward extrapolation to scale the reward signal
        based on teacher-student divergence.
        """
        # G-OPD's reward correction is integrated into the advantage computation
        # This hook provides the interface for external reward signals
        if isinstance(batch, dict):
            rewards = batch.get("rewards", torch.zeros(1))
        else:
            rewards = getattr(batch, "rewards", torch.zeros(1))

        return rewards


class GOPDTeacherSidecarHook:
    """TeacherSidecarHook adapter for the G-OPD method.

    Handles context distillation: the teacher receives additional context
    (e.g., correct solutions, critiques) in its prompt.
    """

    def teacher_forward(
        self,
        batch: Any,
        teacher_model: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Prepare and run teacher forward with context distillation inputs.

        G-OPD's teacher receives enriched inputs (correct solutions or
        critique-based prompts) to generate better supervision signals.
        """
        from easyopd.methods.g_opd.ref_input_utils import prepare_ref_model_inputs

        tokenizer = kwargs.get("tokenizer")
        ref_mode = config.get("ref_mode", "standard") if isinstance(config, dict) else getattr(config, "ref_mode", "standard")

        ref_inputs = prepare_ref_model_inputs(
            batch=batch,
            tokenizer=tokenizer,
            mode=ref_mode,
        )

        return {
            "type": "context_distillation",
            "ref_inputs": ref_inputs,
            "ref_mode": ref_mode,
        }
