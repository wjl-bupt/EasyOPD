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

"""Hook adapters for the `sod` (Step-wise On-policy Distillation) method.

Wraps the existing SOD core functions into the EasyOPD hook interfaces.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, LossHook, Metrics, RolloutHook, LossContext


class SODLossHook:
    """LossHook adapter for the SOD method.

    Wraps `apply_stepwise_opd` into the standard LossHook interface.
    The SOD method modifies the advantage weights rather than computing
    a separate loss, but we expose it through LossHook for uniformity.
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute step-wise weighted OPD loss.

        SOD applies adaptive per-step weights to the standard OPD signal.
        The actual loss computation uses the weighted advantages.

        Args:
            student_logits: Student log-probs [batch, seq_len].
            teacher_logits: Teacher/ref log-probs [batch, seq_len].
            mask: Response mask [batch, seq_len].
            config: Must contain 'epsilon' and 'delta' for weight computation.
            **kwargs: Must contain 'advantages' tensor.

        Returns:
            (weighted_loss, metrics) with step-wise weight statistics.
        """
        from easyopd.methods.sod.core import apply_stepwise_opd, compute_stepwise_opd_weights

        epsilon = config.get("epsilon", 1e-6) if isinstance(config, dict) else getattr(config, "epsilon", 1e-6)
        delta = config.get("delta", 0.5) if isinstance(config, dict) else getattr(config, "delta", 0.5)
        advantages = kwargs.get("advantages")

        # SOD expects per-token log-probs [batch, seq_len], not full logits [batch, seq_len, vocab]
        # If 3D tensors are passed, reduce to 2D by taking max log-prob
        s_logits = student_logits
        t_logits = teacher_logits
        if s_logits.dim() == 3:
            import torch.nn.functional as F
            s_logits = F.log_softmax(s_logits, dim=-1).max(dim=-1).values
        if t_logits.dim() == 3:
            import torch.nn.functional as F
            t_logits = F.log_softmax(t_logits, dim=-1).max(dim=-1).values

        if advantages is None:
            # If no advantages provided, just compute weights for diagnostics
            weight_mask, step_info = compute_stepwise_opd_weights(
                old_log_probs=s_logits,
                ref_log_prob=t_logits,
                response_mask=mask,
                epsilon=epsilon,
                delta=delta,
            )
            # Return zero loss with weight diagnostics
            metrics = {
                "sod/mean_weight": weight_mask[mask.bool()].mean().item() if mask.any() else 0.0,
                "sod/num_steps": sum(len(info.get("steps", [])) for info in step_info) / max(len(step_info), 1),
            }
            return torch.tensor(0.0, device=student_logits.device), metrics

        # Apply step-wise OPD to advantages
        weighted_advantages, step_info = apply_stepwise_opd(
            advantages=advantages,
            old_log_probs=s_logits,
            ref_log_prob=t_logits,
            response_mask=mask,
            epsilon=epsilon,
            delta=delta,
        )

        metrics = {
            "sod/mean_weight": (weighted_advantages / (advantages + 1e-8))[mask.bool()].mean().item()
            if mask.any() and advantages.abs().sum() > 0
            else 1.0,
        }

        # SOD modifies advantages rather than producing a separate loss
        # Return the weighted advantages as "loss" (to be used by the caller)
        return weighted_advantages.sum() / mask.sum().clamp(min=1), metrics

    def compute_loss_with_context(
        self,
        context: LossContext,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute loss using the unified LossContext interface.
        
        This is the new preferred interface that supports all OPD method types.
        """
        from easyopd.methods.sod.core import apply_stepwise_opd, compute_stepwise_opd_weights

        # Extract parameters from LossContext
        advantages = context.advantages
        old_log_probs = context.old_log_probs
        response_mask = context.response_mask
        config = context.config

        epsilon = config.get("epsilon", 1e-6) if isinstance(config, dict) else getattr(config, "epsilon", 1e-6)
        delta = config.get("delta", 0.5) if isinstance(config, dict) else getattr(config, "delta", 0.5)

        if advantages is None:
            # If no advantages provided, just compute weights for diagnostics
            weight_mask, step_info = compute_stepwise_opd_weights(
                old_log_probs=old_log_probs,
                ref_log_prob=context.teacher_log_probs,
                response_mask=response_mask,
                epsilon=epsilon,
                delta=delta,
            )
            # Return zero loss with weight diagnostics
            metrics = {
                "sod/mean_weight": weight_mask[response_mask.bool()].mean().item() if response_mask.any() else 0.0,
                "sod/num_steps": sum(len(info.get("steps", [])) for info in step_info) / max(len(step_info), 1),
            }
            return torch.tensor(0.0, device=old_log_probs.device), metrics

        # Apply step-wise OPD to advantages
        weighted_advantages, step_info = apply_stepwise_opd(
            advantages=advantages,
            old_log_probs=old_log_probs,
            ref_log_prob=context.teacher_log_probs,
            response_mask=response_mask,
            epsilon=epsilon,
            delta=delta,
        )

        metrics = {
            "sod/mean_weight": (weighted_advantages / (advantages + 1e-8))[response_mask.bool()].mean().item()
            if response_mask.any() and advantages.abs().sum() > 0
            else 1.0,
        }

        # SOD modifies advantages rather than producing a separate loss
        # Return the weighted advantages as "loss" (to be used by the caller)
        return weighted_advantages.sum() / response_mask.sum().clamp(min=1), metrics


class SODRolloutHook:
    """RolloutHook adapter for the SOD method.

    Attaches step boundary metadata to the batch after rollout,
    which is needed for the step-wise weight computation.
    """

    def on_rollout_end(
        self,
        batch: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Attach step boundary information to the batch.

        Identifies assistant turn boundaries from the response mask
        and stores them in the batch for later use by the LossHook.
        """
        from easyopd.methods.sod.core import _extract_step_boundaries

        # Extract response_mask from batch
        if isinstance(batch, dict):
            response_mask = batch.get("response_mask")
        else:
            response_mask = getattr(batch, "response_mask", None)

        if response_mask is not None:
            step_boundaries = _extract_step_boundaries(response_mask)
            if isinstance(batch, dict):
                batch["sod_step_boundaries"] = step_boundaries
            else:
                batch.sod_step_boundaries = step_boundaries

        return batch
