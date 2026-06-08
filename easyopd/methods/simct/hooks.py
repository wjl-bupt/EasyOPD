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

"""Hook adapters for the `simct` (span-based cross-tokenizer KD) method.

Wraps the existing simct method functions into the EasyOPD hook interfaces.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, LossHook, Metrics, LossContext


class SimCTLossHook:
    """LossHook adapter for the SimCT span-based cross-tokenizer KD method.

    Wraps the simct loss functions into the standard LossHook interface.
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute span-based cross-tokenizer KL divergence loss.

        Delegates to the existing simct loss implementation when running
        inside the full verl pipeline (model_output contains distillation_losses).
        Falls back to a direct KL computation for standalone/benchmark usage.
        """
        model_output = kwargs.get("model_output", {})

        # If distillation_losses are pre-computed by the logit processor (full pipeline),
        # delegate to the full implementation
        if "distillation_losses" in model_output:
            from easyopd.methods.simct.losses import compute_distillation_loss_simct_cross_tokenizer
            distillation_config = kwargs.get("distillation_config", config)
            data = kwargs.get("data")
            loss, metrics_dict = compute_distillation_loss_simct_cross_tokenizer(
                config=config,
                distillation_config=distillation_config,
                model_output=model_output,
                data=data,
            )
            return loss, metrics_dict

        # Standalone fallback: compute simple forward KL between student and teacher
        import torch.nn.functional as F
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        loss = (kl * mask).sum() / mask.sum().clamp(min=1)
        metrics = {"simct/kl_div": loss.detach().item()}
        return loss, metrics

    def compute_loss_with_context(
        self,
        context: LossContext,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute loss using the unified LossContext interface.

        This is the new preferred interface that supports all OPD method types.
        """
        model_output = context.model_inputs or {}
        config = context.config

        if "distillation_losses" in model_output:
            from easyopd.methods.simct.losses import compute_distillation_loss_simct_cross_tokenizer
            distillation_config = (context.extra_kwargs or {}).get("distillation_config", config)
            data = (context.extra_kwargs or {}).get("data")
            loss, metrics_dict = compute_distillation_loss_simct_cross_tokenizer(
                config=config,
                distillation_config=distillation_config,
                model_output=model_output,
                data=data,
            )
            return loss, metrics_dict

        # Standalone fallback
        import torch.nn.functional as F
        student_logits = context.student_log_probs
        teacher_logits = context.teacher_log_probs
        mask = context.response_mask
        if student_logits is not None and teacher_logits is not None and mask is not None:
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            loss = (kl * mask).sum() / mask.sum().clamp(min=1)
            return loss, {"simct/kl_div": loss.detach().item()}

        return torch.tensor(0.0), {}
