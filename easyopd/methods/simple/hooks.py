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

"""Hook adapters for the `simple` (cross-tokenizer KD) method.

Wraps the existing simple method functions into the EasyOPD hook interfaces.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import AlignmentHook, Config, LossHook, Metrics, TeacherSidecarHook, LossContext


class SimpleLossHook:
    """LossHook adapter for the simple cross-tokenizer KD method.

    Wraps `compute_simple_xtok_logits_processor` and
    `compute_distillation_loss_simple_cross_tokenizer` into the standard
    LossHook interface.
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute cross-tokenizer KL divergence loss.

        Delegates to the existing simple loss implementation when running
        inside the full verl pipeline (model_output contains distillation_losses).
        Falls back to a direct KL computation for standalone/benchmark usage.
        """
        # The simple method's loss function expects specific kwargs
        model_output = kwargs.get("model_output", {})

        # If distillation_losses are pre-computed by the logit processor (full pipeline),
        # delegate to the full implementation
        if "distillation_losses" in model_output:
            from easyopd.methods.simple.losses import (
                compute_distillation_loss_simple_cross_tokenizer,
            )
            distillation_config = kwargs.get("distillation_config", config)
            data = kwargs.get("data")
            loss, metrics_dict = compute_distillation_loss_simple_cross_tokenizer(
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
        metrics = {"simple/kl_div": loss.detach().item()}
        return loss, metrics

    def compute_loss_with_context(
        self,
        context: LossContext,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute loss using the unified LossContext interface.

        This is the new preferred interface that supports all OPD method types.
        """
        # Extract parameters from LossContext
        model_output = context.model_inputs or {}
        config = context.config

        if "distillation_losses" in model_output:
            from easyopd.methods.simple.losses import (
                compute_distillation_loss_simple_cross_tokenizer,
            )
            distillation_config = (context.extra_kwargs or {}).get("distillation_config", config)
            data = (context.extra_kwargs or {}).get("data")
            loss, metrics_dict = compute_distillation_loss_simple_cross_tokenizer(
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
            return loss, {"simple/kl_div": loss.detach().item()}

        return torch.tensor(0.0), {}


class SimpleAlignmentHook:
    """AlignmentHook adapter for the simple method.

    Wraps `find_overlap_tokens` and `align_sequences` into the standard
    AlignmentHook interface.
    """

    def build_alignment(
        self,
        student_tokenizer: Any,
        teacher_tokenizer: Any,
        input_ids: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Build overlap vocabulary alignment between student and teacher.

        Returns:
            Dict with 'student_overlap_ids' and 'teacher_overlap_ids' tensors.
        """
        from easyopd.methods.simple.alignment import find_overlap_tokens

        student_ids, teacher_ids = find_overlap_tokens(
            student_tokenizer, teacher_tokenizer
        )

        return {
            "student_overlap_ids": student_ids,
            "teacher_overlap_ids": teacher_ids,
        }


class SimpleTeacherSidecarHook:
    """TeacherSidecarHook adapter for the simple method.

    Wraps the teacher sidecar forward pass (HF-based teacher with
    cross-tokenizer logit extraction).
    """

    def teacher_forward(
        self,
        batch: Any,
        teacher_model: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Run teacher forward and extract logits on overlap vocabulary.

        The simple method uses a dedicated teacher sidecar that runs
        independently. This hook provides the interface for integration.
        """
        # The simple method's teacher forward is handled by the TeacherActorGroup
        # which runs as a separate Ray worker. This hook provides metadata
        # about what the teacher sidecar produces.
        return {
            "type": "cross_tokenizer_sidecar",
            "produces": ["teacher_hidden_states", "teacher_input_ids", "teacher_loss_mask"],
        }
