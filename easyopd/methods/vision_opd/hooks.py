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

"""Hook adapters for the `vision_opd` (Vision On-Policy Self-Distillation) method.

Wraps the existing Vision-OPD core functions into the EasyOPD hook interfaces.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, LossHook, Metrics, TeacherSidecarHook


class VisionOPDLossHook:
    """LossHook adapter for the Vision-OPD method.

    Wraps `compute_self_distillation_loss` into the standard LossHook interface.
    Vision-OPD uses an EMA teacher that receives fine-grained visual inputs
    to guide the student model.
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute Vision-OPD self-distillation loss.

        Args:
            student_logits: Student log-probs or logits.
            teacher_logits: EMA teacher log-probs (with fine-grained visual input).
            mask: Response mask.
            config: Vision-OPD config (topk, temperature, etc.).
            **kwargs: Optional 'teacher_top_k_log_probs', 'teacher_top_k_ids'.

        Returns:
            (loss, metrics) with distillation loss statistics.
        """
        from easyopd.methods.vision_opd.core import compute_self_distillation_loss

        alpha = config.get("alpha", 0.5) if isinstance(config, dict) else getattr(config, "alpha", 0.5)
        full_logit_distillation = config.get("full_logit_distillation", True) if isinstance(config, dict) else getattr(config, "full_logit_distillation", True)

        # full_logit_distillation requires student_all_log_probs and teacher_all_log_probs
        # which are only available in the full pipeline. Fall back to non-full mode.
        student_all_log_probs = kwargs.get("student_all_log_probs")
        teacher_all_log_probs = kwargs.get("teacher_all_log_probs")
        if full_logit_distillation and student_all_log_probs is None:
            full_logit_distillation = False
            alpha = 1.0  # non-full-logit mode requires reverse KL (alpha=1.0)

        # Vision-OPD expects per-token log-probs [batch, seq_len], not full logits
        import torch.nn.functional as F
        s_log_probs = student_logits
        t_log_probs = teacher_logits
        if s_log_probs.dim() == 3:
            s_log_probs = F.log_softmax(s_log_probs, dim=-1).max(dim=-1).values
        if t_log_probs.dim() == 3:
            t_log_probs = F.log_softmax(t_log_probs, dim=-1).max(dim=-1).values

        loss, metrics_dict = compute_self_distillation_loss(
            student_log_probs=s_log_probs,
            teacher_log_probs=t_log_probs,
            response_mask=mask,
            alpha=alpha,
            full_logit_distillation=full_logit_distillation,
            is_clip=None,
            student_all_log_probs=student_all_log_probs,
            teacher_all_log_probs=teacher_all_log_probs,
        )

        return loss, metrics_dict


class VisionOPDTeacherSidecarHook:
    """TeacherSidecarHook adapter for the Vision-OPD method.

    Handles the EMA teacher forward pass with fine-grained visual inputs
    (bounding-box cropped images).
    """

    def teacher_forward(
        self,
        batch: Any,
        teacher_model: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Run EMA teacher forward with fine-grained visual inputs.

        The Vision-OPD teacher receives bounding-box cropped images
        to provide more detailed visual supervision signals.
        """
        from easyopd.methods.vision_opd.teacher_utils import (
            prepare_teacher_messages_with_bbox_images,
            teacher_images_available,
        )
        from easyopd.methods.vision_opd.core import ema_update_teacher

        tokenizer = kwargs.get("tokenizer")
        processor = kwargs.get("processor")
        ema_decay = config.get("ema_decay", 0.999) if isinstance(config, dict) else getattr(config, "ema_decay", 0.999)

        # Check if teacher images are available
        if not teacher_images_available(batch):
            return None

        # Prepare teacher inputs with bbox-cropped images
        teacher_messages = prepare_teacher_messages_with_bbox_images(
            batch=batch,
            processor=processor,
        )

        return {
            "type": "vision_opd_ema_teacher",
            "teacher_messages": teacher_messages,
            "ema_decay": ema_decay,
        }
