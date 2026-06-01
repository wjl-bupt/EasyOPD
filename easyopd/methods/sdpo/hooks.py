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

"""Hook adapters for the `sdpo` (Self-Distilled Policy Optimization) method.

Wraps the existing SDPO core functions into the EasyOPD hook interfaces.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, LossHook, Metrics, TeacherSidecarHook


class SDPOLossHook:
    """LossHook adapter for the SDPO method.

    Wraps `compute_sdpo_self_distillation_loss` into the standard LossHook
    interface. SDPO uses self-distillation from the model's own high-reward
    trajectories (no external teacher needed).
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute SDPO self-distillation loss.

        Args:
            student_logits: Current policy log-probs.
            teacher_logits: EMA/reprompted teacher log-probs.
            mask: Response mask.
            config: SDPO config (kl_coef, temperature, etc.).
            **kwargs: Optional 'demonstrations', 'rewards'.

        Returns:
            (loss, metrics) with self-distillation loss statistics.
        """
        from easyopd.methods.sdpo.core import compute_sdpo_self_distillation_loss

        kl_coef = config.get("kl_coef", 1.0) if isinstance(config, dict) else getattr(config, "kl_coef", 1.0)
        temperature = config.get("temperature", 1.0) if isinstance(config, dict) else getattr(config, "temperature", 1.0)

        loss, metrics_dict = compute_sdpo_self_distillation_loss(
            student_log_probs=student_logits,
            teacher_log_probs=teacher_logits,
            mask=mask,
            kl_coef=kl_coef,
            temperature=temperature,
        )

        return loss, metrics_dict


class SDPOTeacherSidecarHook:
    """TeacherSidecarHook adapter for the SDPO method.

    Handles the self-distillation teacher: either an EMA copy of the
    student or a reprompted version using successful demonstrations.
    """

    def teacher_forward(
        self,
        batch: Any,
        teacher_model: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Prepare self-distillation teacher signals.

        SDPO's "teacher" is the model itself, either:
        1. An EMA copy of the student weights, or
        2. The student reprompted with successful demonstrations.
        """
        from easyopd.methods.sdpo.core import (
            build_reprompt_text,
            select_demonstration,
            compute_ema_update,
        )

        teacher_mode = config.get("teacher_mode", "ema") if isinstance(config, dict) else getattr(config, "teacher_mode", "ema")
        ema_decay = config.get("ema_decay", 0.999) if isinstance(config, dict) else getattr(config, "ema_decay", 0.999)
        tokenizer = kwargs.get("tokenizer")

        if teacher_mode == "reprompt" and tokenizer is not None:
            # Select best demonstration and build reprompt
            demonstration = select_demonstration(batch)
            if demonstration is not None:
                reprompt_text = build_reprompt_text(
                    demonstration=demonstration,
                    tokenizer=tokenizer,
                )
                return {
                    "type": "sdpo_reprompt",
                    "reprompt_text": reprompt_text,
                    "teacher_mode": "reprompt",
                }

        # Default: EMA teacher
        return {
            "type": "sdpo_ema",
            "ema_decay": ema_decay,
            "teacher_mode": "ema",
        }
