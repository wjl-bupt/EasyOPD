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

"""Hook adapters for the `sdpo` (Self-Distillation Policy Optimization) method.

Wraps the SDPO core functions into the EasyOPD hook interfaces:
    * ``SDPOLossHook``           -> ``LossHook``  (self-distillation loss)
    * ``SDPOTeacherSidecarHook`` -> ``TeacherSidecarHook`` (reprompt teacher batch)
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, Metrics


class SDPOLossHook:
    """LossHook adapter for SDPO.

    Computes the token-level self-distillation loss between the student
    (``pi(.|x)``) and the feedback-informed self-teacher (``pi(.|x,f)``).
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute the SDPO self-distillation loss.

        Args:
            student_logits: ``[B, T]`` student per-token log-probs (grad).
            teacher_logits: ``[B, T]`` self-teacher per-token log-probs.
            mask: ``[B, T]`` response mask.
            config: SDPO ``self_distillation`` config (``alpha``, ``is_clip``...).
            **kwargs: optional ``old_log_probs``, ``self_distillation_mask``,
                ``loss_agg_mode``.

        Returns:
            ``(loss, metrics)``.
        """
        from easyopd.methods.sdpo.core import compute_sdpo_loss

        def _get(key, default):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        return compute_sdpo_loss(
            student_log_probs=student_logits,
            teacher_log_probs=teacher_logits,
            response_mask=mask,
            alpha=_get("alpha", 0.5),
            is_clip=_get("is_clip", None),
            old_log_probs=kwargs.get("old_log_probs"),
            self_distillation_mask=kwargs.get("self_distillation_mask"),
            loss_agg_mode=kwargs.get("loss_agg_mode", "token-mean"),
        )


class SDPOTeacherSidecarHook:
    """TeacherSidecarHook adapter for SDPO.

    Builds the self-teacher batch by reprompting failed rollouts with a correct
    demonstration (and/or environment feedback) from the same rollout group,
    then appending the original response for re-scoring. Returns the tensors
    needed for the teacher forward pass.
    """

    def teacher_forward(
        self,
        batch: Any,
        teacher_model: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Construct the SDPO reprompt teacher batch.

        Args:
            batch: verl ``DataProto`` after rollout + reward.
            teacher_model: unused — SDPO uses the live policy as self-teacher.
            config: SDPO ``self_distillation`` config (dict-like).
            **kwargs: must provide ``reward_tensor`` and ``tokenizer``; optional
                ``apply_chat_template_kwargs``.

        Returns:
            ``(tensors, metrics)`` from ``build_sdpo_teacher_inputs`` or ``None``
            when required inputs are unavailable.
        """
        from easyopd.methods.sdpo.core import build_sdpo_teacher_inputs

        reward_tensor = kwargs.get("reward_tensor")
        tokenizer = kwargs.get("tokenizer")
        if reward_tensor is None or tokenizer is None:
            return None

        return build_sdpo_teacher_inputs(
            batch=batch,
            reward_tensor=reward_tensor,
            cfg=config,
            tokenizer=tokenizer,
            apply_chat_template_kwargs=kwargs.get("apply_chat_template_kwargs"),
        )
