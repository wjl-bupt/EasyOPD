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

"""Hook adapters for the `opcd` (On-Policy Context Distillation) method.

Wraps the existing OPCD core functions into the EasyOPD hook interfaces.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, LossHook, Metrics, RolloutHook


class OPCDLossHook:
    """LossHook adapter for the OPCD method.

    Wraps `compute_opcd_loss` (KL divergence between context-conditioned
    teacher and context-free student) into the standard LossHook interface.
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute OPCD KL loss.

        The teacher receives context (experience/system prompt) while the
        student generates without context. The loss minimizes KL divergence
        between their output distributions.

        Args:
            student_logits: Student log-probs (context-free).
            teacher_logits: Teacher log-probs (context-conditioned).
            mask: Response mask.
            config: OPCD config (kl_loss_type, kl_topk, etc.).
            **kwargs: Additional arguments.

        Returns:
            (loss, metrics) with KL divergence statistics.
        """
        from easyopd.methods.opcd.core import compute_opcd_loss

        kl_loss_type = config.get("kl_loss_type", "full") if isinstance(config, dict) else getattr(config, "kl_loss_type", "full")
        kl_topk = config.get("kl_topk", 0) if isinstance(config, dict) else getattr(config, "kl_topk", 0)

        loss, metrics_dict = compute_opcd_loss(
            student_log_probs=student_logits,
            teacher_log_probs=teacher_logits,
            response_mask=mask,
            kl_loss_type=kl_loss_type,
        )

        return loss, metrics_dict


class OPCDRolloutHook:
    """RolloutHook adapter for the OPCD method.

    Handles the consolidate stage: after rollout, injects experience
    (successful demonstrations) into the teacher's context for the
    next training iteration.
    """

    def on_rollout_end(
        self,
        batch: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Process batch after rollout for experience injection.

        Identifies successful trajectories and prepares experience prompts
        for the teacher's context in the consolidate stage.
        """
        from easyopd.methods.opcd.core import build_experience_prompt, truncate_experience

        max_experience_len = config.get("max_experience_len", 2048) if isinstance(config, dict) else getattr(config, "max_experience_len", 2048)
        tokenizer = kwargs.get("tokenizer")

        # Extract successful demonstrations from batch
        if isinstance(batch, dict):
            rewards = batch.get("rewards")
            responses = batch.get("responses")
        else:
            rewards = getattr(batch, "rewards", None)
            responses = getattr(batch, "responses", None)

        if rewards is not None and responses is not None and tokenizer is not None:
            # Build experience prompts from high-reward trajectories
            experience = build_experience_prompt(
                responses=responses,
                rewards=rewards,
                tokenizer=tokenizer,
            )
            experience = truncate_experience(experience, max_experience_len)

            if isinstance(batch, dict):
                batch["opcd_experience"] = experience
            else:
                batch.opcd_experience = experience

        return batch
