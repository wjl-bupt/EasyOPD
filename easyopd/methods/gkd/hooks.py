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

"""Hook adapters for the `gkd` (Generalized Knowledge Distillation) method.

Wraps the existing GKD core functions into the EasyOPD hook interfaces.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, LossHook, Metrics


class GKDLossHook:
    """LossHook adapter for the GKD method.

    Wraps `generalized_jsd` and `gkd_loss` into the standard LossHook interface.
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute Generalized JSD loss between student and teacher.

        Args:
            student_logits: Student log-probs [batch, seq_len, vocab].
            teacher_logits: Teacher log-probs [batch, seq_len, vocab].
            mask: Response mask [batch, seq_len].
            config: Must contain 'beta' (JSD interpolation parameter).
            **kwargs: Optional 'temperature' for scaling.

        Returns:
            (loss, metrics) where metrics includes per-token JSD stats.
        """
        from easyopd.methods.gkd.core import gkd_loss

        beta = config.get("beta", 0.5) if isinstance(config, dict) else getattr(config, "beta", 0.5)
        temperature = kwargs.get("temperature", 1.0)

        loss, metrics_dict = gkd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            response_mask=mask,
            beta=beta,
            temperature=temperature,
        )

        return loss, metrics_dict
