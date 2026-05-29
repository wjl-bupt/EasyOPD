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

from easyopd.hooks import Config, LossHook, Metrics


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

        Delegates to the existing simct loss implementation which uses
        span virtual-vocabulary logits on top of the shared overlap vocabulary.
        """
        from easyopd.methods.simct.losses import compute_simct_loss

        model_output = kwargs.get("model_output", {})
        response_mask = kwargs.get("response_mask", mask)

        loss, metrics_dict = compute_simct_loss(
            model_output=model_output,
            response_mask=response_mask,
            config=config,
        )

        return loss, metrics_dict
