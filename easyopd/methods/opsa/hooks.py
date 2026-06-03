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

"""Hook adapters for the `opsa` (On-Policy Self-Distillation for Safety Alignment) method.

Wraps the existing OPSA core functions into the EasyOPD hook interfaces so that
the unified HookDispatcher can route loss computation and teacher signal
generation to OPSA without modifying verl core code.

Paper: "Reducing the Safety Tax in LLM Safety Alignment with On-Policy
        Self-Distillation" (Fu et al., arXiv 2605.15239)
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Config, LossHook, Metrics, TeacherSidecarHook, LossContext


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    """Safely fetch a value from either a dict-like or attribute-like config."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class OPSALossHook:
    """LossHook adapter for the OPSA method.

    Wraps :func:`easyopd.methods.opsa.core.opsa_loss` (the full OPSA loss with
    temperature scaling, early-window weighting, and forward KL D_KL(p_T || p_S))
    into the standard LossHook interface.

    Note:
        OPSA expects ``student_logits`` / ``teacher_logits`` to be raw vocab-sized
        logits. The hook will pass them through ``opsa_loss`` which internally
        applies temperature scaling and ``log_softmax``. Methods that already
        provide log-probs at full vocab can pass them in unchanged (the
        temperature-scaled log-softmax is idempotent up to a constant).
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute the full OPSA window-weighted KL loss.

        Args:
            student_logits: Student model logits over the response,
                shape ``(batch, seq_len, vocab)``.
            teacher_logits: Teacher (frozen base + privileged context) logits,
                shape ``(batch, seq_len, vocab)``.
            mask: Response mask, shape ``(batch, seq_len)``.
            config: OPSA-specific config (supports either dict or dataclass).
            **kwargs: Reserved for future use.

        Returns:
            Tuple of ``(scalar_loss, metrics_dict)``.
        """
        from easyopd.methods.opsa.core import opsa_loss

        temperature = float(_cfg_get(config, "opsa_temperature", _cfg_get(config, "temperature", 1.0)))
        window_size = int(_cfg_get(config, "opsa_window_size", _cfg_get(config, "window_size", 32)))
        decay_type = str(_cfg_get(config, "opsa_decay_type", _cfg_get(config, "decay_type", "linear")))
        min_weight = float(_cfg_get(config, "opsa_min_weight", _cfg_get(config, "min_weight", 0.1)))
        use_window_weighting = bool(_cfg_get(config, "opsa_use_window_weighting", _cfg_get(config, "use_window_weighting", True)))
        loss_agg_mode = str(_cfg_get(config, "opsa_loss_agg_mode", _cfg_get(config, "loss_agg_mode", "token-mean")))

        loss, metrics_dict = opsa_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            response_mask=mask,
            temperature=temperature,
            window_size=window_size,
            decay_type=decay_type,
            min_weight=min_weight,
            use_window_weighting=use_window_weighting,
            loss_agg_mode=loss_agg_mode,
        )

        # Ensure metrics_dict only contains float-castable scalars (LossHook contract).
        clean_metrics: Metrics = {}
        for k, v in (metrics_dict or {}).items():
            try:
                clean_metrics[k] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            except (TypeError, ValueError):
                continue

        return loss, clean_metrics

    def compute_loss_with_context(
        self,
        context: LossContext,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute loss using the unified LossContext interface.
        
        This is the new preferred interface that supports all OPD method types.
        """
        from easyopd.methods.opsa.core import opsa_loss

        # Extract parameters from LossContext
        student_logits = context.student_log_probs
        teacher_logits = context.teacher_log_probs
        response_mask = context.response_mask
        config = context.config

        temperature = float(_cfg_get(config, "opsa_temperature", _cfg_get(config, "temperature", 1.0)))
        window_size = int(_cfg_get(config, "opsa_window_size", _cfg_get(config, "window_size", 32)))
        decay_type = str(_cfg_get(config, "opsa_decay_type", _cfg_get(config, "decay_type", "linear")))
        min_weight = float(_cfg_get(config, "opsa_min_weight", _cfg_get(config, "min_weight", 0.1)))
        use_window_weighting = bool(_cfg_get(config, "opsa_use_window_weighting", _cfg_get(config, "use_window_weighting", True)))
        loss_agg_mode = str(_cfg_get(config, "opsa_loss_agg_mode", _cfg_get(config, "loss_agg_mode", "token-mean")))

        loss, metrics_dict = opsa_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            response_mask=response_mask,
            temperature=temperature,
            window_size=window_size,
            decay_type=decay_type,
            min_weight=min_weight,
            use_window_weighting=use_window_weighting,
            loss_agg_mode=loss_agg_mode,
        )

        # Ensure metrics_dict only contains float-castable scalars (LossHook contract).
        clean_metrics: Metrics = {}
        for k, v in (metrics_dict or {}).items():
            try:
                clean_metrics[k] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            except (TypeError, ValueError):
                continue

        return loss, clean_metrics


class OPSATeacherSidecarHook:
    """TeacherSidecarHook adapter for the OPSA method.

    Produces the metadata describing how to build the teacher's input sequence
    from the student's prompt: prepend a *type-conditional privileged context*
    (paper Section 3.2):

      * ``opsa_harmful_context`` — used for harmful queries (forces refusal).
      * ``opsa_benign_context`` — used for benign queries (avoids over-refusal).

    The actual teacher forward pass is performed by verl's existing
    ``RefPolicyWorker`` after the trainer rewrites the input ids; this hook
    therefore only returns the configuration payload that the trainer's
    ``_maybe_build_opsa_batch`` consumes.
    """

    DEFAULT_HARMFUL_CONTEXT = (
        "You are a helpful and safe AI assistant. When you encounter a harmful, "
        "dangerous, or unethical request, you MUST refuse clearly and explain why. "
        "Safety is your highest priority."
    )
    DEFAULT_BENIGN_CONTEXT = (
        "You are a helpful AI assistant. Respond to the user's request directly "
        "and helpfully. Do not over-refuse safe and reasonable requests."
    )

    def teacher_forward(
        self,
        batch: Any,
        teacher_model: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Return the privileged-context metadata for the trainer.

        Args:
            batch: The training batch (unused at this layer; the trainer is
                responsible for rewriting prompt token ids).
            teacher_model: The frozen reference policy (unused at this layer).
            config: OPSA config containing ``opsa_harmful_context`` /
                ``opsa_benign_context`` (or their default fallbacks).
            **kwargs: Reserved for future use.

        Returns:
            A plain dict with the privileged-context payload:
                ``{"type": "opsa_privileged_context",
                   "harmful_context": str,
                   "benign_context": str,
                   "topk_logits_k": int}``
        """
        harmful = _cfg_get(config, "opsa_harmful_context", self.DEFAULT_HARMFUL_CONTEXT)
        benign = _cfg_get(config, "opsa_benign_context", self.DEFAULT_BENIGN_CONTEXT)
        topk_logits_k = int(_cfg_get(config, "opsa_topk_logits_k", 512))
        kl_type = str(_cfg_get(config, "opsa_kl_type", "mixed"))
        mixed_kl_weight = float(_cfg_get(config, "opsa_mixed_kl_weight", 0.5))

        return {
            "type": "opsa_privileged_context",
            "harmful_context": str(harmful),
            "benign_context": str(benign),
            "topk_logits_k": topk_logits_k,
            "kl_type": kl_type,
            "mixed_kl_weight": mixed_kl_weight,
        }
