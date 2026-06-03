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

"""EasyOPD Hook Interfaces.

Defines the five core hook protocols that OPD methods can implement to inject
their logic into the verl training loop without modifying verl core code.

Hook Types:
    - LossHook: Token- or distribution-level supervision signal
    - RolloutHook: Trajectory metadata attachment after rollout
    - RewardHook: Judge or rubric feedback computation
    - AlignmentHook: Token- or span-level mapping between tokenizers
    - TeacherSidecarHook: Teacher-side signal generation (logits, hidden states)

Methods only need to implement the hooks they require. The HookDispatcher
(in hook_dispatch.py) handles routing and graceful fallback for unimplemented hooks.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Type aliases for clarity
# ---------------------------------------------------------------------------

# A generic batch is a dict-like object (typically verl's DataProto or dict)
Batch = Any
Config = Any
Metrics = dict[str, float]


# ---------------------------------------------------------------------------
# Loss Context Data Class
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional


@dataclass
class LossContext:
    """Unified context for OPD loss computation.
    
    This dataclass provides a standardized interface for all OPD methods,
    supporting logit-based, advantage-based, and reward-based approaches
    without losing any existing method's required information.
    """
    # Core tensors for logit-based methods (vopd, opsa, gkd, simple, simct)
    student_log_probs: Optional[torch.Tensor] = None
    teacher_log_probs: Optional[torch.Tensor] = None
    response_mask: Optional[torch.Tensor] = None
    
    # For advantage-based methods (sod)
    advantages: Optional[torch.Tensor] = None
    old_log_probs: Optional[torch.Tensor] = None
    
    # For reward-based methods (ropd, gad)
    rewards: Optional[torch.Tensor] = None
    
    # Additional context needed by various methods
    model_inputs: Optional[dict] = None
    actor_ref: Optional[Any] = None
    config: Optional[Config] = None
    
    # For compatibility with existing kwargs-based interface
    extra_kwargs: Optional[dict] = None
    
    def __post_init__(self):
        """Ensure backward compatibility with existing kwargs usage."""
        if self.extra_kwargs is None:
            self.extra_kwargs = {}
        
        # Auto-populate extra_kwargs with standard fields for compatibility
        if self.advantages is not None and "advantages" not in self.extra_kwargs:
            self.extra_kwargs["advantages"] = self.advantages
        if self.old_log_probs is not None and "old_log_probs" not in self.extra_kwargs:
            self.extra_kwargs["old_log_probs"] = self.old_log_probs
        if self.model_inputs is not None and "model_output" not in self.extra_kwargs:
            self.extra_kwargs["model_output"] = self.model_inputs
        if self.response_mask is not None and "response_mask" not in self.extra_kwargs:
            self.extra_kwargs["response_mask"] = self.response_mask

    def to_kwargs(self) -> dict:
        """Convert to kwargs format for backward compatibility."""
        kwargs = {}
        if self.student_log_probs is not None:
            kwargs["student_logits"] = self.student_log_probs
        if self.teacher_log_probs is not None:
            kwargs["teacher_logits"] = self.teacher_log_probs
        if self.response_mask is not None:
            kwargs["mask"] = self.response_mask
        if self.config is not None:
            kwargs["config"] = self.config
        
        # Include all extra_kwargs
        kwargs.update(self.extra_kwargs)
        return kwargs


# ---------------------------------------------------------------------------
# Hook Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class LossHook(Protocol):
    """Hook for computing distillation/supervision loss.

    This is the most commonly implemented hook. It receives student and teacher
    logits (or log-probs) along with a mask, and returns the loss tensor plus
    optional diagnostic metrics.
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Compute the distillation loss.

        Args:
            student_logits: Student model output logits or log-probs.
                Shape depends on method (e.g. [batch, seq_len, vocab] or
                [batch, seq_len, overlap_vocab]).
            teacher_logits: Teacher model output logits or log-probs.
                Same shape convention as student_logits.
            mask: Boolean or float mask indicating valid positions.
                Shape: [batch, seq_len].
            config: Method-specific configuration dict or object.
            **kwargs: Additional method-specific arguments (e.g. alignment_map,
                      advantages, old_log_probs).

        Returns:
            A tuple of (loss_tensor, metrics_dict) where:
                - loss_tensor: Scalar loss to be added to the training objective.
                - metrics_dict: Dictionary of diagnostic metrics (e.g.
                  {"kl_div": 0.5, "alignment_coverage": 0.92}).
        """
        ...


@runtime_checkable
class RolloutHook(Protocol):
    """Hook for post-rollout processing.

    Attaches trajectory metadata (step boundaries, tool calls, observations)
    to the training batch after rollout generation completes.
    """

    def on_rollout_end(
        self,
        batch: Batch,
        config: Config,
        **kwargs: Any,
    ) -> Batch:
        """Process batch after rollout generation.

        Args:
            batch: The training batch containing rollout results.
            config: Method-specific configuration.
            **kwargs: Additional context (e.g. tokenizer, step_info).

        Returns:
            The (possibly modified) batch with additional metadata attached.
        """
        ...


@runtime_checkable
class RewardHook(Protocol):
    """Hook for computing method-specific rewards.

    Used by methods that need custom reward signals beyond the standard
    reward model (e.g. judge feedback, rubric scoring, corrected rewards).
    """

    def compute_reward(
        self,
        batch: Batch,
        config: Config,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute method-specific rewards.

        Args:
            batch: The training batch containing generated responses.
            config: Method-specific configuration.
            **kwargs: Additional context (e.g. teacher_outputs, environment_feedback).

        Returns:
            Reward tensor of shape [batch_size] or [batch_size, seq_len].
        """
        ...


@runtime_checkable
class AlignmentHook(Protocol):
    """Hook for building token/span alignment between tokenizers.

    Used by cross-tokenizer methods that need to map between student and
    teacher token spaces.
    """

    def build_alignment(
        self,
        student_tokenizer: Any,
        teacher_tokenizer: Any,
        input_ids: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Build alignment map between student and teacher tokenizers.

        Args:
            student_tokenizer: The student model's tokenizer.
            teacher_tokenizer: The teacher model's tokenizer.
            input_ids: Input token IDs to align.
            config: Method-specific configuration.
            **kwargs: Additional context.

        Returns:
            An alignment map object (format depends on method, e.g. overlap
            vocabulary indices, span mappings, position correspondences).
        """
        ...


@runtime_checkable
class TeacherSidecarHook(Protocol):
    """Hook for teacher-side signal generation.

    Handles teacher model forward passes and signal extraction (logits,
    hidden states, judge outputs) that are needed by the loss computation.
    """

    def teacher_forward(
        self,
        batch: Batch,
        teacher_model: Any,
        config: Config,
        **kwargs: Any,
    ) -> Any:
        """Run teacher model forward and extract signals.

        Args:
            batch: The training batch (may need preprocessing for teacher input).
            teacher_model: The teacher model instance.
            config: Method-specific configuration.
            **kwargs: Additional context (e.g. teacher_tokenizer, alignment_map).

        Returns:
            Teacher outputs object containing the signals needed by LossHook
            (e.g. logits, hidden states, top-k log-probs).
        """
        ...


# ---------------------------------------------------------------------------
# Composite hook container
# ---------------------------------------------------------------------------


class MethodHooks:
    """Container for a method's hook implementations.

    A method registers its hook implementations here. Only the hooks that the
    method actually implements need to be provided; others remain None.
    """

    def __init__(
        self,
        loss_hook: Optional[LossHook] = None,
        rollout_hook: Optional[RolloutHook] = None,
        reward_hook: Optional[RewardHook] = None,
        alignment_hook: Optional[AlignmentHook] = None,
        teacher_sidecar_hook: Optional[TeacherSidecarHook] = None,
    ) -> None:
        self.loss_hook = loss_hook
        self.rollout_hook = rollout_hook
        self.reward_hook = reward_hook
        self.alignment_hook = alignment_hook
        self.teacher_sidecar_hook = teacher_sidecar_hook

    @property
    def has_loss(self) -> bool:
        return self.loss_hook is not None

    @property
    def has_rollout(self) -> bool:
        return self.rollout_hook is not None

    @property
    def has_reward(self) -> bool:
        return self.reward_hook is not None

    @property
    def has_alignment(self) -> bool:
        return self.alignment_hook is not None

    @property
    def has_teacher_sidecar(self) -> bool:
        return self.teacher_sidecar_hook is not None

    def active_hooks(self) -> list[str]:
        """Return names of all active (non-None) hooks."""
        hooks = []
        if self.has_loss:
            hooks.append("loss")
        if self.has_rollout:
            hooks.append("rollout")
        if self.has_reward:
            hooks.append("reward")
        if self.has_alignment:
            hooks.append("alignment")
        if self.has_teacher_sidecar:
            hooks.append("teacher_sidecar")
        return hooks

    def __repr__(self) -> str:
        return f"MethodHooks(active={self.active_hooks()})"
