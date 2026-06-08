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

"""Hook adapters for the `ropd` (Rubric-based On-Policy Distillation) method.

ROPD is a *black-box* OPD method: it does **not** modify the actor loss
(verl's standard PG loss is reused), and it does **not** require teacher
sidecar logits. Its single point of contact with the training loop is a
custom reward manager (registered via verl's reward-manager registry) that
scores rollouts against a teacher-grounded rubric using a
``teacher + rubricator + verifier`` judge triple.

The two adapters in this module exist purely to satisfy EasyOPD's unified
``HookDispatcher`` contract so that ROPD can be discovered, validated, and
introspected by the same tooling as the other methods. They contain
**no** algorithmic logic of their own — see
``easyopd.methods.ropd.reward_manager`` and ``easyopd.methods.ropd.pipeline``
for the actual rubric / scoring implementation.
"""

from __future__ import annotations

from typing import Any

import torch

from easyopd.hooks import Batch, Config, LossHook, Metrics, RewardHook, LossContext


class ROPDLossHook:
    """Placeholder LossHook for the (black-box) ROPD method.

    ROPD does not modify the actor loss — verl's default PG loss is used as-is.
    This adapter exists so that ``HookDispatcher._build_hooks(ROPDMethod, ...)``
    yields a ``MethodHooks`` container with ``has_loss = True``, satisfying
    framework expectations that every registered method exposes a LossHook.

    The :meth:`compute_loss` implementation below returns a zero scalar with
    no metrics, signalling to callers that no additional distillation loss
    should be added on top of the standard PG objective. ROPD's training
    signal flows entirely through rewards produced by
    :class:`ROPDRewardHook` (and, at runtime, through the ``ropd`` reward
    manager registered into verl).
    """

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        config: Config,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Metrics]:
        """Return a no-op zero loss compatible with the standard PG objective.

        Args:
            student_logits: Unused. ROPD does not consume student logits at the
                LossHook layer.
            teacher_logits: Unused.
            mask: Unused.
            config: Unused.
            **kwargs: Reserved for forward compatibility.

        Returns:
            Tuple of ``(zero_scalar_loss, empty_metrics_dict)``. Callers that
            add this to verl's PG loss obtain the unmodified PG loss.
        """
        # Use student_logits' device/dtype if available so the returned tensor
        # is compatible with whatever accumulator the caller is using.
        if isinstance(student_logits, torch.Tensor):
            zero = torch.zeros((), dtype=student_logits.dtype, device=student_logits.device)
        else:
            zero = torch.zeros(())
        return zero, {}

    def compute_loss_with_context(
        self,
        context: LossContext,
    ) -> tuple[torch.Tensor, Metrics]:
        """Return a no-op zero loss using the unified LossContext interface.
        
        ROPD is a black-box method that does not modify the actor loss.
        The training signal flows entirely through rewards.
        """
        # Use context's device/dtype if available
        if context.student_log_probs is not None:
            zero = torch.zeros((), dtype=context.student_log_probs.dtype, device=context.student_log_probs.device)
        else:
            zero = torch.zeros(())
        return zero, {}


class ROPDRewardHook:
    """RewardHook adapter for the ROPD rubric-based reward manager.

    Provides a thin, side-effect-free entry point that satisfies
    ``easyopd.hooks.RewardHook``. The actual rubric scoring is performed
    by :class:`easyopd.methods.ropd.reward_manager.ROPDRewardManager` (which
    is registered into verl's reward-manager registry by
    :func:`easyopd.methods.ropd.register`); the hook here is mostly an
    introspection / contract surface used by ``HookDispatcher`` and by the
    EasyOPD test matrix.

    When called directly (i.e. outside verl), the hook will instantiate the
    underlying reward manager lazily and forward the call. This keeps import
    cost low and avoids requiring a live judge runtime simply for the hook
    object to exist.
    """

    def __init__(self) -> None:
        # Lazy: do not import or instantiate the reward manager up front.
        self._reward_manager = None

    def _get_reward_manager(self, config: Config) -> Any:
        """Lazily build (and cache) the ROPD reward manager from config."""
        if self._reward_manager is not None:
            return self._reward_manager
        # Local import to avoid pulling in heavy judge / pipeline dependencies
        # at module import time.
        from easyopd.methods.ropd.reward_manager import ROPDRewardManager

        # ROPDRewardManager requires tokenizer, num_examine, compute_score
        # which are only available in the full verl training context.
        # Extract them from config if available.
        tokenizer = config.get("tokenizer") if isinstance(config, dict) else getattr(config, "tokenizer", None)
        num_examine = config.get("num_examine", 1) if isinstance(config, dict) else getattr(config, "num_examine", 1)
        compute_score = config.get("compute_score") if isinstance(config, dict) else getattr(config, "compute_score", None)

        if tokenizer is None:
            raise RuntimeError(
                "ROPDRewardHook requires 'tokenizer' in config to instantiate "
                "the reward manager. This hook is designed to run inside the "
                "full verl training pipeline."
            )

        self._reward_manager = ROPDRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            ropd=config.get("ropd") if isinstance(config, dict) else getattr(config, "ropd", None),
        )
        return self._reward_manager

    def compute_reward(
        self,
        batch: Batch,
        config: Config,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Score a batch of rollouts using ROPD's rubric judge triple.

        Args:
            batch: A verl ``DataProto`` (or compatible batch object) containing
                generated responses.
            config: The full ROPD config (typically the merged
                ``easyopd/config/ropd/*.yaml`` payload).
            **kwargs: Reserved for runtime extensions (e.g. ``teacher_outputs``).

        Returns:
            A reward tensor of shape ``[batch_size]`` (or
            ``[batch_size, seq_len]`` if the underlying manager returns
            per-token rewards).
        """
        reward_manager = self._get_reward_manager(config)
        # Delegate to the maintained reward manager. We deliberately do not
        # re-implement scoring here so that algorithmic behaviour matches the
        # upstream ROPD mainline byte-for-byte.
        return reward_manager(batch, **kwargs)
