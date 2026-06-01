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

"""EasyOPD Hook Dispatcher.

Routes hook calls from verl's training loop to the active method's hook
implementations. The dispatcher is the single integration point between
verl core code and EasyOPD method logic.

Usage in verl (thin hook call points)::

    # In ray_trainer.py __init__:
    from easyopd.hook_dispatch import HookDispatcher
    self.hook_dispatcher = HookDispatcher.from_config(self.config)

    # In training loop:
    batch = self.hook_dispatcher.on_rollout_end(batch)
    loss, metrics = self.hook_dispatcher.compute_loss(student_logits, teacher_logits, mask)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from easyopd.hooks import (
    AlignmentHook,
    Batch,
    Config,
    LossHook,
    MethodHooks,
    Metrics,
    RewardHook,
    RolloutHook,
    TeacherSidecarHook,
)

logger = logging.getLogger(__name__)


class HookDispatcher:
    """Central dispatcher that routes hook calls to the active method.

    The dispatcher holds a reference to the active method's MethodHooks
    container and provides safe dispatch methods that gracefully handle
    cases where a hook is not implemented.

    Attributes:
        method_name: Name of the active method (or None if no method is active).
        hooks: The MethodHooks container for the active method.
        config: Method-specific configuration.
    """

    def __init__(
        self,
        method_name: Optional[str] = None,
        hooks: Optional[MethodHooks] = None,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            method_name: Name of the active OPD method.
            hooks: MethodHooks container with the method's hook implementations.
            config: Method-specific configuration dict.
        """
        self.method_name = method_name
        self.hooks = hooks or MethodHooks()
        self.config = config or {}
        self._enabled = method_name is not None

        if self._enabled:
            logger.info(
                "HookDispatcher initialized for method '%s' with hooks: %s",
                method_name,
                self.hooks.active_hooks(),
            )

    @property
    def enabled(self) -> bool:
        """Whether the dispatcher has an active method."""
        return self._enabled

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any, auto_resolve_data: bool = True) -> "HookDispatcher":
        """Create a HookDispatcher from a training config.

        Looks for EasyOPD method configuration in the config object and
        sets up the appropriate hooks. Also auto-resolves dataset references
        (HuggingFace dataset names) to local Parquet files.

        Args:
            config: The full training configuration (OmegaConf or dict).
            auto_resolve_data: If True, automatically download and convert
                datasets specified by HuggingFace name in the config.

        Returns:
            A configured HookDispatcher instance.
        """
        # Try to extract EasyOPD method name from config
        method_name = cls._extract_method_name(config)

        if method_name is None:
            logger.debug("No EasyOPD method configured; dispatcher is inactive.")
            return cls()

        # Import here to avoid circular imports
        from easyopd.registry import ensure_discovered, get_method, is_registered

        ensure_discovered()

        if not is_registered(method_name):
            logger.warning(
                "EasyOPD method '%s' specified in config but not registered. "
                "Dispatcher will be inactive.",
                method_name,
            )
            return cls()

        # Auto-resolve data references (HuggingFace -> local Parquet)
        if auto_resolve_data and isinstance(config, dict):
            from easyopd.data_provider import resolve_data_in_config
            config = resolve_data_in_config(config)

        method_cls = get_method(method_name)

        # Try to get hooks from the method class
        hooks = cls._build_hooks(method_cls, config)
        method_config = cls._extract_method_config(config)

        return cls(
            method_name=method_name,
            hooks=hooks,
            config=method_config,
        )

    # ------------------------------------------------------------------
    # Dispatch methods (called from verl training loop)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[Optional[torch.Tensor], Metrics]:
        """Dispatch to the method's LossHook.

        Args:
            student_logits: Student model output logits.
            teacher_logits: Teacher model output logits.
            mask: Valid position mask.
            **kwargs: Additional arguments passed to the hook.

        Returns:
            Tuple of (loss, metrics). Returns (None, {}) if no LossHook is active.
        """
        if not self._enabled or not self.hooks.has_loss:
            return None, {}

        return self.hooks.loss_hook.compute_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=mask,
            config=self.config,
            **kwargs,
        )

    def on_rollout_end(self, batch: Batch, **kwargs: Any) -> Batch:
        """Dispatch to the method's RolloutHook.

        Args:
            batch: Training batch after rollout.
            **kwargs: Additional context.

        Returns:
            The (possibly modified) batch. Returns batch unchanged if no
            RolloutHook is active.
        """
        if not self._enabled or not self.hooks.has_rollout:
            return batch

        return self.hooks.rollout_hook.on_rollout_end(
            batch=batch,
            config=self.config,
            **kwargs,
        )

    def compute_reward(self, batch: Batch, **kwargs: Any) -> Optional[torch.Tensor]:
        """Dispatch to the method's RewardHook.

        Args:
            batch: Training batch with generated responses.
            **kwargs: Additional context.

        Returns:
            Reward tensor, or None if no RewardHook is active.
        """
        if not self._enabled or not self.hooks.has_reward:
            return None

        return self.hooks.reward_hook.compute_reward(
            batch=batch,
            config=self.config,
            **kwargs,
        )

    def build_alignment(
        self,
        student_tokenizer: Any,
        teacher_tokenizer: Any,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """Dispatch to the method's AlignmentHook.

        Args:
            student_tokenizer: Student tokenizer.
            teacher_tokenizer: Teacher tokenizer.
            input_ids: Input token IDs.
            **kwargs: Additional context.

        Returns:
            Alignment map, or None if no AlignmentHook is active.
        """
        if not self._enabled or not self.hooks.has_alignment:
            return None

        return self.hooks.alignment_hook.build_alignment(
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            input_ids=input_ids,
            config=self.config,
            **kwargs,
        )

    def teacher_forward(
        self,
        batch: Batch,
        teacher_model: Any,
        **kwargs: Any,
    ) -> Any:
        """Dispatch to the method's TeacherSidecarHook.

        Args:
            batch: Training batch.
            teacher_model: Teacher model instance.
            **kwargs: Additional context.

        Returns:
            Teacher outputs, or None if no TeacherSidecarHook is active.
        """
        if not self._enabled or not self.hooks.has_teacher_sidecar:
            return None

        return self.hooks.teacher_sidecar_hook.teacher_forward(
            batch=batch,
            teacher_model=teacher_model,
            config=self.config,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_method_name(config: Any) -> Optional[str]:
        """Extract the EasyOPD method name from config.

        Supports multiple config formats:
            - OmegaConf DictConfig with nested access
            - Plain dict
            - Config with `easyopd.method.name` path
        """
        # Try OmegaConf-style access
        try:
            from omegaconf import OmegaConf

            if hasattr(config, "_metadata"):  # OmegaConf object
                name = OmegaConf.select(config, "easyopd.method.name", default=None)
                if name:
                    return name
                # Fallback: check actor.policy_loss.loss_mode for legacy configs
                name = OmegaConf.select(config, "actor_rollout_ref.actor.policy_loss.loss_mode", default=None)
                if name and name in ("simple", "simct", "gkd", "sod", "g_opd", "opcd", "vision_opd", "sdpo", "vopd"):
                    # Map vopd -> vision_opd
                    if name == "vopd":
                        return "vision_opd"
                    return name
        except ImportError:
            pass

        # Try plain dict access
        if isinstance(config, dict):
            easyopd_cfg = config.get("easyopd", {})
            if isinstance(easyopd_cfg, dict):
                method_cfg = easyopd_cfg.get("method", {})
                if isinstance(method_cfg, dict):
                    return method_cfg.get("name")

        return None

    @staticmethod
    def _extract_method_config(config: Any) -> dict:
        """Extract method-specific config section."""
        try:
            from omegaconf import OmegaConf

            if hasattr(config, "_metadata"):
                method_cfg = OmegaConf.select(config, "easyopd.method", default=None)
                if method_cfg:
                    return OmegaConf.to_container(method_cfg, resolve=True)
        except ImportError:
            pass

        if isinstance(config, dict):
            easyopd_cfg = config.get("easyopd", {})
            if isinstance(easyopd_cfg, dict):
                return easyopd_cfg.get("method", {})

        return {}

    @staticmethod
    def _build_hooks(method_cls: Any, config: Any) -> MethodHooks:
        """Build MethodHooks from a method class.

        Checks multiple strategies in order:
        1. Method class has a `build_hooks()` classmethod
        2. Method's module has a sibling `hooks` module with hook classes
        3. Method class has hook attributes directly

        Args:
            method_cls: The registered method metadata class.
            config: Full training config.

        Returns:
            MethodHooks container.
        """
        # Strategy 1: Method class has a build_hooks() classmethod
        if hasattr(method_cls, "build_hooks") and callable(getattr(method_cls, "build_hooks")):
            try:
                hooks = method_cls.build_hooks(config)
                if isinstance(hooks, MethodHooks):
                    return hooks
            except Exception as e:
                logger.warning(
                    "Failed to call %s.build_hooks(): %s. Falling back.",
                    method_cls.__qualname__,
                    e,
                )

        # Strategy 2: Import hooks from the method's hooks.py module
        method_module = getattr(method_cls, "__module__", None)
        if method_module:
            # e.g. "easyopd.methods.gkd" -> "easyopd.methods.gkd.hooks"
            hooks_module_name = f"{method_module}.hooks"
            try:
                import importlib

                hooks_module = importlib.import_module(hooks_module_name)
                hooks_kwargs = {}

                # Look for standard hook class names
                # Method-specific names are checked first (e.g. GKDLossHook)
                # to avoid accidentally matching the Protocol base class.
                hook_class_mapping = {
                    "loss_hook": [f"{method_cls.__name__.replace('Method', '')}LossHook", "LossHook"],
                    "rollout_hook": [f"{method_cls.__name__.replace('Method', '')}RolloutHook", "RolloutHook"],
                    "reward_hook": [f"{method_cls.__name__.replace('Method', '')}RewardHook", "RewardHook"],
                    "alignment_hook": [f"{method_cls.__name__.replace('Method', '')}AlignmentHook", "AlignmentHook"],
                    "teacher_sidecar_hook": [f"{method_cls.__name__.replace('Method', '')}TeacherSidecarHook", "TeacherSidecarHook"],
                }

                for hook_attr, class_names in hook_class_mapping.items():
                    for cls_name in class_names:
                        hook_cls = getattr(hooks_module, cls_name, None)
                        if hook_cls is not None:
                            # Skip Protocol base classes (they cannot be instantiated)
                            if getattr(hook_cls, "_is_protocol", False):
                                continue
                            try:
                                hooks_kwargs[hook_attr] = hook_cls()
                            except TypeError:
                                # Cannot instantiate (e.g. abstract class)
                                continue
                            break

                if hooks_kwargs:
                    return MethodHooks(**hooks_kwargs)
            except (ImportError, ModuleNotFoundError):
                pass  # No hooks.py module for this method

        # Strategy 3: Method class has hook attributes directly
        hooks_kwargs = {}
        for attr, hook_type in [
            ("loss_hook", LossHook),
            ("rollout_hook", RolloutHook),
            ("reward_hook", RewardHook),
            ("alignment_hook", AlignmentHook),
            ("teacher_sidecar_hook", TeacherSidecarHook),
        ]:
            hook_impl = getattr(method_cls, attr, None)
            if hook_impl is not None:
                hooks_kwargs[attr] = hook_impl

        return MethodHooks(**hooks_kwargs)

    def __repr__(self) -> str:
        if not self._enabled:
            return "HookDispatcher(enabled=False)"
        return (
            f"HookDispatcher(method={self.method_name!r}, "
            f"hooks={self.hooks.active_hooks()})"
        )
