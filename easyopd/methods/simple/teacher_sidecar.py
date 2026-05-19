# Copyright 2026 EasyOPD Contributors
#
# EasyOPD-side teacher manager that lives alongside (NOT inside) verl's
# `MultiTeacherModelManager`. Created by `ray_trainer.py` when the
# distillation loss is `simple` (cross-tokenizer mode), instead of the
# normal verl teacher manager.
#
# Responsibilities:
#   1. Spin up an SGLang teacher pool via `TeacherActorGroup` on the GPU
#      slots that verl reserved for teachers (n_gpus_per_node * nnodes).
#   2. Expose `compute_hidden_states_batch(prompts, loss_masks)` to be
#      called from `agent_loop._postprocess`.
#   3. Provide a shutdown / sleep interface so the trainer's existing
#      teardown path keeps working.
#
# Why a separate sidecar rather than reusing verl's manager?
#   * verl's `TeacherModelManager` returns an `LLMServerClient` whose API
#     is "async generate(prompt_ids, sampling_params)" — it does NOT
#     surface hidden states, only logprobs.
#   * verl's manager spawns rollout *replicas* via PlacementGroups; we
#     need raw single-GPU Ray actors (see TeacherActorGroup).
#   * The two layouts are incompatible enough that emulating verl's
#     interface would either require deep verl edits or a duck-typed
#     LLMServerClient that does ZMQ-level translation. A separate sidecar
#     is the lowest-risk path and matches the design we agreed to.

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import numpy as np

from easyopd.methods.simple.teacher_actor import TeacherActorConfig
from easyopd.methods.simple.teacher_group import TeacherActorGroup

logger = logging.getLogger(__name__)


def _resolve_dp_size(distillation_config: Any, teacher_cfg: Any) -> int:
    """Compute how many independent SGLang teacher actors to spawn.

    Default rule: total teacher GPU pool / per-actor GPU footprint.

    The teacher pool size is `distillation.n_gpus_per_node *
    distillation.nnodes`; per-actor footprint is `tp_size * pp_size`
    (we always use ep_size=1 / pp_size=1 in the SGLang teacher path).
    """
    n_gpus_per_node = int(getattr(distillation_config, "n_gpus_per_node", 0) or 0)
    nnodes = int(getattr(distillation_config, "nnodes", 0) or 0)
    pool_size = n_gpus_per_node * nnodes
    if pool_size <= 0:
        raise ValueError(
            "[EasyOPD:simple sidecar] distillation pool size resolved to 0 "
            f"(n_gpus_per_node={n_gpus_per_node}, nnodes={nnodes})."
        )
    inference = teacher_cfg.inference
    per_actor = (
        int(getattr(inference, "tensor_model_parallel_size", 1) or 1)
        * int(getattr(inference, "pipeline_model_parallel_size", 1) or 1)
    )
    if pool_size % per_actor != 0:
        raise ValueError(
            f"[EasyOPD:simple sidecar] teacher pool size {pool_size} is not "
            f"divisible by per-actor footprint {per_actor}."
        )
    return pool_size // per_actor


class EasyOPDSimpleTeacherSidecar:
    """Owns the EasyOPD `TeacherActorGroup` for cross-tokenizer KD."""

    def __init__(self, config) -> None:
        """
        Args:
            config: the full Hydra/OmegaConf trainer config (the same
                object passed to verl's ray trainer). We pull the
                distillation sub-tree out internally.
        """
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import DistillationConfig

        self.full_config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)

        teacher_models = self.distillation_config.teacher_models
        if len(teacher_models) != 1:
            # Multi-teacher cross-tokenizer KD is not yet supported (KDFlow
            # parity with the simple_ctkd reference is single-teacher).
            raise NotImplementedError(
                f"[EasyOPD:simple sidecar] expected exactly one teacher in "
                f"cross-tokenizer mode, got {len(teacher_models)}."
            )
        self.teacher_key, teacher_cfg = next(iter(teacher_models.items()))
        self.teacher_cfg = teacher_cfg

        dp_size = _resolve_dp_size(self.distillation_config, teacher_cfg)
        n_gpus_per_node = int(self.distillation_config.n_gpus_per_node)

        actor_config = TeacherActorConfig(
            model_path=teacher_cfg.model_path,
            tp_size=int(teacher_cfg.inference.tensor_model_parallel_size),
            pp_size=int(teacher_cfg.inference.pipeline_model_parallel_size),
            ep_size=1,
            mem_fraction_static=float(
                getattr(teacher_cfg.inference, "gpu_memory_utilization", 0.6) or 0.6
            ),
            context_length=getattr(teacher_cfg.inference, "max_model_len", None),
            quantization=None,
            enable_sleep=False,  # per design: long-resident, no sleep
            offload_tags="all",
        )

        logger.warning(
            "[EasyOPD:simple sidecar] launching TeacherActorGroup "
            "model=%s dp_size=%d tp=%d pp=%d mem_fraction=%.2f",
            actor_config.model_path,
            dp_size,
            actor_config.tp_size,
            actor_config.pp_size,
            actor_config.mem_fraction_static,
        )

        # NOTE: we do not pass a placement group here — the teacher pool
        # GPUs are reserved by verl's resource pool wiring but we let
        # Ray default-schedule the actors with `num_gpus=0.2` and
        # `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`. This works
        # when the teacher GPUs are physically distinct from the student
        # rollout GPUs (which is the case in our 6+2 layout).
        self.actor_group = TeacherActorGroup(
            actor_config=actor_config,
            dp_size=dp_size,
            num_gpus_per_node=n_gpus_per_node,
            num_gpus_per_actor=0.2,
        )

        # Cache tokenizers eagerly (cheap, ~hundreds of KB) so callers
        # don't need to re-load.
        from transformers import AutoTokenizer

        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_cfg.model_path, trust_remote_code=True
        )
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

    # ------------------------------------------------------------------
    # Inference API consumed by agent_loop / postprocess
    # ------------------------------------------------------------------

    def compute_hidden_states_batch(
        self,
        prompts: List[str],
        loss_masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Forward to the actor group. Returns one numpy array per sample."""
        return self.actor_group.compute_hidden_states_batch(
            prompts=prompts, loss_masks=loss_masks
        )

    def encode_for_teacher(
        self,
        prompt_text: str,
        response_text: str,
        max_length: Optional[int] = None,
    ) -> tuple[List[int], np.ndarray, str]:
        """Encode `prompt + response` with the teacher tokenizer and produce
        a per-token loss mask + the concatenated text the SGLang engine
        will receive.

        Returns:
            teacher_ids: List[int] of length T.
            loss_mask:   np.bool_[T] — True at response positions.
            full_text:   str — `prompt_text + response_text`. SGLang
                tokenizes internally so we DO pass text, not ids; ids are
                returned only for downstream cross-tokenizer alignment.
        """
        tea = self.teacher_tokenizer
        full_text = prompt_text + response_text
        full_ids = tea(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tea(prompt_text, add_special_tokens=False)["input_ids"]
        if max_length is not None and len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
        # Prompt boundary heuristic (matches KDFlow): use len(prompt_ids)
        # as the response start, clamped to len(full_ids).
        boundary = min(len(prompt_ids), len(full_ids))
        mask = np.zeros(len(full_ids), dtype=bool)
        mask[boundary:] = True
        return full_ids, mask, full_text

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        try:
            self.actor_group.shutdown()
        except Exception:
            logger.exception("[EasyOPD:simple sidecar] shutdown failed")


__all__ = ["EasyOPDSimpleTeacherSidecar"]
