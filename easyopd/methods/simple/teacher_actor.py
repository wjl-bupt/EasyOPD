# Copyright 2026 EasyOPD Contributors
#
# Single Ray actor wrapping one SGLang teacher engine.
#
# Ported from KDFlow's `kdflow/ray/train/teacher_actor.py` but rewritten so that
# EasyOPD does not need to import KDFlow at runtime: KDFlow's `strategy.args.kd.*`
# argument-bag is replaced by an explicit `TeacherActorConfig` dataclass, and
# the `remove_pad_token` helper is inlined as a small numpy-only utility.
#
# Each actor owns one GPU (via base_gpu_id binding inside the SGLang engine
# subprocess; combined with RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 so
# Ray does not interfere with the engine's own device selection). The actor
# only performs prefill (max_new_tokens=0) on already-formed prompt+response
# texts and returns hidden_states at the response positions.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import ray

# Try SGLang first; fall back to HuggingFace engine if SGLang is not installed.
try:
    from easyopd.methods.simple.sglang_engine import EngineConfig, SGLangEngineService
    _SGLANG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _SGLANG_AVAILABLE = False
    from easyopd.methods.simple.hf_engine import HFEngineConfig, HFTeacherEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight utilities (KDFlow parity, no KDFlow import required)
# ---------------------------------------------------------------------------

def _remove_pad_token_numpy(
    input_ids: np.ndarray, attention_mask: np.ndarray
) -> List[np.ndarray]:
    """Return a list of unpadded id arrays, one per row.

    Mirrors KDFlow's `kdflow.utils.utils.remove_pad_token` semantics
    (works for either left- or right-padded sequences) but is numpy-only
    so this module does not depend on torch.
    """
    out: List[np.ndarray] = []
    for ids, mask in zip(input_ids, attention_mask):
        out.append(np.asarray(ids)[np.asarray(mask).astype(bool)])
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TeacherActorConfig:
    """Static configuration for a single teacher actor.

    Mirrors the subset of KDFlow's `strategy.args.kd.*` fields that are
    relevant to teacher prefill, plus the EasyOPD-side knobs the launch
    script exposes.
    """

    model_path: str
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    mem_fraction_static: float = 0.6
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    # Per the EasyOPD design discussion: teacher pool is co-resident with the
    # student pool but on disjoint GPUs, so the default is to NOT sleep
    # between forwards. Sleep/wakeup are still exposed for special configs.
    enable_sleep: bool = False
    offload_tags: Optional[str] = "all"


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------

@ray.remote
class TeacherRayActor:
    """One Ray actor wrapping one SGLang engine subprocess.

    The actor exposes a `compute_hidden_states(prompts, loss_masks)` RPC
    that runs SGLang prefill (`max_new_tokens=0`) and returns the hidden
    states at the `loss_mask`-True positions, as a list of numpy arrays
    of shape `[num_loss_tokens_i, hidden_dim]`.

    GPU binding contract:
        * Caller ensures `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` is
          set in the actor's runtime_env (see TeacherActorGroup), so Ray
          does NOT mask physical GPUs.
        * SGLang then uses `base_gpu_id` to bind to a specific physical GPU.
        * Combined effect: each actor sees all GPUs, but only touches one.
    """

    def __init__(
        self,
        config: TeacherActorConfig,
        base_gpu_id: int = 0,
        nnodes: int = 1,
        node_rank: int = 0,
        dist_init_addr: Optional[str] = None,
    ) -> None:
        logger.info(
            "[TeacherRayActor] __init__ STARTED PID=%d base_gpu_id=%d "
            "model_path=%s",
            os.getpid(),
            base_gpu_id,
            config.model_path,
        )
        self.config = config
        self.base_gpu_id = base_gpu_id
        self.node_rank = node_rank

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if _SGLANG_AVAILABLE:
            engine_config = EngineConfig(
                model_path=config.model_path,
                tp_size=config.tp_size,
                ep_size=config.ep_size,
                pp_size=config.pp_size,
                chunked_prefill_size=-1,
                disable_radix_cache=True,
                quantization=config.quantization,
                mem_fraction_static=config.mem_fraction_static,
                context_length=config.context_length,
                offload_tags=config.offload_tags,
                base_gpu_id=base_gpu_id,
                nnodes=nnodes,
                node_rank=node_rank,
                dist_init_addr=dist_init_addr,
            )
            self.engine_service = SGLangEngineService(engine_config)
        else:
            logger.info(
                "[TeacherRayActor] SGLang not available, using HF engine."
            )
            hf_config = HFEngineConfig(
                model_path=config.model_path,
                base_gpu_id=base_gpu_id,
                mem_fraction_static=config.mem_fraction_static,
                context_length=config.context_length,
                quantization=config.quantization,
            )
            self.engine_service = HFTeacherEngine(hf_config)

        self.engine_service.start()

        if self.config.enable_sleep and self.node_rank == 0:
            logger.info(
                "[TeacherRayActor] sleep after init (offload_tags=%s)",
                self.config.offload_tags,
            )
            self.engine_service.sleep(tags=self.config.offload_tags)

        logger.info(
            "[TeacherRayActor] ready PID=%d base_gpu_id=%d tp=%d ep=%d pp=%d",
            os.getpid(),
            base_gpu_id,
            config.tp_size,
            config.ep_size,
            config.pp_size,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ready(self) -> bool:
        """Return True once the engine subprocess has finished initialization."""
        return bool(self.engine_service._started)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_hidden_states(
        self,
        prompts_ref: Any,
        input_ids_ref: Any,
        loss_masks_ref: Any,
        batch_indices: List[int],
    ) -> List[Tuple[int, np.ndarray]]:
        """Run prefill on the assigned subset of a batch and return hidden states.

        Args:
            prompts_ref: ray.ObjectRef OR an in-memory list of full
                prompt+response strings (one per sample in the *full* batch).
            input_ids_ref: ray.ObjectRef OR an in-memory list of pre-tokenized
                teacher input ids. If not None, these ids are sent directly to
                SGLang and prompts are only kept as a fallback/debug payload.
            loss_masks_ref: ray.ObjectRef OR list of np.bool_ arrays, one
                per sample in the full batch. Each mask has shape
                `[teacher_seq_len_i]` and selects the tokens whose hidden
                states the loss will consume (typically: the response
                positions, possibly with span-masking applied).
            batch_indices: which indices of the full batch this actor
                should process (assigned by `TeacherActorGroup` via
                token-balanced load balancing).

        Returns:
            A list of `(original_batch_idx, hidden_states_np)` tuples,
            preserving the caller's original batch order so the group can
            re-sort. `hidden_states_np` has shape
            `[num_loss_tokens_i, hidden_dim]` and dtype matching the
            engine's compute dtype (typically bfloat16/float16 in numpy).
        """
        # Resolve refs lazily; accept both ObjectRef and direct payloads
        # so this RPC is unit-testable without a Ray cluster.
        prompts_full = (
            ray.get(prompts_ref) if isinstance(prompts_ref, ray.ObjectRef) else prompts_ref
        )
        input_ids_full = (
            ray.get(input_ids_ref)
            if isinstance(input_ids_ref, ray.ObjectRef)
            else input_ids_ref
        )
        loss_masks_full = (
            ray.get(loss_masks_ref)
            if isinstance(loss_masks_ref, ray.ObjectRef)
            else loss_masks_ref
        )

        if not batch_indices:
            return []

        # Pull the assigned slice of the global batch.
        prompts = [prompts_full[i] for i in batch_indices]
        input_ids = (
            [list(input_ids_full[i]) for i in batch_indices]
            if input_ids_full is not None
            else None
        )
        loss_masks = [
            np.asarray(loss_masks_full[i]).astype(bool) for i in batch_indices
        ]

        # Prefill-only forward: max_new_tokens=0.
        hidden_states_list = self.engine_service.generate(
            prompt=None if input_ids is not None else prompts,
            input_ids=input_ids,
            loss_masks=loss_masks,
            sampling_params={"max_new_tokens": 0},
            return_hidden_states=True,
        )

        # Pair each result back with its original batch index so the caller
        # can re-order across actors.
        results: List[Tuple[int, np.ndarray]] = list(
            zip(batch_indices, hidden_states_list)
        )
        return results

    # ------------------------------------------------------------------
    # Memory hygiene
    # ------------------------------------------------------------------

    def sleep(self, tags: Optional[str] = None) -> None:
        if tags is None:
            tags = self.config.offload_tags
        self.engine_service.sleep(tags=tags)

    def wakeup(self, tags: Optional[str] = None) -> None:
        if tags is None:
            tags = self.config.offload_tags
        self.engine_service.wakeup(tags=tags)

    def shutdown(self) -> None:
        self.engine_service.shutdown()
        logger.info("[TeacherRayActor] shutdown PID=%d", os.getpid())


__all__ = [
    "TeacherActorConfig",
    "TeacherRayActor",
    "_remove_pad_token_numpy",
]
