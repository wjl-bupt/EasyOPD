# Copyright 2026 EasyOPD Contributors
#
# Multi-actor teacher group: schedules N independent SGLang teacher actors
# (each on its own GPU) and dispatches batches across them with
# token-balanced load balancing.
#
# Ported from KDFlow's `kdflow/ray/train/teacher_group.py` with these changes:
#   * No dependency on KDFlow's `strategy` object — config is passed
#     explicitly as `TeacherActorConfig`.
#   * Resource scheduling is decoupled from KDFlow's PlacementGroup helper:
#     this group accepts an optional verl-style `RayResourcePool`
#     (or, for unit tests, can run without one), and falls back to a plain
#     fractional-GPU schedule otherwise.
#   * The `forward` API is renamed to `compute_hidden_states_batch` and
#     returns numpy hidden_states keyed by sample index, matching the new
#     EasyOPD-side data flow (see `simple/losses.py`).

from __future__ import annotations

import logging
import time
from itertools import chain
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from easyopd.methods.simple.teacher_actor import (
    TeacherActorConfig,
    TeacherRayActor,
)

logger = logging.getLogger(__name__)


# Environment variables that must be set on each teacher actor so that Ray
# does NOT mask physical GPUs — SGLang then binds via `base_gpu_id`. This is
# the KDFlow convention; we keep it as it is the only way to share placement
# groups with student rollout actors without GPU collisions.
_NOSET_VISIBLE_DEVICES_ENV_VARS = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


class TeacherActorGroup:
    """Manages a pool of independent teacher actors, one per GPU.

    Layout (matches KDFlow):
        * `dp_size` independent actors, each running its own SGLang engine
          subprocess.
        * Each actor occupies `tp_size * pp_size` GPUs (typically 1).
        * Total GPU footprint = dp_size * tp_size * pp_size.

    Scheduling:
        * Per the EasyOPD design, we skew resource accounting by passing
          `num_gpus_per_actor` as a small fraction (default 0.2) and rely on
          `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` plus `base_gpu_id`
          for actual binding. This is the KDFlow pattern and is required to
          let teacher actors share a placement group with student rollouts
          on disjoint GPU IDs without Ray re-masking devices.
        * When colocating with verl's strict 8-GPU PG (default for the
          shared-mode launch.sh), set `num_gpus_per_actor=0` so that the
          teacher does NOT pre-occupy entries in Ray's GPU ledger — verl's
          `_check_resource_available()` reads available GPUs *after* PG
          creation, and any fractional pre-allocation makes verl reject
          its own 8-GPU request. CPU resource is independent
          (`num_cpus_per_actor`, default 1.0) so the actor still schedules.

    Load balancing:
        * `compute_hidden_states_batch` greedily assigns each sample to the
          actor with the currently smallest assigned token-count (token = sum
          of `loss_mask`), so latency is bounded by the worst-loaded actor's
          load rather than naive index-mod-N partitioning.

    Output:
        * Returns a list of numpy hidden-state arrays in the same order as
          the input prompts. Each array has shape `[num_loss_tokens_i, H]`.
    """

    def __init__(
        self,
        actor_config: TeacherActorConfig,
        dp_size: int,
        num_gpus_per_node: int = 8,
        num_gpus_per_actor: float = 0.2,
        num_cpus_per_actor: float = 0.1,
        pg: Optional[PlacementGroup] = None,
        reordered_bundle_indices: Optional[Sequence[int]] = None,
        reordered_gpu_ids: Optional[Sequence[int]] = None,
        base_gpu_ids: Optional[Sequence[int]] = None,
        teacher_visible_devices: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            actor_config: shared per-actor configuration (model path, tp/pp size,
                memory fraction, etc.).
            dp_size: number of teacher replicas; equals the number of Ray
                actors created.
            num_gpus_per_node: physical GPUs per node (used to compute
                `base_gpu_id` for each actor when no PG is provided).
            num_gpus_per_actor: fractional Ray GPU resource per actor. Kept
                small (~0.2) so the actual GPU is bound by `base_gpu_id`,
                not by Ray's CUDA_VISIBLE_DEVICES masking. Set to 0 to
                fully bypass Ray's GPU ledger when colocating with a strict
                full-GPU PG (e.g. verl's shared-mode 8S+8T layout).
            num_cpus_per_actor: Ray CPU resource per actor (default 1.0).
                Independent from `num_gpus_per_actor` so we can ask for 0
                GPU and >0 CPU at the same time.
            pg: optional placement group within which actors should be
                scheduled. If provided, `reordered_bundle_indices` and
                `reordered_gpu_ids` should specify the per-actor bundle/GPU
                mapping (verl-style).
            reordered_bundle_indices: bundle index per actor (length
                dp_size * tp_size * pp_size). Required when `pg` is given.
            reordered_gpu_ids: physical GPU id per bundle (same length).
                Used to compute `base_gpu_id`.
            base_gpu_ids: explicit physical GPU id per teacher engine.
                Takes precedence over `reordered_gpu_ids` and round-robin
                fallback. Use this for colocated layouts such as 8S+8T.
            teacher_visible_devices: optional CUDA_VISIBLE_DEVICES list for
                teacher actors. Use this to let teacher actors see physical
                GPUs that the top-level student process masks out.
        """
        logger.info(
            "[TeacherActorGroup] init dp_size=%d tp_size=%d pp_size=%d "
            "model_path=%s",
            dp_size,
            actor_config.tp_size,
            actor_config.pp_size,
            actor_config.model_path,
        )
        self.actor_config = actor_config
        self.dp_size = dp_size
        self.tp_size = actor_config.tp_size
        self.pp_size = actor_config.pp_size
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = num_gpus_per_actor
        # CPU resource is decoupled from GPU resource: when colocating with
        # verl's strict 8-GPU PG (`num_gpus_per_actor=0`), Ray would still
        # need a non-zero `num_cpus` to schedule the actor on a worker node.
        # Keep this small (default 0.1) so the teacher does not eat into
        # verl's CPU budget — verl's actor_rollout_ref worker group asks for
        # 1 CPU per worker (8 CPUs total) on top of its 8 GPUs, and Ray's
        # CPU ledger is shared. Setting num_cpus_per_actor=1.0 caused 8
        # teacher actors to consume 8 CPUs, leaving only 7 for verl
        # (RAY_NUM_CPUS=16 - 1 driver - 8 teachers = 7), which made verl's
        # 8-worker request "infeasible" and the trainer would hang in
        # ray.get() forever.
        self.num_cpus_per_actor = float(num_cpus_per_actor)

        self._pg = pg
        self._reordered_bundle_indices = (
            list(reordered_bundle_indices) if reordered_bundle_indices is not None else None
        )
        self._reordered_gpu_ids = (
            list(reordered_gpu_ids) if reordered_gpu_ids is not None else None
        )
        self._base_gpu_ids = list(base_gpu_ids) if base_gpu_ids is not None else None
        if self._base_gpu_ids is not None:
            if len(self._base_gpu_ids) != self.dp_size:
                raise ValueError(
                    "base_gpu_ids length must equal dp_size: "
                    f"{len(self._base_gpu_ids)} != {self.dp_size}"
                )
        self._teacher_visible_devices = (
            [int(gpu_id) for gpu_id in teacher_visible_devices]
            if teacher_visible_devices is not None
            else None
        )

        self.teacher_engines: List[ray.actor.ActorHandle] = []
        # Worker actors are auxiliary actors spawned only when a single
        # engine spans multiple nodes (multi-node tp/pp). For our 7B teacher
        # / DP=2 / TP=1 default they will be empty.
        self._worker_actors: List[ray.actor.ActorHandle] = []

        self._create_actors()
        ray.get([actor.ready.remote() for actor in self.teacher_engines])
        logger.info("[TeacherActorGroup] all %d actors ready", self.dp_size)

    # ------------------------------------------------------------------
    # Actor creation
    # ------------------------------------------------------------------

    def _create_actors(self) -> None:
        env_vars = {name: "1" for name in _NOSET_VISIBLE_DEVICES_ENV_VARS}
        if self._teacher_visible_devices is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu_id) for gpu_id in self._teacher_visible_devices
            )
        num_gpu_per_engine = self.tp_size * self.pp_size
        nnodes_per_engine = max(num_gpu_per_engine // self.num_gpus_per_node, 1)

        for i in range(self.dp_size):
            if nnodes_per_engine > 1:
                self._create_multi_node_engine(i, num_gpu_per_engine, nnodes_per_engine, env_vars)
            else:
                self._create_single_node_engine(i, num_gpu_per_engine, env_vars)

    def _create_single_node_engine(
        self,
        engine_idx: int,
        num_gpu_per_engine: int,
        env_vars: dict,
    ) -> None:
        if self._base_gpu_ids is not None:
            base_gpu_id = int(self._base_gpu_ids[engine_idx])
        elif self._reordered_gpu_ids is not None:
            base_gpu_id = int(self._reordered_gpu_ids[engine_idx * num_gpu_per_engine])
        else:
            base_gpu_id = (engine_idx * num_gpu_per_engine) % self.num_gpus_per_node

        options: dict = {
            "num_cpus": self.num_cpus_per_actor,
            "num_gpus": self.num_gpus_per_actor,
            "max_concurrency": 2,
            "runtime_env": {"env_vars": env_vars},
        }
        if self._pg is not None and self._reordered_bundle_indices is not None:
            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=self._pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=self._reordered_bundle_indices[
                    engine_idx * num_gpu_per_engine
                ],
            )

        logger.info(
            "[TeacherActorGroup] launching engine #%d base_gpu_id=%d",
            engine_idx,
            base_gpu_id,
        )
        actor = TeacherRayActor.options(**options).remote(
            self.actor_config,
            base_gpu_id=base_gpu_id,
        )
        self.teacher_engines.append(actor)

    def _create_multi_node_engine(
        self,
        engine_idx: int,
        num_gpu_per_engine: int,
        nnodes_per_engine: int,
        env_vars: dict,
    ) -> None:
        """Multi-node tp/pp engine: spawn one head + N-1 worker actors and
        share a dist_init_addr. Identical structure to KDFlow's branch but
        kept for completeness; not exercised by the default 7B-DP=2 layout.
        """
        if self._pg is None or self._reordered_bundle_indices is None:
            raise RuntimeError(
                "Multi-node teacher engines require a placement group with "
                "explicit bundle/gpu mappings."
            )

        dist_init_addr = self._get_dist_init_addr(engine_idx, num_gpu_per_engine)
        engine_actors: List[ray.actor.ActorHandle] = []
        for node_idx in range(nnodes_per_engine):
            gpu_offset = engine_idx * num_gpu_per_engine + node_idx * self.num_gpus_per_node
            assert self._reordered_gpu_ids is not None
            base_gpu_id = int(self._reordered_gpu_ids[gpu_offset])
            bundle_idx = self._reordered_bundle_indices[gpu_offset]

            options = {
                "num_cpus": self.num_cpus_per_actor,
                "num_gpus": self.num_gpus_per_actor,
                "max_concurrency": 2,
                "runtime_env": {"env_vars": env_vars},
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=self._pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_idx,
                ),
            }
            actor = TeacherRayActor.options(**options).remote(
                self.actor_config,
                base_gpu_id=base_gpu_id,
                nnodes=nnodes_per_engine,
                node_rank=node_idx,
                dist_init_addr=dist_init_addr,
            )
            engine_actors.append(actor)

        # First actor is the "head" used for RPCs; others are workers.
        self.teacher_engines.append(engine_actors[0])
        self._worker_actors.extend(engine_actors[1:])

    @staticmethod
    def _format_host(host: str) -> str:
        if ":" in host and not host.startswith("["):
            return f"[{host}]"
        return host

    def _get_dist_init_addr(self, engine_idx: int, num_gpu_per_engine: int) -> str:
        offset = engine_idx * num_gpu_per_engine
        assert self._reordered_bundle_indices is not None
        bundle_idx = self._reordered_bundle_indices[offset]

        @ray.remote(num_cpus=0, num_gpus=0)
        def _get_node_ip_and_free_port():
            import socket

            ip = ray.util.get_node_ip_address()
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                s.bind(("::", 0))
                port = s.getsockname()[1]
            return ip, port

        ip, port = ray.get(
            _get_node_ip_and_free_port.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._pg,
                    placement_group_bundle_index=bundle_idx,
                )
            ).remote()
        )
        return f"{self._format_host(ip)}:{port}"

    # ------------------------------------------------------------------
    # Inference: token-balanced batch dispatch
    # ------------------------------------------------------------------

    def compute_hidden_states_batch(
        self,
        prompts: List[str],
        loss_masks: List[np.ndarray],
        input_ids: Optional[List[List[int]]] = None,
        wait_timeout_s: float = 600.0,
        method_name: str = "simple",
    ) -> List[np.ndarray]:
        """Run prefill on a full batch by sharding samples across actors.

        Args:
            prompts: list of B teacher-side prompt+response strings (text,
                already concatenated).
            input_ids: optional list of B pre-tokenized teacher input id
                sequences. If provided, SGLang receives these directly and
                `prompts` is only used as a fallback/debug payload.
            loss_masks: list of B numpy boolean masks. Mask `i` has shape
                `[teacher_seq_len_i]` and selects the tokens whose hidden
                states the loss will consume (typically the response
                positions, possibly with span-masking).
            wait_timeout_s: hard upper bound on the total wait time.

        Returns:
            List of B numpy hidden-state arrays, in the same order as
            `prompts`. Array `i` has shape `[mask_i.sum(), hidden_dim]`.
        """
        if len(prompts) != len(loss_masks):
            raise ValueError(
                f"prompts ({len(prompts)}) and loss_masks ({len(loss_masks)}) "
                f"length mismatch."
            )
        if input_ids is not None and len(input_ids) != len(loss_masks):
            raise ValueError(
                f"input_ids ({len(input_ids)}) and loss_masks ({len(loss_masks)}) "
                f"length mismatch."
            )
        if not prompts:
            return []

        input_lengths = [len(ids) for ids in input_ids] if input_ids is not None else [len(p) for p in prompts]

        # Token-balanced greedy assignment: always send the next sample to
        # the actor that currently has the fewest tokens scheduled.
        sample_token_counts = [int(np.asarray(m).astype(bool).sum()) for m in loss_masks]
        actor_assignments: List[List[int]] = [[] for _ in range(self.dp_size)]
        actor_tokens = [0] * self.dp_size
        for sample_idx, ntok in enumerate(sample_token_counts):
            tgt = min(range(self.dp_size), key=lambda x: actor_tokens[x])
            actor_assignments[tgt].append(sample_idx)
            actor_tokens[tgt] += ntok

        # ray.put once so we don't re-serialize the prompts/masks per actor.
        prompts_ref = ray.put(prompts)
        input_ids_ref = ray.put(input_ids) if input_ids is not None else None
        masks_ref = ray.put(loss_masks)

        futures: List[ray.ObjectRef] = []
        for actor, batch_indices in zip(self.teacher_engines, actor_assignments):
            futures.append(
                actor.compute_hidden_states.remote(
                    prompts_ref, input_ids_ref, masks_ref, batch_indices
                )
            )

        # ray.wait loop with periodic timeout warnings (KDFlow parity).
        future_to_idx = {f: i for i, f in enumerate(futures)}
        pending = list(futures)
        raw_results: List[Optional[List[Tuple[int, np.ndarray]]]] = [None] * len(futures)
        start = time.time()
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=120)
            elapsed = time.time() - start
            if elapsed > wait_timeout_s:
                pending_actor_idx = [future_to_idx[f] for f in pending]
                raise RuntimeError(
                    f"[TeacherActorGroup:{method_name}] timed out after {elapsed:.1f}s "
                    f"waiting on actors {pending_actor_idx}; batch_size={len(prompts)}, "
                    f"input_len_min={min(input_lengths) if input_lengths else 0}, "
                    f"input_len_max={max(input_lengths) if input_lengths else 0}"
                )
            if not ready:
                pending_actor_idx = [future_to_idx[f] for f in pending]
                logger.warning(
                    "[TeacherActorGroup] ray.wait still pending after %.1fs "
                    "actors=%s",
                    elapsed,
                    pending_actor_idx,
                )
                continue
            for ref in ready:
                actor_idx = future_to_idx[ref]
                try:
                    raw_results[actor_idx] = ray.get(ref)
                except Exception as exc:
                    sample_indices = actor_assignments[actor_idx]
                    sample_lengths = [input_lengths[i] for i in sample_indices]
                    raise RuntimeError(
                        f"[TeacherActorGroup:{method_name}] actor {actor_idx} failed while "
                        f"computing hidden states; sample_indices={sample_indices}, "
                        f"input_len_min={min(sample_lengths) if sample_lengths else 0}, "
                        f"input_len_max={max(sample_lengths) if sample_lengths else 0}"
                    ) from exc

        # Flatten and re-sort to original order.
        flat: List[Tuple[int, np.ndarray]] = list(
            chain.from_iterable(r for r in raw_results if r is not None)
        )
        flat.sort(key=lambda x: x[0])

        # Sanity check: every sample should be returned exactly once.
        if len(flat) != len(prompts):
            returned_idx = [i for i, _ in flat]
            raise RuntimeError(
                f"[TeacherActorGroup] expected {len(prompts)} hidden-state "
                f"results but got {len(flat)} (indices={returned_idx})."
            )
        return [hs for _, hs in flat]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def sleep(self, tags: Optional[str] = None) -> None:
        ray.get([actor.sleep.remote(tags=tags) for actor in self.teacher_engines])

    def wakeup(self, tags: Optional[str] = None) -> None:
        ray.get([actor.wakeup.remote(tags=tags) for actor in self.teacher_engines])

    def shutdown(self) -> None:
        ray.get(
            [actor.shutdown.remote() for actor in self.teacher_engines + self._worker_actors]
        )
        logger.info("[TeacherActorGroup] shutdown complete")


__all__ = ["TeacherActorGroup"]
