# Copyright 2026 EasyOPD Contributors
#
# Lightweight unit test for `TeacherActorGroup`'s token-balanced scheduler.
#
# We do NOT spin up real SGLang engines (those need a GPU + model); instead
# we monkey-patch `TeacherRayActor` with a pure-Python stub so the test can
# run on CPU and verifies the *scheduling* logic in isolation.

from __future__ import annotations

import importlib
import os
import sys
from typing import List, Tuple
from unittest import mock

import numpy as np
import pytest

ray = pytest.importorskip("ray")


@pytest.fixture(scope="module")
def ray_local():
    """Make sure Ray is initialized.

    If a cluster is already running (Ray autoconnects via env vars on this
    host), we attach to it; otherwise we start a small local one. Either
    way we never call `ray.shutdown()` so other tests can reuse the cluster.
    """
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True, include_dashboard=False)
        except ValueError:
            # Existing cluster present — connect without resource overrides.
            ray.init(address="auto", ignore_reinit_error=True, include_dashboard=False)
    yield


@ray.remote
class _StubTeacherActor:
    """Drop-in replacement for `TeacherRayActor` that records assignments
    and returns deterministic fake hidden states (a 1-d array whose
    contents encode the original batch index)."""

    def __init__(self, *args, **kwargs):
        self._calls: List[List[int]] = []

    def ready(self) -> bool:
        return True

    def compute_hidden_states(
        self, prompts_ref, input_ids_ref, masks_ref, batch_indices: List[int]
    ) -> List[Tuple[int, np.ndarray]]:
        self._calls.append(list(batch_indices))
        # Resolve refs (they may be ObjectRefs created by the group via ray.put).
        if isinstance(prompts_ref, ray.ObjectRef):
            prompts = ray.get(prompts_ref)
        else:
            prompts = prompts_ref
        if isinstance(masks_ref, ray.ObjectRef):
            masks = ray.get(masks_ref)
        else:
            masks = masks_ref
        out: List[Tuple[int, np.ndarray]] = []
        for i in batch_indices:
            mask = np.asarray(masks[i]).astype(bool)
            n_loss = int(mask.sum())
            # encode the original index in every row so the test can prove
            # ordering was preserved across the network.
            arr = np.full((n_loss, 4), float(i), dtype=np.float32)
            out.append((i, arr))
        return out

    def get_calls(self) -> List[List[int]]:
        return self._calls

    def sleep(self, tags=None):
        return None

    def wakeup(self, tags=None):
        return None

    def shutdown(self):
        return None


def _make_group_with_stub(dp_size: int, tp_size: int = 1, pp_size: int = 1):
    """Construct a TeacherActorGroup whose actors are stubs."""
    from easyopd.methods.simple import teacher_actor as ta_module
    from easyopd.methods.simple import teacher_group as tg_module
    from easyopd.methods.simple.teacher_actor import TeacherActorConfig

    config = TeacherActorConfig(
        model_path="/nonexistent",  # stub doesn't load anything
        tp_size=tp_size,
        pp_size=pp_size,
    )
    with mock.patch.object(tg_module, "TeacherRayActor", _StubTeacherActor):
        group = tg_module.TeacherActorGroup(
            actor_config=config,
            dp_size=dp_size,
            num_gpus_per_actor=0.01,
            num_gpus_per_node=8,
        )
    return group


def test_token_balanced_assignment(ray_local):
    """Samples should be greedily assigned to the actor with fewest tokens."""
    group = _make_group_with_stub(dp_size=2)

    # Token loads: 100, 1, 1, 1, 1, 1
    # Greedy expectation:
    #   step 0 (loads [0,0]):  -> actor 0 (tie -> first), gets 100
    #   step 1 (loads [100,0]): -> actor 1, gets 1 (load=1)
    #   step 2 (loads [100,1]): -> actor 1, gets 1 (load=2)
    #   step 3-5 similarly all go to actor 1
    prompts = [f"sample {i}" for i in range(6)]
    masks = [
        np.ones(100, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
    ]
    out = group.compute_hidden_states_batch(prompts, masks)

    assert len(out) == 6
    # Original-order preserved: arr[i] should be filled with i.
    for i, arr in enumerate(out):
        assert arr.shape[0] == int(masks[i].sum())
        assert np.all(arr == float(i))

    # Verify scheduling decisions.
    actor0_calls = ray.get(group.teacher_engines[0].get_calls.remote())
    actor1_calls = ray.get(group.teacher_engines[1].get_calls.remote())
    assert actor0_calls == [[0]]
    assert actor1_calls == [[1, 2, 3, 4, 5]]

    group.shutdown()


def test_uniform_load_round_robin(ray_local):
    """When all samples have equal token counts the assignment alternates."""
    group = _make_group_with_stub(dp_size=3)

    prompts = [f"s{i}" for i in range(9)]
    masks = [np.ones(5, dtype=bool) for _ in range(9)]
    out = group.compute_hidden_states_batch(prompts, masks)

    assert len(out) == 9
    for i, arr in enumerate(out):
        assert np.all(arr == float(i))

    actor_calls = [
        ray.get(group.teacher_engines[k].get_calls.remote())[0] for k in range(3)
    ]
    # Each of the three actors should have received exactly 3 samples.
    assert sorted([len(c) for c in actor_calls]) == [3, 3, 3]
    # Union of all assigned indices must be {0,...,8} exactly once.
    flat = sorted(sum(actor_calls, []))
    assert flat == list(range(9))

    group.shutdown()


def test_empty_batch_is_noop(ray_local):
    group = _make_group_with_stub(dp_size=2)
    out = group.compute_hidden_states_batch([], [])
    assert out == []
    group.shutdown()


def test_length_mismatch_raises(ray_local):
    group = _make_group_with_stub(dp_size=2)
    with pytest.raises(ValueError):
        group.compute_hidden_states_batch(
            ["a", "b"],
            [np.ones(1, dtype=bool)],
        )
    group.shutdown()
