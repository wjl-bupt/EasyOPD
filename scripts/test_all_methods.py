#!/usr/bin/env python3
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

"""Comprehensive test script for all EasyOPD methods.

Validates that:
1. All 8 methods can be registered and discovered
2. Each method's hooks can be instantiated
3. Each method's config can be loaded
4. HookDispatcher can route to each method
5. Basic hook interface contracts are satisfied

Usage::

    python scripts/test_all_methods.py
    python scripts/test_all_methods.py --verbose
    python scripts/test_all_methods.py --method gkd  # Test single method
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name} ({self.duration:.2f}s) {self.message}"


def run_test(name: str, test_fn) -> TestResult:
    """Run a test function and capture result."""
    start = time.perf_counter()
    try:
        test_fn()
        duration = time.perf_counter() - start
        return TestResult(name, True, duration=duration)
    except Exception as e:
        duration = time.perf_counter() - start
        return TestResult(name, False, message=f"{type(e).__name__}: {e}", duration=duration)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

EXPECTED_METHODS = ["echo_kd", "g_opd", "gad", "gkd", "lightning_opd", "opcd", "opsa", "ropd", "sdpo", "simct", "simple", "sod", "vision_opd"]

# Methods that integrate with verl outside the actor-side HookDispatcher and
# therefore legitimately have no actor LossHook / RolloutHook / RewardHook /
# AlignmentHook / TeacherSidecarHook. GAD repurposes the PPO *critic* as a
# Bradley-Terry discriminator and dispatches via
# `verl/workers/critic/dp_critic.py`, so the actor side stays on verl's
# unmodified PPO loss. Lightning-OPD plugs into verl's ADV_ESTIMATOR_REGISTRY
# plus a data adapter that lifts precomputed teacher log-probabilities from
# parquet into a padded tensor; actor / critic / reward-manager surfaces
# remain on verl's unmodified path.
NON_ACTOR_HOOK_METHODS = {"gad", "lightning_opd"}


def test_auto_discover():
    """Test that auto_discover finds all expected methods."""
    from easyopd.registry import _reset_registry, auto_discover, list_methods

    _reset_registry()
    auto_discover()
    methods = list_methods()
    assert methods == EXPECTED_METHODS, f"Expected {EXPECTED_METHODS}, got {methods}"


def test_from_hparams_all_methods():
    """Test that from_hparams works for all methods."""
    from easyopd import EasyOPD

    for method_name in EXPECTED_METHODS:
        instance = EasyOPD.from_hparams(method_name, auto_resolve_data=False)
        assert instance.method_name == method_name
        assert instance.method_cls is not None
        assert instance.description != ""


def test_hook_dispatch_all_methods():
    """Test that HookDispatcher can build hooks for all methods."""
    from easyopd.hook_dispatch import HookDispatcher
    from easyopd.registry import _reset_registry, auto_discover, get_method

    _reset_registry()
    auto_discover()

    for method_name in EXPECTED_METHODS:
        method_cls = get_method(method_name)
        hooks = HookDispatcher._build_hooks(method_cls, {})
        if method_name in NON_ACTOR_HOOK_METHODS:
            # critic-only / non-actor methods (e.g. GAD): no actor hooks is
            # the contract. They wire into verl outside the HookDispatcher
            # path (e.g. via the critic worker or reward-manager registry).
            continue
        # Every other method should have at least a LossHook
        assert hooks.has_loss, f"Method '{method_name}' should have a LossHook"
        active = hooks.active_hooks()
        assert len(active) >= 1, f"Method '{method_name}' should have at least 1 active hook"


def test_method_metadata():
    """Test that all methods have required metadata attributes."""
    from easyopd.registry import _reset_registry, auto_discover, get_method

    _reset_registry()
    auto_discover()

    required_attrs = ["name", "description", "paper_url", "verl_modified_files"]

    for method_name in EXPECTED_METHODS:
        method_cls = get_method(method_name)
        for attr in required_attrs:
            assert hasattr(method_cls, attr), (
                f"Method '{method_name}' missing required attribute '{attr}'"
            )


def test_config_loading():
    """Test that all method configs can be loaded.

    After the config-layout polish (see `easyopd/config/{gad,ropd,lightning_opd}.yaml`),
    every registered method MUST have a top-level `easyopd/config/{name}.yaml`
    entry that `EasyOPD.from_hparams(name)` can load without an explicit
    `config_path`.  No silent-skip is allowed.
    """
    from easyopd import EasyOPD

    config_dir = PROJECT_ROOT / "easyopd" / "config"

    for method_name in EXPECTED_METHODS:
        config_path = config_dir / f"{method_name}.yaml"
        assert config_path.exists(), (
            f"Method '{method_name}' is missing a top-level default config at "
            f"{config_path}.  Every registered method MUST expose an EasyOPD "
            f"entry config so that `EasyOPD.from_hparams('{method_name}')` "
            f"works without an explicit config_path."
        )
        # 1) Explicit-path load
        instance = EasyOPD.from_hparams(
            method_name, config_path=str(config_path), auto_resolve_data=False
        )
        assert isinstance(instance.config, dict)
        assert len(instance.config) > 0, f"Config for '{method_name}' is empty"
        # 2) Default-path load (no config_path arg) — exercises the
        #    `_CONFIG_DIR / f"{method_name}.yaml"` fallback in `from_hparams`.
        default_instance = EasyOPD.from_hparams(method_name, auto_resolve_data=False)
        assert isinstance(default_instance.config, dict)
        assert len(default_instance.config) > 0, (
            f"Default config for '{method_name}' is empty"
        )


def test_hook_interfaces():
    """Test that hook implementations satisfy the Protocol interfaces."""
    from easyopd.hooks import LossHook, RolloutHook, RewardHook, AlignmentHook, TeacherSidecarHook
    from easyopd.hook_dispatch import HookDispatcher
    from easyopd.registry import _reset_registry, auto_discover, get_method

    _reset_registry()
    auto_discover()

    for method_name in EXPECTED_METHODS:
        method_cls = get_method(method_name)
        hooks = HookDispatcher._build_hooks(method_cls, {})

        if hooks.loss_hook is not None:
            assert isinstance(hooks.loss_hook, LossHook), (
                f"Method '{method_name}' LossHook does not satisfy Protocol"
            )
        if hooks.rollout_hook is not None:
            assert isinstance(hooks.rollout_hook, RolloutHook), (
                f"Method '{method_name}' RolloutHook does not satisfy Protocol"
            )
        if hooks.reward_hook is not None:
            assert isinstance(hooks.reward_hook, RewardHook), (
                f"Method '{method_name}' RewardHook does not satisfy Protocol"
            )
        if hooks.alignment_hook is not None:
            assert isinstance(hooks.alignment_hook, AlignmentHook), (
                f"Method '{method_name}' AlignmentHook does not satisfy Protocol"
            )
        if hooks.teacher_sidecar_hook is not None:
            assert isinstance(hooks.teacher_sidecar_hook, TeacherSidecarHook), (
                f"Method '{method_name}' TeacherSidecarHook does not satisfy Protocol"
            )


def test_method_hook_coverage():
    """Test expected hook coverage for each method."""
    from easyopd.hook_dispatch import HookDispatcher
    from easyopd.registry import _reset_registry, auto_discover, get_method

    _reset_registry()
    auto_discover()

    # Expected hooks per method (based on paper design)
    expected_hooks = {
        "echo_kd": {"loss"},
        "gkd": {"loss"},
        "sod": {"loss", "rollout"},
        "opcd": {"loss", "rollout"},
        "g_opd": {"loss", "reward", "teacher_sidecar"},
        "vision_opd": {"loss", "teacher_sidecar"},
        "sdpo": {"loss", "teacher_sidecar"},
        "opsa": {"loss", "teacher_sidecar"},
        "simple": {"loss", "alignment", "teacher_sidecar"},
        "simct": {"loss"},
        "ropd": {"loss", "reward"},
        # GAD modifies the PPO critic (Bradley-Terry discriminator) instead
        # of the actor; it deliberately exposes no actor-side hook.
        "gad": set(),
        # Lightning-OPD integrates via verl's ADV_ESTIMATOR_REGISTRY and a
        # data adapter for precomputed teacher log-probabilities; it has no
        # actor-side HookDispatcher hook by design.
        "lightning_opd": set(),
    }

    for method_name, expected in expected_hooks.items():
        method_cls = get_method(method_name)
        hooks = HookDispatcher._build_hooks(method_cls, {})
        active = set(hooks.active_hooks())
        assert expected.issubset(active), (
            f"Method '{method_name}': expected hooks {expected}, got {active}"
        )


def test_dispatcher_from_config():
    """Test HookDispatcher.from_config() for all methods."""
    from easyopd.hook_dispatch import HookDispatcher
    from easyopd.registry import _reset_registry, auto_discover

    _reset_registry()
    auto_discover()

    for method_name in EXPECTED_METHODS:
        config = {"easyopd": {"method": {"name": method_name}}}
        dispatcher = HookDispatcher.from_config(config)
        assert dispatcher.enabled, f"Dispatcher for '{method_name}' should be enabled"
        assert dispatcher.method_name == method_name


def test_eval_agent_script():
    """Test that the SOD eval_agent.py script runs in mock mode."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "examples" / "sod" / "eval_agent.py"), "--mock"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"eval_agent.py failed: {result.stderr}"
    assert "Task Success Rate" in result.stdout


def test_data_provider():
    """Test that DataProvider can resolve data for all methods."""
    from easyopd.data_provider import DataProvider, resolve_data_in_config

    provider = DataProvider()

    # Test recipes exist for all methods
    for method_name in EXPECTED_METHODS:
        recipe = provider.get_recipe(method_name)
        assert recipe is not None, f"No recipe for method '{method_name}'"
        assert "dataset" in recipe
        assert "prompt_template" in recipe

    # Test config resolution (using cached data if available)
    config = {
        "method": {"name": "gkd"},
        "data": {
            "dataset": "openai/gsm8k",
            "dataset_split": "train",
            "val_split": "test",
            "prompt_template": "math_qa",
            "prompt_key": "content",
        },
    }
    resolved = resolve_data_in_config(config)
    assert "train_files" in resolved["data"]
    assert len(resolved["data"]["train_files"]) > 0
    # Verify the file exists
    import os
    for f in resolved["data"]["train_files"]:
        assert os.path.exists(f), f"Resolved file does not exist: {f}"


# ---------------------------------------------------------------------------
# Single method test
# ---------------------------------------------------------------------------

def test_single_method(method_name: str):
    """Run all tests for a single method."""
    from easyopd import EasyOPD
    from easyopd.hook_dispatch import HookDispatcher
    from easyopd.registry import _reset_registry, auto_discover, get_method

    _reset_registry()
    auto_discover()

    results = []

    # 1. Registration
    def _test_registration():
        method_cls = get_method(method_name)
        assert method_cls is not None

    results.append(run_test(f"{method_name}/registration", _test_registration))

    # 2. Metadata
    def _test_metadata():
        method_cls = get_method(method_name)
        assert hasattr(method_cls, "name")
        assert hasattr(method_cls, "description")
        assert hasattr(method_cls, "paper_url")
        assert hasattr(method_cls, "verl_modified_files")

    results.append(run_test(f"{method_name}/metadata", _test_metadata))

    # 3. Hook instantiation
    def _test_hooks():
        method_cls = get_method(method_name)
        hooks = HookDispatcher._build_hooks(method_cls, {})
        if method_name in NON_ACTOR_HOOK_METHODS:
            # Non-actor methods (e.g. GAD): no actor LossHook is the contract.
            return
        assert hooks.has_loss

    results.append(run_test(f"{method_name}/hooks", _test_hooks))

    # 4. Config loading
    def _test_config():
        instance = EasyOPD.from_hparams(method_name, auto_resolve_data=False)
        assert instance.config is not None

    results.append(run_test(f"{method_name}/config", _test_config))

    # 5. Dispatcher
    def _test_dispatcher():
        config = {"easyopd": {"method": {"name": method_name}}}
        dispatcher = HookDispatcher.from_config(config)
        assert dispatcher.enabled

    results.append(run_test(f"{method_name}/dispatcher", _test_dispatcher))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test all EasyOPD methods")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--method", "-m", type=str, default=None, help="Test single method")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  EasyOPD Method Validation Suite")
    print(f"{'='*70}\n")

    results: list[TestResult] = []

    if args.method:
        # Test single method
        print(f"Testing method: {args.method}\n")
        results = test_single_method(args.method)
    else:
        # Run all tests
        all_tests = [
            ("Auto-discovery (all 10 methods)", test_auto_discover),
            ("from_hparams() for all methods", test_from_hparams_all_methods),
            ("HookDispatcher builds hooks for all methods", test_hook_dispatch_all_methods),
            ("Method metadata attributes", test_method_metadata),
            ("Config loading for all methods", test_config_loading),
            ("Hook Protocol interface compliance", test_hook_interfaces),
            ("Method hook coverage (paper design)", test_method_hook_coverage),
            ("HookDispatcher.from_config() for all methods", test_dispatcher_from_config),
            ("DataProvider: recipes + auto-resolve", test_data_provider),
            ("SOD eval_agent.py mock mode", test_eval_agent_script),
        ]

        for name, test_fn in all_tests:
            result = run_test(name, test_fn)
            results.append(result)
            if args.verbose or not result.passed:
                print(f"  {result}")

    # Summary
    print(f"\n{'='*70}")
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_time = sum(r.duration for r in results)

    print(f"  Results: {passed} passed, {failed} failed ({total_time:.2f}s total)")

    if failed > 0:
        print(f"\n  Failed tests:")
        for r in results:
            if not r.passed:
                print(f"    {r}")

    print(f"{'='*70}\n")

    # Print data format summary
    if not args.method:
        print_data_format_summary()

    return 0 if failed == 0 else 1


def print_data_format_summary():
    """Print a summary of data format expectations across methods."""
    print(f"\n{'='*70}")
    print(f"  Data Format Summary")
    print(f"{'='*70}\n")
    print("""
  All text-based methods in EasyOPD use verl's unified data format:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Format: Parquet files with HuggingFace datasets compatibility  │
  │                                                                  │
  │  Required columns:                                               │
  │    • prompt_key (default: "content") — chat messages list        │
  │      Format: [{"role": "user", "content": "..."}, ...]          │
  │                                                                  │
  │  Optional columns (method-specific):                             │
  │    • images / videos — for multimodal methods (vision_opd)       │
  │    • bbox_images — teacher-side cropped images (vision_opd)      │
  │    • opd_teacher — teacher logits/responses (g_opd multi-teacher)│
  │    • experience — context distillation data (opcd)               │
  │                                                                  │
  │  Config keys:                                                    │
  │    data.train_files: ["path/to/train.parquet"]                  │
  │    data.val_files: ["path/to/val.parquet"]                      │
  │    data.prompt_key: "content"                                    │
  │    data.max_prompt_length: 1024                                  │
  │    data.truncation: "right" | "error"                           │
  └─────────────────────────────────────────────────────────────────┘

  Method-specific data requirements:
  ┌──────────────┬────────────────────────────────────────────────────┐
  │ Method       │ Additional Data Requirements                       │
  ├──────────────┼────────────────────────────────────────────────────┤
  │ gkd          │ Standard prompt-only (no extra columns)            │
  │ sod          │ Standard prompt-only (agent TIR tasks)             │
  │ simple       │ Standard prompt-only (cross-tokenizer handled      │
  │              │ internally via alignment)                           │
  │ simct        │ Standard prompt-only (span alignment internal)     │
  │ opcd         │ Optional: experience column for context injection  │
  │ g_opd        │ Optional: opd_teacher for multi-teacher distill    │
  │ vision_opd   │ Required: images + bbox_images columns             │
  │ sdpo         │ Standard prompt-only (self-distillation)           │
  │ opsa         │ Standard prompt-only (privileged context injected  │
  │              │ on-the-fly for the teacher; safety-only data)      │
  │ ropd         │ Standard prompt-only; rewards generated by a       │
  │              │ teacher + rubricator + verifier judge triple at    │
  │              │ training time (black-box reward-manager method)    │
  └──────────────┴────────────────────────────────────────────────────┘

  ✅ All text-based methods (gkd, sod, simple, simct, opcd, g_opd, sdpo, opsa, ropd)
     share the SAME base data format: Parquet with chat-template prompts.
     No method-specific data preprocessing is needed.
""")


if __name__ == "__main__":
    sys.exit(main())
