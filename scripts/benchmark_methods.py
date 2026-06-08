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

"""EasyOPD System Efficiency Benchmark.

Measures and compares the system overhead of different OPD methods:
- Training throughput (tokens/sec)
- Peak GPU memory usage
- Wall-clock training time
- Per-component overhead breakdown (teacher forward, alignment, loss computation)

Usage::

    python scripts/benchmark_methods.py --methods gkd,sod,simple --steps 10
    python scripts/benchmark_methods.py --all --steps 5 --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single method."""

    method_name: str
    num_steps: int
    total_wall_time_sec: float = 0.0
    avg_step_time_sec: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    allocated_gpu_memory_mb: float = 0.0

    # Per-component breakdown
    teacher_forward_time_sec: float = 0.0
    alignment_time_sec: float = 0.0
    loss_computation_time_sec: float = 0.0
    reward_computation_time_sec: float = 0.0
    rollout_hook_time_sec: float = 0.0

    # Metadata
    batch_size: int = 0
    seq_length: int = 0
    vocab_size: int = 0
    device: str = ""
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Full benchmark report across all methods."""

    timestamp: str = ""
    results: list[BenchmarkResult] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class MethodBenchmark:
    """Benchmarks a single OPD method's hook overhead."""

    def __init__(
        self,
        method_name: str,
        num_steps: int = 10,
        batch_size: int = 4,
        seq_length: int = 512,
        vocab_size: int = 32000,
    ):
        self.method_name = method_name
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def run(self) -> BenchmarkResult:
        """Run the benchmark and return results."""
        result = BenchmarkResult(
            method_name=self.method_name,
            num_steps=self.num_steps,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            vocab_size=self.vocab_size,
        )

        try:
            import torch

            if torch.cuda.is_available():
                result.device = torch.cuda.get_device_name(0)
                torch.cuda.reset_peak_memory_stats()
            else:
                result.device = "cpu"

            # Import and setup method hooks
            from easyopd.hook_dispatch import HookDispatcher
            from easyopd.registry import auto_discover, get_method

            auto_discover()
            method_cls = get_method(self.method_name)
            hooks = HookDispatcher._build_hooks(method_cls, {})

            # Create synthetic data
            device = "cuda" if torch.cuda.is_available() else "cpu"
            student_logits = torch.randn(
                self.batch_size, self.seq_length, self.vocab_size, device=device
            )
            teacher_logits = torch.randn(
                self.batch_size, self.seq_length, self.vocab_size, device=device
            )
            mask = torch.ones(self.batch_size, self.seq_length, device=device)
            batch = {
                "student_logits": student_logits,
                "teacher_logits": teacher_logits,
                "response_mask": mask,
                "responses": torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length), device=device),
            }

            # Warmup
            if hooks.has_loss:
                for _ in range(2):
                    hooks.loss_hook.compute_loss(
                        student_logits, teacher_logits, mask, config={}, 
                    )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark loss computation
            if hooks.has_loss:
                start = time.perf_counter()
                for _ in range(self.num_steps):
                    hooks.loss_hook.compute_loss(
                        student_logits, teacher_logits, mask, config={},
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result.loss_computation_time_sec = time.perf_counter() - start

            # Benchmark rollout hook
            if hooks.has_rollout:
                start = time.perf_counter()
                for _ in range(self.num_steps):
                    hooks.rollout_hook.on_rollout_end(batch, config={})
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result.rollout_hook_time_sec = time.perf_counter() - start

            # Benchmark reward hook
            if hooks.has_reward:
                start = time.perf_counter()
                for _ in range(self.num_steps):
                    hooks.reward_hook.compute_reward(batch, config={})
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                result.reward_computation_time_sec = time.perf_counter() - start

            # Total wall time
            result.total_wall_time_sec = (
                result.loss_computation_time_sec
                + result.rollout_hook_time_sec
                + result.reward_computation_time_sec
                + result.teacher_forward_time_sec
                + result.alignment_time_sec
            )

            if self.num_steps > 0:
                result.avg_step_time_sec = result.total_wall_time_sec / self.num_steps

            # Throughput
            total_tokens = self.batch_size * self.seq_length * self.num_steps
            if result.total_wall_time_sec > 0:
                result.throughput_tokens_per_sec = total_tokens / result.total_wall_time_sec

            # Memory
            if torch.cuda.is_available():
                result.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                result.allocated_gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        except Exception as e:
            result.error = str(e)
            logger.error("Benchmark failed for method '%s': %s", self.method_name, e)

        return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_markdown_table(results: list[BenchmarkResult]) -> str:
    """Generate a Markdown comparison table from benchmark results."""
    lines = []
    lines.append("| Method | Avg Step (ms) | Throughput (tok/s) | Peak Memory (MB) | Loss (ms) | Rollout (ms) | Error |")
    lines.append("|--------|--------------|-------------------|-----------------|-----------|-------------|-------|")

    for r in results:
        error = r.error[:20] if r.error else "—"
        lines.append(
            f"| {r.method_name:<8} "
            f"| {r.avg_step_time_sec * 1000:.2f} "
            f"| {r.throughput_tokens_per_sec:.0f} "
            f"| {r.peak_gpu_memory_mb:.0f} "
            f"| {r.loss_computation_time_sec / max(r.num_steps, 1) * 1000:.2f} "
            f"| {r.rollout_hook_time_sec / max(r.num_steps, 1) * 1000:.2f} "
            f"| {error} |"
        )

    return "\n".join(lines)


def generate_json_report(report: BenchmarkReport) -> str:
    """Generate JSON report."""
    data = {
        "timestamp": report.timestamp,
        "system_info": report.system_info,
        "results": [asdict(r) for r in report.results],
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_system_info() -> dict[str, Any]:
    """Collect system information."""
    info: dict[str, Any] = {
        "python_version": sys.version,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    except ImportError:
        info["torch_version"] = "not installed"

    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EasyOPD System Efficiency Benchmark")

    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods to benchmark (e.g. gkd,sod,simple)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all available methods",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of benchmark steps per method (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for benchmark (default: 4)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for benchmark (default: 512)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size for benchmark (default: 32000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON report",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Determine methods to benchmark
    if args.all:
        from easyopd import EasyOPD

        methods = EasyOPD.list_methods()
    elif args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        print("Error: Specify --methods or --all")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  EasyOPD System Efficiency Benchmark")
    print(f"{'='*60}")
    print(f"  Methods: {methods}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq length: {args.seq_length}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"{'='*60}\n")

    # Run benchmarks
    results: list[BenchmarkResult] = []
    for method in methods:
        print(f"  Benchmarking: {method}...", end=" ", flush=True)
        benchmark = MethodBenchmark(
            method_name=method,
            num_steps=args.steps,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
        )
        result = benchmark.run()
        results.append(result)

        if result.error:
            print(f"ERROR: {result.error}")
        else:
            print(f"OK ({result.avg_step_time_sec * 1000:.2f} ms/step)")

    # Generate report
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        results=results,
        system_info=get_system_info(),
    )

    # Print markdown table
    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}\n")
    print(generate_markdown_table(results))

    # Save JSON report
    if args.output:
        json_report = generate_json_report(report)
        with open(args.output, "w") as f:
            f.write(json_report)
        print(f"\nJSON report saved to: {args.output}")

    print(f"\n{'='*60}")
    print("  Benchmark complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
