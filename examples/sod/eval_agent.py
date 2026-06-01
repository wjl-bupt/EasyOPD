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

"""SOD Agentic OPD Evaluation Script.

Evaluates a trained SOD agent on tool-integrated reasoning (TIR) tasks,
computing trajectory-level metrics for paper reporting.

Usage::

    # Live evaluation with sandbox environment
    python examples/sod/eval_agent.py --model_path <checkpoint> --env code_execution

    # Mock evaluation with pre-recorded trajectories
    python examples/sod/eval_agent.py --mock --trajectories data/sod_trajectories.json

    # Output structured report
    python examples/sod/eval_agent.py --model_path <checkpoint> --output results.json
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single agent step."""

    step_idx: int
    action: str
    observation: str
    reward: float = 0.0
    is_valid: bool = True
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    latency_sec: float = 0.0


@dataclass
class TrajectoryResult:
    """Result of a complete agent trajectory on one task."""

    task_id: str
    task_description: str = ""
    success: bool = False
    num_steps: int = 0
    invalid_actions: int = 0
    cumulative_reward: float = 0.0
    steps: list[StepResult] = field(default_factory=list)
    wall_time_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalReport:
    """Aggregate evaluation report across all tasks."""

    model_path: str = ""
    env_type: str = ""
    num_tasks: int = 0
    timestamp: str = ""

    # Aggregate metrics
    task_success_rate: float = 0.0
    avg_steps_per_task: float = 0.0
    invalid_action_rate: float = 0.0
    avg_cumulative_reward: float = 0.0
    avg_wall_time_sec: float = 0.0

    # Per-task results
    trajectories: list[TrajectoryResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mock environment (for offline evaluation)
# ---------------------------------------------------------------------------


class MockEnvironment:
    """Mock sandbox environment for offline evaluation.

    Loads pre-recorded trajectories from a JSON file and replays them.
    """

    def __init__(self, trajectories_path: str):
        with open(trajectories_path, "r") as f:
            self.data = json.load(f)
        self.tasks = self.data.get("tasks", [])
        self._current_task_idx = 0

    def get_tasks(self) -> list[dict]:
        """Return list of evaluation tasks."""
        return self.tasks

    def reset(self, task_id: str) -> dict:
        """Reset environment for a new task."""
        for i, task in enumerate(self.tasks):
            if task.get("id") == task_id:
                self._current_task_idx = i
                return {"observation": task.get("initial_observation", ""), "task": task}
        return {"observation": "", "task": {}}

    def step(self, action: str) -> dict:
        """Execute an action and return observation."""
        task = self.tasks[self._current_task_idx]
        steps = task.get("steps", [])

        # Find matching step
        for step in steps:
            if step.get("action") == action:
                return {
                    "observation": step.get("observation", ""),
                    "reward": step.get("reward", 0.0),
                    "done": step.get("done", False),
                    "is_valid": step.get("is_valid", True),
                }

        # Default: invalid action
        return {
            "observation": "Invalid action.",
            "reward": -0.1,
            "done": False,
            "is_valid": False,
        }


# ---------------------------------------------------------------------------
# Agent evaluator
# ---------------------------------------------------------------------------


class AgentEvaluator:
    """Evaluates an agent on TIR tasks."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        env_type: str = "code_execution",
        max_steps: int = 20,
        mock_env: Optional[MockEnvironment] = None,
    ):
        self.model_path = model_path
        self.env_type = env_type
        self.max_steps = max_steps
        self.mock_env = mock_env

    def evaluate_task(self, task: dict) -> TrajectoryResult:
        """Evaluate the agent on a single task.

        Args:
            task: Task dict with 'id', 'description', and optionally 'steps'.

        Returns:
            TrajectoryResult with step-by-step results.
        """
        task_id = task.get("id", "unknown")
        result = TrajectoryResult(
            task_id=task_id,
            task_description=task.get("description", ""),
        )

        start_time = time.perf_counter()

        try:
            if self.mock_env is not None:
                result = self._evaluate_mock(task, result)
            else:
                result = self._evaluate_live(task, result)
        except Exception as e:
            result.error = str(e)
            logger.error("Evaluation failed for task '%s': %s", task_id, e)

        result.wall_time_sec = time.perf_counter() - start_time
        result.num_steps = len(result.steps)
        result.invalid_actions = sum(1 for s in result.steps if not s.is_valid)

        return result

    def _evaluate_mock(self, task: dict, result: TrajectoryResult) -> TrajectoryResult:
        """Evaluate using mock environment (pre-recorded trajectories)."""
        env_state = self.mock_env.reset(task["id"])
        steps_data = task.get("steps", [])

        for i, step_data in enumerate(steps_data):
            if i >= self.max_steps:
                break

            action = step_data.get("action", "")
            env_result = self.mock_env.step(action)

            step = StepResult(
                step_idx=i,
                action=action,
                observation=env_result.get("observation", ""),
                reward=env_result.get("reward", 0.0),
                is_valid=env_result.get("is_valid", True),
                tool_name=step_data.get("tool_name"),
                tool_args=step_data.get("tool_args"),
            )
            result.steps.append(step)
            result.cumulative_reward += step.reward

            if env_result.get("done", False):
                result.success = True
                break

        return result

    def _evaluate_live(self, task: dict, result: TrajectoryResult) -> TrajectoryResult:
        """Evaluate using live sandbox environment.

        Note: This requires a running sandbox service. If not available,
        use --mock mode with pre-recorded trajectories.
        """
        logger.warning(
            "Live evaluation not yet implemented. "
            "Use --mock mode with pre-recorded trajectories, "
            "or implement sandbox integration for env_type='%s'.",
            self.env_type,
        )

        # Placeholder: mark as not evaluated
        result.error = f"Live evaluation for env_type='{self.env_type}' not implemented"
        return result

    def evaluate_all(self, tasks: list[dict]) -> EvalReport:
        """Evaluate the agent on all tasks and compute aggregate metrics.

        Args:
            tasks: List of task dicts.

        Returns:
            EvalReport with aggregate and per-task metrics.
        """
        report = EvalReport(
            model_path=self.model_path or "mock",
            env_type=self.env_type,
            num_tasks=len(tasks),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        for task in tasks:
            trajectory = self.evaluate_task(task)
            report.trajectories.append(trajectory)

        # Compute aggregate metrics
        if report.trajectories:
            n = len(report.trajectories)
            report.task_success_rate = sum(1 for t in report.trajectories if t.success) / n
            report.avg_steps_per_task = sum(t.num_steps for t in report.trajectories) / n
            total_steps = sum(t.num_steps for t in report.trajectories)
            total_invalid = sum(t.invalid_actions for t in report.trajectories)
            report.invalid_action_rate = total_invalid / max(total_steps, 1)
            report.avg_cumulative_reward = sum(t.cumulative_reward for t in report.trajectories) / n
            report.avg_wall_time_sec = sum(t.wall_time_sec for t in report.trajectories) / n

        return report


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_markdown_table(report: EvalReport) -> str:
    """Generate a Markdown summary table from the evaluation report."""
    lines = []
    lines.append(f"## SOD Agent Evaluation Report")
    lines.append(f"")
    lines.append(f"- **Model**: {report.model_path}")
    lines.append(f"- **Environment**: {report.env_type}")
    lines.append(f"- **Tasks**: {report.num_tasks}")
    lines.append(f"- **Timestamp**: {report.timestamp}")
    lines.append(f"")
    lines.append(f"### Aggregate Metrics")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Task Success Rate | {report.task_success_rate:.2%} |")
    lines.append(f"| Avg Steps per Task | {report.avg_steps_per_task:.1f} |")
    lines.append(f"| Invalid Action Rate | {report.invalid_action_rate:.2%} |")
    lines.append(f"| Avg Cumulative Reward | {report.avg_cumulative_reward:.3f} |")
    lines.append(f"| Avg Wall Time (sec) | {report.avg_wall_time_sec:.2f} |")
    lines.append(f"")
    lines.append(f"### Per-Task Results")
    lines.append(f"")
    lines.append(f"| Task ID | Success | Steps | Invalid | Reward | Time (s) |")
    lines.append(f"|---------|---------|-------|---------|--------|----------|")

    for t in report.trajectories:
        lines.append(
            f"| {t.task_id} | {'✓' if t.success else '✗'} | {t.num_steps} "
            f"| {t.invalid_actions} | {t.cumulative_reward:.3f} | {t.wall_time_sec:.2f} |"
        )

    return "\n".join(lines)


def save_report(report: EvalReport, output_path: str) -> None:
    """Save the evaluation report as JSON."""
    # Convert to serializable dict
    data = {
        "model_path": report.model_path,
        "env_type": report.env_type,
        "num_tasks": report.num_tasks,
        "timestamp": report.timestamp,
        "metrics": {
            "task_success_rate": report.task_success_rate,
            "avg_steps_per_task": report.avg_steps_per_task,
            "invalid_action_rate": report.invalid_action_rate,
            "avg_cumulative_reward": report.avg_cumulative_reward,
            "avg_wall_time_sec": report.avg_wall_time_sec,
        },
        "trajectories": [
            {
                "task_id": t.task_id,
                "task_description": t.task_description,
                "success": t.success,
                "num_steps": t.num_steps,
                "invalid_actions": t.invalid_actions,
                "cumulative_reward": t.cumulative_reward,
                "wall_time_sec": t.wall_time_sec,
                "error": t.error,
                "steps": [
                    {
                        "step_idx": s.step_idx,
                        "action": s.action,
                        "observation": s.observation[:200],  # Truncate for readability
                        "reward": s.reward,
                        "is_valid": s.is_valid,
                        "tool_name": s.tool_name,
                    }
                    for s in t.steps
                ],
            }
            for t in report.trajectories
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Report saved to: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOD Agentic OPD Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="code_execution",
        choices=["code_execution", "web_browsing", "tool_calling"],
        help="Sandbox environment type (default: code_execution)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock mode with pre-recorded trajectories",
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        default=None,
        help="Path to pre-recorded trajectories JSON file (for mock mode)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Maximum steps per task (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def create_sample_trajectories() -> list[dict]:
    """Create sample trajectories for demonstration/testing."""
    return [
        {
            "id": "task_001",
            "description": "Write a Python function to compute fibonacci numbers",
            "steps": [
                {
                    "action": "think: I need to write a fibonacci function",
                    "observation": "Planning complete.",
                    "reward": 0.0,
                    "done": False,
                    "is_valid": True,
                    "tool_name": "think",
                },
                {
                    "action": "code: def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
                    "observation": "Code executed successfully. Output: None",
                    "reward": 0.5,
                    "done": False,
                    "is_valid": True,
                    "tool_name": "code",
                    "tool_args": {"code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"},
                },
                {
                    "action": "code: print(fib(10))",
                    "observation": "Code executed successfully. Output: 55",
                    "reward": 1.0,
                    "done": True,
                    "is_valid": True,
                    "tool_name": "code",
                    "tool_args": {"code": "print(fib(10))"},
                },
            ],
        },
        {
            "id": "task_002",
            "description": "Sort a list of numbers using bubble sort",
            "steps": [
                {
                    "action": "invalid_tool: something",
                    "observation": "Invalid action.",
                    "reward": -0.1,
                    "done": False,
                    "is_valid": False,
                    "tool_name": "invalid_tool",
                },
                {
                    "action": "code: def bubble_sort(arr):\n  for i in range(len(arr)):\n    for j in range(len(arr)-1-i):\n      if arr[j]>arr[j+1]: arr[j],arr[j+1]=arr[j+1],arr[j]\n  return arr",
                    "observation": "Code executed successfully.",
                    "reward": 0.5,
                    "done": False,
                    "is_valid": True,
                    "tool_name": "code",
                },
                {
                    "action": "code: print(bubble_sort([5,3,8,1,2]))",
                    "observation": "Code executed successfully. Output: [1, 2, 3, 5, 8]",
                    "reward": 1.0,
                    "done": True,
                    "is_valid": True,
                    "tool_name": "code",
                },
            ],
        },
        {
            "id": "task_003",
            "description": "Calculate the area of a circle with radius 5",
            "steps": [
                {
                    "action": "code: import math; print(math.pi * 5**2)",
                    "observation": "Code executed successfully. Output: 78.53981633974483",
                    "reward": 1.0,
                    "done": True,
                    "is_valid": True,
                    "tool_name": "code",
                },
            ],
        },
    ]


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"\n{'='*60}")
    print(f"  SOD Agentic OPD Evaluation")
    print(f"{'='*60}")

    # Setup environment
    if args.mock:
        if args.trajectories and os.path.exists(args.trajectories):
            mock_env = MockEnvironment(args.trajectories)
            tasks = mock_env.get_tasks()
        else:
            # Use built-in sample trajectories
            print("  Mode: Mock (built-in sample trajectories)")
            tasks = create_sample_trajectories()
            mock_env = MockEnvironment.__new__(MockEnvironment)
            mock_env.tasks = tasks
            mock_env._current_task_idx = 0
            mock_env.data = {"tasks": tasks}
    else:
        mock_env = None
        tasks = []
        if args.model_path is None:
            print("Error: --model_path is required for live evaluation.")
            print("Use --mock for offline evaluation with pre-recorded trajectories.")
            sys.exit(1)

    print(f"  Model: {args.model_path or 'mock'}")
    print(f"  Environment: {args.env}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}\n")

    # Run evaluation
    evaluator = AgentEvaluator(
        model_path=args.model_path,
        env_type=args.env,
        max_steps=args.max_steps,
        mock_env=mock_env,
    )

    report = evaluator.evaluate_all(tasks)

    # Print results
    print(generate_markdown_table(report))

    # Save report
    if args.output:
        save_report(report, args.output)
        print(f"\nJSON report saved to: {args.output}")
    else:
        # Default output path
        default_output = "sod_eval_results.json"
        save_report(report, default_output)
        print(f"\nJSON report saved to: {default_output}")

    print(f"\n{'='*60}")
    print(f"  Evaluation complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
