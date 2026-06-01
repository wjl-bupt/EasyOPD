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

"""EasyOPD Diagnostics Framework.

Provides unified metrics collection, anomaly detection, and reporting
for all OPD methods. Each method declares its diagnostic metrics, and
the MetricsCollector automatically gathers, validates, and reports them.

Usage::

    from easyopd.diagnostics import MetricsCollector

    collector = MetricsCollector(method_name="simple")
    collector.collect({"kl_div": 0.5, "alignment_coverage": 0.92})
    collector.report()  # logs to wandb/tensorboard
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric declaration
# ---------------------------------------------------------------------------


@dataclass
class MetricSpec:
    """Specification for a diagnostic metric.

    Attributes:
        name: Metric name (e.g. "kl_div", "alignment_coverage").
        description: Human-readable description.
        unit: Optional unit string (e.g. "nats", "%", "tokens/sec").
        warn_below: If metric falls below this value, emit a warning.
        warn_above: If metric exceeds this value, emit a warning.
        critical_below: If metric falls below this, emit an error-level warning.
        critical_above: If metric exceeds this, emit an error-level warning.
    """

    name: str
    description: str = ""
    unit: str = ""
    warn_below: Optional[float] = None
    warn_above: Optional[float] = None
    critical_below: Optional[float] = None
    critical_above: Optional[float] = None


# ---------------------------------------------------------------------------
# Pre-defined metric specs for each method
# ---------------------------------------------------------------------------

# Common metrics shared across methods
COMMON_METRICS = [
    MetricSpec(
        name="loss",
        description="Total training loss",
        warn_above=100.0,
        critical_above=1000.0,
    ),
    MetricSpec(
        name="grad_norm",
        description="Gradient norm",
        warn_above=10.0,
        critical_above=100.0,
    ),
]

# Method-specific declared metrics
METHOD_METRICS: dict[str, list[MetricSpec]] = {
    "simple": [
        MetricSpec(
            name="overlap_vocab_size",
            description="Number of tokens in overlap vocabulary",
            unit="tokens",
            warn_below=100,
            critical_below=10,
        ),
        MetricSpec(
            name="alignment_coverage",
            description="Fraction of response positions with valid alignment",
            unit="%",
            warn_below=0.3,
            critical_below=0.1,
        ),
        MetricSpec(
            name="cross_tokenizer_kl",
            description="KL divergence on overlap vocabulary",
            unit="nats",
            warn_above=50.0,
        ),
    ],
    "simct": [
        MetricSpec(
            name="span_coverage",
            description="Fraction of tokens covered by span alignment",
            unit="%",
            warn_below=0.3,
            critical_below=0.1,
        ),
        MetricSpec(
            name="span_kl",
            description="KL divergence on span virtual vocabulary",
            unit="nats",
            warn_above=50.0,
        ),
    ],
    "gkd": [
        MetricSpec(
            name="jsd",
            description="Generalized Jensen-Shannon Divergence",
            unit="nats",
            warn_above=20.0,
        ),
        MetricSpec(
            name="on_policy_ratio",
            description="Ratio of on-policy vs off-policy samples",
            unit="ratio",
            warn_below=0.1,
        ),
    ],
    "sod": [
        MetricSpec(
            name="step_coverage",
            description="Fraction of response with identified steps",
            unit="%",
            warn_below=0.2,
        ),
        MetricSpec(
            name="mean_step_weight",
            description="Average step-wise OPD weight",
            unit="weight",
            warn_below=0.1,
            warn_above=2.0,
        ),
        MetricSpec(
            name="num_steps",
            description="Average number of reasoning steps per sample",
            unit="steps",
        ),
    ],
    "g_opd": [
        MetricSpec(
            name="reward_scale",
            description="G-OPD reward scaling factor",
            unit="scale",
        ),
        MetricSpec(
            name="mean_advantage",
            description="Mean G-OPD advantage",
            unit="advantage",
        ),
        MetricSpec(
            name="ref_kl",
            description="KL divergence from reference model",
            unit="nats",
            warn_above=50.0,
        ),
    ],
    "opcd": [
        MetricSpec(
            name="context_kl",
            description="KL between context-conditioned and context-free",
            unit="nats",
            warn_above=50.0,
        ),
        MetricSpec(
            name="experience_length",
            description="Average experience prompt length",
            unit="tokens",
        ),
        MetricSpec(
            name="consolidate_success_rate",
            description="Fraction of samples with successful experience",
            unit="%",
        ),
    ],
    "vision_opd": [
        MetricSpec(
            name="self_distillation_loss",
            description="Self-distillation KL loss",
            unit="nats",
            warn_above=50.0,
        ),
        MetricSpec(
            name="ema_decay",
            description="EMA teacher decay rate",
            unit="rate",
        ),
        MetricSpec(
            name="teacher_image_coverage",
            description="Fraction of samples with teacher images",
            unit="%",
            warn_below=0.5,
        ),
    ],
    "sdpo": [
        MetricSpec(
            name="self_distillation_loss",
            description="SDPO self-distillation loss",
            unit="nats",
            warn_above=50.0,
        ),
        MetricSpec(
            name="demonstration_reward",
            description="Average reward of selected demonstrations",
            unit="reward",
        ),
        MetricSpec(
            name="ema_decay",
            description="EMA teacher decay rate",
            unit="rate",
        ),
    ],
}


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Unified metrics collection and reporting for EasyOPD methods.

    Collects metrics from hook dispatch calls, checks for anomalies,
    and reports to logging backends (wandb, tensorboard, or plain logging).

    Attributes:
        method_name: Name of the active method.
        declared_metrics: List of MetricSpec for the method.
        history: Rolling history of collected metrics.
    """

    def __init__(
        self,
        method_name: str,
        max_history: int = 100,
        report_fn: Optional[Callable[[dict[str, float], int], None]] = None,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            method_name: Name of the OPD method.
            max_history: Maximum number of steps to keep in history.
            report_fn: Optional callback for reporting metrics.
                       Signature: (metrics_dict, global_step) -> None.
                       If None, metrics are only logged.
        """
        self.method_name = method_name
        self.max_history = max_history
        self.report_fn = report_fn
        self._step = 0

        # Get declared metrics for this method
        self.declared_metrics = COMMON_METRICS + METHOD_METRICS.get(method_name, [])
        self._metric_specs: dict[str, MetricSpec] = {
            m.name: m for m in self.declared_metrics
        }

        # Rolling history
        self._history: dict[str, list[float]] = defaultdict(list)
        self._anomaly_counts: dict[str, int] = defaultdict(int)

    @property
    def declared_metric_names(self) -> list[str]:
        """Return names of all declared metrics."""
        return [m.name for m in self.declared_metrics]

    def collect(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
    ) -> list[str]:
        """Collect metrics from a training step.

        Args:
            metrics: Dictionary of metric_name -> value.
            step: Optional global step number. If None, auto-increments.

        Returns:
            List of warning messages generated by anomaly detection.
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        warnings_list: list[str] = []

        for name, value in metrics.items():
            # Store in history
            self._history[name].append(value)
            if len(self._history[name]) > self.max_history:
                self._history[name] = self._history[name][-self.max_history:]

            # Check for anomalies
            anomaly_msg = self._check_anomaly(name, value)
            if anomaly_msg:
                warnings_list.append(anomaly_msg)

        # Report metrics
        if self.report_fn is not None:
            # Prefix metrics with method name
            prefixed = {
                f"easyopd/{self.method_name}/{k}": v
                for k, v in metrics.items()
            }
            self.report_fn(prefixed, self._step)

        return warnings_list

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all collected metrics.

        Returns:
            Dict of metric_name -> {mean, min, max, last, count}.
        """
        summary = {}
        for name, values in self._history.items():
            if values:
                summary[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                    "count": len(values),
                }
        return summary

    def reset(self) -> None:
        """Reset all collected metrics and history."""
        self._history.clear()
        self._anomaly_counts.clear()
        self._step = 0

    def _check_anomaly(self, name: str, value: float) -> Optional[str]:
        """Check if a metric value is anomalous.

        Args:
            name: Metric name.
            value: Metric value.

        Returns:
            Warning message string, or None if no anomaly.
        """
        spec = self._metric_specs.get(name)
        if spec is None:
            return None

        # Critical thresholds (error-level)
        if spec.critical_below is not None and value < spec.critical_below:
            self._anomaly_counts[name] += 1
            msg = (
                f"CRITICAL: {self.method_name}/{name} = {value:.4f} "
                f"is below critical threshold {spec.critical_below} "
                f"({spec.description})"
            )
            logger.error(msg)
            return msg

        if spec.critical_above is not None and value > spec.critical_above:
            self._anomaly_counts[name] += 1
            msg = (
                f"CRITICAL: {self.method_name}/{name} = {value:.4f} "
                f"is above critical threshold {spec.critical_above} "
                f"({spec.description})"
            )
            logger.error(msg)
            return msg

        # Warning thresholds
        if spec.warn_below is not None and value < spec.warn_below:
            self._anomaly_counts[name] += 1
            msg = (
                f"WARNING: {self.method_name}/{name} = {value:.4f} "
                f"is below warning threshold {spec.warn_below} "
                f"({spec.description})"
            )
            logger.warning(msg)
            return msg

        if spec.warn_above is not None and value > spec.warn_above:
            self._anomaly_counts[name] += 1
            msg = (
                f"WARNING: {self.method_name}/{name} = {value:.4f} "
                f"is above warning threshold {spec.warn_above} "
                f"({spec.description})"
            )
            logger.warning(msg)
            return msg

        return None


# ---------------------------------------------------------------------------
# Convenience: create collector for a method
# ---------------------------------------------------------------------------


def create_collector(
    method_name: str,
    report_fn: Optional[Callable] = None,
) -> MetricsCollector:
    """Create a MetricsCollector for the given method.

    Args:
        method_name: Name of the OPD method.
        report_fn: Optional reporting callback.

    Returns:
        Configured MetricsCollector instance.
    """
    return MetricsCollector(method_name=method_name, report_fn=report_fn)


def get_declared_metrics(method_name: str) -> list[MetricSpec]:
    """Get the declared metrics for a method.

    Args:
        method_name: Name of the OPD method.

    Returns:
        List of MetricSpec for the method (including common metrics).
    """
    return COMMON_METRICS + METHOD_METRICS.get(method_name, [])
