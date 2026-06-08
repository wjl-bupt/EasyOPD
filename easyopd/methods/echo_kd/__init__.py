# Copyright 2026 EasyOPD Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""echo_kd: a minimal demo method demonstrating EasyOPD's hook integration.

This method exists solely as a reference example for "how to add a new
method without touching verl/". It computes a trivial MSE distillation
loss between student and teacher log-probs.
"""

from easyopd.registry import register_method

from .hooks import EchoKDLossHook  # noqa: F401

__all__ = ["EchoKDMethod"]


@register_method("echo_kd")
class EchoKDMethod:
    """Demo method metadata class."""

    name = "echo_kd"
    description = "Echo-KD: minimal MSE distillation demo."
    paper_url = ""
    verl_modified_files = []
