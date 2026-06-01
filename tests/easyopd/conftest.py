# Pytest configuration shared by every test under `tests/easyopd/`.
#
# This conftest registers project-specific markers so CI invocations like
# `pytest tests/easyopd/ -m "not gpu"` filter cleanly without producing
# `PytestUnknownMarkWarning` noise.

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: tests that require a CUDA-capable GPU at runtime "
        "(skipped on CPU-only CI runners)",
    )
