"""Contract test: no legacy slime/ naming in lightning_opd code."""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest

# Files/dirs to scan
SCAN_ROOTS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "easyopd", "methods", "lightning_opd"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "easyopd", "config", "lightning_opd"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "lightning_opd_trainer"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "algo", "lightning_opd.md"),
]

# Patterns that must NOT appear (except as string literals in this test file)
FORBIDDEN_PATTERNS = [
    r"\bslime\.",
    r"\bSLIME_",
    r"\bis_offline_opd\b",
    r"\bis_lightning_opd\b",
    r"\bLightningOpd\b",  # wrong casing
    r"\blightning-opd\b",  # should be lightning_opd
]

# Extensions to scan
SCAN_EXTENSIONS = {".py", ".yaml", ".yml", ".sh", ".md", ".rst"}


def _collect_files():
    files = []
    for root_dir in SCAN_ROOTS:
        root_dir = os.path.normpath(root_dir)
        if not os.path.exists(root_dir):
            continue
        if os.path.isfile(root_dir):
            files.append(root_dir)
        else:
            for dirpath, _, filenames in os.walk(root_dir):
                for fname in filenames:
                    if os.path.splitext(fname)[1] in SCAN_EXTENSIONS:
                        files.append(os.path.join(dirpath, fname))
    return files


def test_no_legacy_names():
    """Scan lightning_opd files for forbidden legacy naming patterns."""
    files = _collect_files()
    violations = []

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue

        for pattern in FORBIDDEN_PATTERNS:
            for match in re.finditer(pattern, content):
                line_num = content[: match.start()].count("\n") + 1
                violations.append(
                    f"{fpath}:{line_num}: found '{match.group()}' (pattern: {pattern})"
                )

    if violations:
        pytest.fail(
            "Legacy naming violations found:\n" + "\n".join(violations)
        )


def test_no_slime_imports():
    """Ensure no slime imports exist in lightning_opd code."""
    import subprocess

    result = subprocess.run(
        ["grep", "-rn", "-E", r"from slime|import slime|slime\.rollout|slime\.backends|slime_plugins",
         os.path.join(os.path.dirname(__file__), "..", "..", "..", "easyopd", "methods", "lightning_opd"),
         os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "lightning_opd_trainer")],
        capture_output=True, text=True
    )
    # grep returns 1 when no match found (which is what we want)
    if result.returncode == 0:
        pytest.fail(f"Found slime imports:\n{result.stdout}")
