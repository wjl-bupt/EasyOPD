from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = PROJECT_ROOT / "examples" / "ropd_trainer" / "train_ropd.sh"


@pytest.fixture(scope="module")
def dryrun_output() -> str:
    env = dict(os.environ)
    env["ROPD_DRYRUN"] = "true"
    env["ROPD_SKIP_REPO_DOTENV"] = "true"
    env["ROPD_PROJECT_ROOT"] = str(PROJECT_ROOT)
    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd="/tmp",
    )
    return result.stdout


def test_train_ropd_dryrun_exits_zero(dryrun_output: str) -> None:
    assert dryrun_output.strip()


def test_train_ropd_dryrun_emits_canonical_command(dryrun_output: str) -> None:
    assert "python3" in dryrun_output
    assert "verl.trainer.main_ppo" in dryrun_output
    assert "reward_model.reward_manager=ropd" in dryrun_output
    assert "+reward_model.reward_kwargs.ropd.provider_resolution.spec_path=" in dryrun_output


def test_train_ropd_dryrun_prints_project_root(dryrun_output: str) -> None:
    assert "PROJECT_ROOT=" in dryrun_output


def test_train_ropd_dryrun_prints_config_template(dryrun_output: str) -> None:
    assert "Config template=easyopd/config/ropd/" in dryrun_output


def test_train_ropd_dryrun_prints_data_root(dryrun_output: str) -> None:
    assert "DATA_ROOT=" in dryrun_output


def test_train_ropd_dryrun_prints_reward_manager_line(dryrun_output: str) -> None:
    assert "Reward manager=ropd" in dryrun_output


def test_train_ropd_dryrun_does_not_leak_legacy_env_names(dryrun_output: str) -> None:
    for legacy in ("BLACK_OPD_", "SHARED_RUBRICS_", "shared_rubric"):
        assert legacy not in dryrun_output, f"unexpected legacy token in dry-run: {legacy}"


def test_train_ropd_supports_ropd_project_root_override(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["ROPD_DRYRUN"] = "true"
    env["ROPD_SKIP_REPO_DOTENV"] = "true"
    env["ROPD_PROJECT_ROOT"] = str(tmp_path)
    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd="/tmp",
    )
    assert f"PROJECT_ROOT={tmp_path}" in result.stdout
