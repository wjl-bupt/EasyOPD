"""Entrypoint dry-run tests for Lightning-OPD shell scripts."""

import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")


def _run_dryrun(script_path, extra_env=None):
    """Run a shell script with dry-run flags."""
    env = os.environ.copy()
    env["LIGHTNING_OPD_DRYRUN"] = "true"
    env["LIGHTNING_OPD_SKIP_REPO_DOTENV"] = "true"
    env["LIGHTNING_OPD_PROJECT_ROOT"] = REPO_ROOT
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", script_path],
        capture_output=True, text=True, env=env, timeout=30,
    )
    return result


def test_train_dryrun_exits_zero():
    """train_lightning_opd.sh dry-run should exit 0."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    result = _run_dryrun(script)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_train_dryrun_prints_project_root():
    """Dry-run should print PROJECT_ROOT."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    result = _run_dryrun(script)
    assert "PROJECT_ROOT" in result.stdout


def test_train_dryrun_prints_model_scale():
    """Dry-run should print MODEL_SCALE."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    result = _run_dryrun(script)
    assert "MODEL_SCALE" in result.stdout


def test_train_dryrun_prints_adv_estimator():
    """Dry-run should print adv_estimator=on_policy_distillation."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    result = _run_dryrun(script)
    assert "adv_estimator=on_policy_distillation" in result.stdout


def test_train_dryrun_prints_data_path():
    """Dry-run should print LIGHTNING_OPD_DATA."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    result = _run_dryrun(script)
    assert "LIGHTNING_OPD_DATA" in result.stdout


def test_prepare_data_dryrun_exits_zero():
    """prepare_data.sh dry-run should exit 0."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "tools", "prepare_data.sh")
    result = _run_dryrun(script)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_prepare_data_dryrun_prints_phase_commands():
    """prepare_data.sh dry-run should print Phase 1 and Phase 2 commands."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "tools", "prepare_data.sh")
    result = _run_dryrun(script)
    assert "Phase 1" in result.stdout
    assert "Phase 2" in result.stdout


def test_generate_sft_data_dryrun_prints_real_command():
    """Step 1 dry-run should preview the EasyOPD generation command."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "tools", "generate_sft_data.sh")
    result = _run_dryrun(script)
    assert result.returncode == 0
    assert "easyopd.methods.lightning_opd.data_curation.generate_responses" in result.stdout
    assert "--input-prompts" in result.stdout
    assert "--output-parquet" in result.stdout


def test_collect_rollouts_dryrun_prints_real_command():
    """Step 3 dry-run should preview the EasyOPD generation command."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "tools", "collect_rollouts.sh")
    result = _run_dryrun(script)
    assert result.returncode == 0
    assert "easyopd.methods.lightning_opd.data_curation.generate_responses" in result.stdout
    assert "--input-prompts" in result.stdout
    assert "--output-parquet" in result.stdout


@pytest.mark.parametrize(
    "script_name",
    ["generate_sft_data.sh", "collect_rollouts.sh"],
)
def test_generation_wrappers_fail_when_required_inputs_missing(script_name):
    """Generation wrappers should fail closed instead of no-oping."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "tools", script_name)
    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env={
            **os.environ.copy(),
            "LIGHTNING_OPD_PROJECT_ROOT": REPO_ROOT,
            "LIGHTNING_OPD_SKIP_REPO_DOTENV": "true",
        },
        timeout=30,
    )
    assert result.returncode != 0
    assert "Set" in result.stderr


def test_project_root_override_respected():
    """LIGHTNING_OPD_PROJECT_ROOT override should be respected."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    result = _run_dryrun(script, {"LIGHTNING_OPD_PROJECT_ROOT": "/tmp/test_root"})
    assert "/tmp/test_root" in result.stdout


def test_model_scale_8b():
    """MODEL_SCALE=8b should use TP=4."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    result = _run_dryrun(script, {"MODEL_SCALE": "8b"})
    assert "tensor_model_parallel_size=4" in result.stdout


def test_train_dryrun_prints_teacher_consistency_inputs():
    """Dry-run should print both SFT and OPD teacher paths."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    result = _run_dryrun(
        script,
        {
            "LIGHTNING_OPD_SFT_TEACHER_MODEL": "/models/teacher-a",
            "LIGHTNING_OPD_OPD_TEACHER_MODEL": "/models/teacher-a",
        },
    )
    assert "SFT_TEACHER_MODEL" in result.stdout
    assert "OPD_TEACHER_MODEL" in result.stdout


def test_prepare_data_dryrun_prints_teacher_consistency_inputs():
    """Dry-run should print both prepare-stage teacher identifiers."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "tools", "prepare_data.sh")
    result = _run_dryrun(
        script,
        {
            "LIGHTNING_OPD_SFT_TEACHER": "Qwen/Qwen3-8B",
            "LIGHTNING_OPD_OPD_TEACHER": "Qwen/Qwen3-8B",
        },
    )
    assert "SFT_TEACHER" in result.stdout
    assert "OPD_TEACHER" in result.stdout
    assert "Teacher consistency: OK (same model)" in result.stdout


def test_prepare_data_fails_on_teacher_mismatch():
    """prepare_data.sh should fail closed when SFT/OPD teachers differ."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "tools", "prepare_data.sh")
    env = {
        "LIGHTNING_OPD_PROJECT_ROOT": REPO_ROOT,
        "LIGHTNING_OPD_SKIP_REPO_DOTENV": "true",
        "LIGHTNING_OPD_SFT_TEACHER": "Qwen/Qwen3-8B",
        "LIGHTNING_OPD_OPD_TEACHER": "Qwen/Qwen3-32B",
    }
    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env={**os.environ.copy(), **env},
        timeout=30,
    )
    assert result.returncode != 0
    assert "Teacher consistency check failed" in result.stderr


def test_train_fails_on_teacher_mismatch():
    """train_lightning_opd.sh should fail closed when teachers differ."""
    script = os.path.join(REPO_ROOT, "examples", "lightning_opd_trainer", "train_lightning_opd.sh")
    env = {
        "LIGHTNING_OPD_PROJECT_ROOT": REPO_ROOT,
        "LIGHTNING_OPD_SKIP_REPO_DOTENV": "true",
        "LIGHTNING_OPD_SFT_TEACHER_MODEL": "Qwen/Qwen3-8B",
        "LIGHTNING_OPD_OPD_TEACHER_MODEL": "Qwen/Qwen3-32B",
        "LIGHTNING_OPD_SFT_CHECKPOINT": "/tmp/student",
        "LIGHTNING_OPD_DATA": "/tmp/precomputed.parquet",
    }
    result = subprocess.run(
        ["bash", script],
        capture_output=True,
        text=True,
        env={**os.environ.copy(), **env},
        timeout=30,
    )
    assert result.returncode != 0
    assert "Teacher consistency check failed" in result.stderr
