"""Contract tests for sft.yaml (paper §3.2 recipe guard).

These tests verify that the key SFT fields match the Lightning-OPD
paper §3.2 recipe.  Any modification to these fields should go through
spec review.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest

CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "easyopd", "config", "lightning_opd"
)


@pytest.fixture
def sft_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.load(os.path.join(CONFIG_DIR, "sft.yaml"))


def test_total_training_steps(sft_cfg):
    """Paper §3.2: 3000 SFT steps."""
    assert sft_cfg.trainer.total_training_steps == 3000


def test_learning_rate(sft_cfg):
    """Paper §3.2: lr=8e-5."""
    assert abs(float(sft_cfg.optim.lr) - 8.0e-5) < 1e-10


def test_train_batch_size(sft_cfg):
    """Paper §3.2: effective batch size 256."""
    assert sft_cfg.data.train_batch_size == 256


def test_dynamic_bsz_enabled(sft_cfg):
    """Paper §3.2: dynamic batch sizing with packing."""
    assert sft_cfg.data.use_dynamic_bsz is True


def test_sft_yaml_has_section_3_2_reference(sft_cfg):
    """sft.yaml header should reference §3.2 for traceability."""
    import omegaconf

    # Read raw file to check header comment
    with open(os.path.join(CONFIG_DIR, "sft.yaml")) as f:
        content = f.read()
    assert "§3.2" in content or "3.2" in content
