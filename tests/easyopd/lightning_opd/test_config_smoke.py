"""Smoke tests for Lightning-OPD config files."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest


CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "easyopd", "config", "lightning_opd"
)


def test_base_yaml_parses():
    """base.yaml should be parseable by OmegaConf."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "base.yaml"))
    assert cfg is not None


def test_base_yaml_adv_estimator():
    """base.yaml should set adv_estimator to on_policy_distillation."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "base.yaml"))
    assert cfg.algorithm.adv_estimator == "on_policy_distillation"


def test_base_yaml_distillation_disabled():
    """base.yaml should have distillation.enabled=False."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "base.yaml"))
    assert cfg.distillation.enabled is False


def test_base_yaml_rollout_n():
    """base.yaml should set rollout.n=1."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "base.yaml"))
    assert cfg.actor_rollout_ref.rollout.n == 1


def test_training_yaml_parses():
    """training.yaml should be parseable."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "training.yaml"))
    assert cfg is not None


def test_data_prep_yaml_parses():
    """data_prep.yaml should be parseable and expose expected keys."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "data_prep.yaml"))
    assert cfg.max_response_len == 4096
    assert cfg.concurrency == 64


def test_sft_yaml_parses():
    """sft.yaml should be parseable by OmegaConf."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(CONFIG_DIR, "sft.yaml"))
    assert cfg is not None
