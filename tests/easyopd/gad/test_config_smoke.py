"""Smoke tests for the GAD Hydra base config."""

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
GAD_BASE = REPO_ROOT / "easyopd/config/gad/base.yaml"


def test_base_yaml_loads():
    cfg = OmegaConf.load(GAD_BASE)
    assert cfg is not None


def test_base_yaml_exposes_gad_node():
    cfg = OmegaConf.load(GAD_BASE)
    assert "gad" in cfg
    # Use .keys() rather than `in` because OmegaConf's __contains__ returns
    # False for keys whose value is the mandatory-missing sentinel (`???`),
    # and `gad.discriminator_init_path` is deliberately `???` in base.yaml.
    gad_keys = set(cfg.gad.keys())
    assert "enable" in gad_keys
    assert "discriminator_init_path" in gad_keys


def test_is_gad_enabled_reads_yaml():
    from easyopd.methods.gad.config import is_gad_enabled

    cfg = OmegaConf.load(GAD_BASE)
    # User must override discriminator_init_path; enabled=true in base.
    assert is_gad_enabled(cfg) is True
