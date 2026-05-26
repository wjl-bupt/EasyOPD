"""Tests for GADConfig loading and validation."""

import pytest
from omegaconf import OmegaConf


def _cfg(**gad_overrides):
    base = {
        "gad": {
            "enable": False,
            "discriminator_init_path": None,
        },
        "reward_model": {"enable": False},
    }
    base["gad"].update(gad_overrides)
    return OmegaConf.create(base)


def test_is_gad_enabled_false_by_default():
    from easyopd.methods.gad.config import is_gad_enabled

    cfg = OmegaConf.create({"trainer": {"foo": 1}})
    assert is_gad_enabled(cfg) is False


def test_is_gad_enabled_true_when_flag_set():
    from easyopd.methods.gad.config import is_gad_enabled

    cfg = _cfg(enable=True, discriminator_init_path="/tmp/disc")
    assert is_gad_enabled(cfg) is True


def test_load_returns_dataclass_when_disabled():
    from easyopd.methods.gad.config import GADConfig

    cfg = _cfg(enable=False)
    gad_cfg = GADConfig.load_from_omegaconf(cfg)
    assert gad_cfg.enable is False
    assert gad_cfg.discriminator_init_path is None


def test_load_raises_when_enabled_without_path():
    from easyopd.methods.gad.config import GADConfig, GADConfigError

    cfg = _cfg(enable=True, discriminator_init_path=None)
    with pytest.raises(GADConfigError) as ei:
        GADConfig.load_from_omegaconf(cfg)
    assert "discriminator_init_path" in str(ei.value)


def test_load_collects_all_violations():
    from easyopd.methods.gad.config import GADConfig, GADConfigError

    cfg = OmegaConf.create(
        {
            "gad": {"enable": True, "discriminator_init_path": None},
            "reward_model": {"enable": True},
        }
    )
    with pytest.raises(GADConfigError) as ei:
        GADConfig.load_from_omegaconf(cfg)
    msg = str(ei.value)
    # Both problems must be reported, not just the first one.
    assert "discriminator_init_path" in msg
    assert "reward_model" in msg
    assert msg.count("\n") >= 1  # multiline message


def test_load_succeeds_when_enabled_with_path():
    from easyopd.methods.gad.config import GADConfig

    cfg = _cfg(enable=True, discriminator_init_path="/tmp/disc.ckpt")
    gad_cfg = GADConfig.load_from_omegaconf(cfg)
    assert gad_cfg.enable is True
    assert gad_cfg.discriminator_init_path == "/tmp/disc.ckpt"
