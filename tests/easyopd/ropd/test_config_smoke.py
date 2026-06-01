from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "easyopd" / "config" / "ropd"


def test_ropd_config_directory_exists() -> None:
    assert CONFIG_DIR.is_dir()


def test_base_yaml_parses_and_uses_ropd_names() -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    rendered = (CONFIG_DIR / "base.yaml").read_text(encoding="utf-8")
    assert cfg is not None
    assert "shared_rubrics" not in rendered
    assert "BLACK_OPD_" not in rendered
    assert "black_opd" not in rendered
    assert cfg["reward_model"]["reward_manager"] == "ropd"


def test_judge_yaml_parses_and_targets_ropd_reward_kwargs() -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "judge.yaml").read_text(encoding="utf-8"))
    assert "ropd" in cfg["reward_model"]["reward_kwargs"]
    assert "max_group_concurrency" in cfg["reward_model"]["reward_kwargs"]["ropd"]


def test_judge_providers_yaml_uses_ropd_env_names() -> None:
    rendered = (CONFIG_DIR / "judge_providers.yaml").read_text(encoding="utf-8")
    assert "ROPD_VLLM_API_KEY" in rendered
    assert "BLACK_OPD_" not in rendered
    assert "shared_rubrics" not in rendered


def test_sft_yaml_parses() -> None:
    cfg = yaml.safe_load((CONFIG_DIR / "sft.yaml").read_text(encoding="utf-8"))
    assert cfg is not None
