# Copyright 2026 EasyOPD Contributors

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from easyopd.methods.simct.losses import register_simct_loss
from verl.trainer.distillation.losses import DISTILLATION_SETTINGS_REGISTRY


ROOT = Path(__file__).resolve().parents[3]


def test_simct_config_template_has_minimal_method_defaults():
    config_path = ROOT / "easyopd" / "config" / "simct.yaml"
    cfg = yaml.safe_load(config_path.read_text())

    loss_cfg = cfg["distillation"]["distillation_loss"]
    assert loss_cfg["loss_mode"] == "simct"
    assert loss_cfg["cross_tokenizer_kl_direction"] == "reverse"
    assert loss_cfg["use_policy_gradient"] is False
    assert loss_cfg["use_task_rewards"] is False


def test_simct_smoke_registration_without_teacher_engine():
    register_simct_loss()
    assert DISTILLATION_SETTINGS_REGISTRY["simct"].use_cross_tokenizer is True
    assert DISTILLATION_SETTINGS_REGISTRY["span_ctkd"].use_cross_tokenizer is True
