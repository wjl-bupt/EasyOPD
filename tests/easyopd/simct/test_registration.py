# Copyright 2026 EasyOPD Contributors

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from easyopd.methods.simple.losses import register_simple_loss
from easyopd.methods.simct.losses import register_simct_loss
from verl.trainer.distillation.losses import DISTILLATION_SETTINGS_REGISTRY


def test_simple_and_simct_registrations_coexist():
    register_simple_loss()
    register_simct_loss()

    assert "simple" in DISTILLATION_SETTINGS_REGISTRY
    assert "simct" in DISTILLATION_SETTINGS_REGISTRY
    assert "span_ctkd" in DISTILLATION_SETTINGS_REGISTRY
    assert DISTILLATION_SETTINGS_REGISTRY["simple"].use_cross_tokenizer is True
    assert DISTILLATION_SETTINGS_REGISTRY["simct"].use_cross_tokenizer is True
    assert DISTILLATION_SETTINGS_REGISTRY["span_ctkd"].use_cross_tokenizer is True
