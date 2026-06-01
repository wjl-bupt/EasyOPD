"""Tests for lightning_opd method registration."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_method_name():
    from easyopd.methods.lightning_opd import METHOD

    assert METHOD.name == "lightning_opd"


def test_register_idempotent():
    from easyopd.methods.lightning_opd import register
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    register()
    fn1 = get_adv_estimator_fn("on_policy_distillation")
    register()
    fn2 = get_adv_estimator_fn("on_policy_distillation")
    assert fn1 is fn2


def test_register_makes_adv_estimator_available():
    from easyopd.methods.lightning_opd import register
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    register()
    fn = get_adv_estimator_fn("on_policy_distillation")
    assert callable(fn)


def test_method_verl_modified_files():
    from easyopd.methods.lightning_opd import METHOD

    assert "verl/trainer/ppo/ray_trainer.py" in METHOD.verl_modified_files
    assert "verl/trainer/ppo/core_algos.py" not in METHOD.verl_modified_files
