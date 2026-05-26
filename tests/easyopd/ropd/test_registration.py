from __future__ import annotations


def test_register_ropd_reward_manager_is_idempotent() -> None:
    from easyopd.methods.ropd.reward_manager import (
        ROPDRewardManager,
        register_ropd_reward_manager,
    )
    from verl.workers.reward_manager.registry import (
        REWARD_MANAGER_REGISTRY,
        get_reward_manager_cls,
    )

    REWARD_MANAGER_REGISTRY.pop("ropd", None)
    register_ropd_reward_manager()
    register_ropd_reward_manager()
    assert get_reward_manager_cls("ropd") is ROPDRewardManager


def test_ropd_reward_manager_class_name_is_target_side() -> None:
    from easyopd.methods.ropd.reward_manager import ROPDRewardManager

    assert ROPDRewardManager.__name__ == "ROPDRewardManager"


def test_ropd_reward_manager_extra_info_defaults_use_ropd_keys() -> None:
    from easyopd.methods.ropd.reward_manager import ROPDRewardManager

    defaults = ROPDRewardManager.EXTRA_INFO_DEFAULTS
    for key in (
        "pair_index",
        "group_size",
        "teacher_score",
        "student_score",
        "reward_gap",
        "student_win",
        "fallback_used",
        "judge_error",
    ):
        assert key in defaults, key
