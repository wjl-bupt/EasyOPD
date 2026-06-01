from __future__ import annotations

from easyopd.methods.ropd.judge import config as judge_config


def test_judge_config_exposes_public_helpers() -> None:
    public_names = (
        "coerce_bool",
        "coerce_mapping",
        "coerce_non_negative_int",
        "coerce_optional_float",
        "coerce_optional_int",
        "coerce_optional_positive_int",
        "coerce_optional_string",
        "coerce_positive_int",
        "merge_nested_mappings",
        "build_circuit_breaker_config",
    )
    for name in public_names:
        assert hasattr(judge_config, name), name
        assert name in judge_config.__all__


def test_judge_config_uses_neutral_class_names_only() -> None:
    assert judge_config.JUDGE_STAGES == ("teacher", "rubricator", "verifier")
    for neutral in (
        "ProviderCircuitBreakerConfig",
        "StageBreakerConfigSet",
        "ProviderLimitsConfig",
        "JudgeDebugConfig",
        "RequestSchedulerConfig",
    ):
        cls = getattr(judge_config, neutral)
        assert cls.__name__ == neutral
        assert neutral in judge_config.__all__


def test_judge_config_drops_legacy_aliases() -> None:
    for legacy in (
        "BlackOPDProviderCircuitBreakerConfig",
        "BlackOPDStageBreakerConfigSet",
        "BlackOPDProviderLimitsConfig",
        "BlackOPDDebugConfig",
        "BlackOPDRequestSchedulerConfig",
        "BLACK_OPD_STAGES",
    ):
        assert not hasattr(judge_config, legacy), legacy
        assert legacy not in judge_config.__all__


def test_request_scheduler_config_has_target_side_defaults() -> None:
    cfg = judge_config.RequestSchedulerConfig()
    assert cfg.num_workers is None or isinstance(cfg.num_workers, int)
    assert cfg.max_queue_size is None or isinstance(cfg.max_queue_size, int)
    assert isinstance(cfg.enabled, bool)
