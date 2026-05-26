from __future__ import annotations

from easyopd.methods.ropd import clients
from easyopd.methods.ropd.judge.config import (
    OpenAIRoleConfig,
    OpenAITransportConfig,
    ProviderLimitsConfig,
    RequestSchedulerConfig,
    merge_nested_mappings,
)


def test_clients_module_uses_neutral_judge_config_names() -> None:
    assert clients.OpenAIRoleConfig is OpenAIRoleConfig
    assert clients.OpenAITransportConfig is OpenAITransportConfig


def test_clients_module_does_not_expose_legacy_names() -> None:
    for legacy_name in (
        "BlackOPDProviderLimitsConfig",
        "BlackOPDRequestSchedulerConfig",
        "BlackOPDDebugConfig",
        "PairLevelClientConfig",
    ):
        assert not hasattr(clients, legacy_name), legacy_name


def test_ropd_client_config_dataclass_is_exposed() -> None:
    assert clients.ROPDClientConfig.__name__ == "ROPDClientConfig"


def test_merge_nested_mappings_deep_copies_values() -> None:
    base = {"provider": {"limits": {"rpm": 1}}, "roles": {"verifier": {"timeout": 3}}}
    override = {"provider": {"limits": {"rpm": 2}}, "roles": {"rubricator": {"timeout": 5}}}

    merged = merge_nested_mappings(base, override)

    assert merged == {
        "provider": {"limits": {"rpm": 2}},
        "roles": {"verifier": {"timeout": 3}, "rubricator": {"timeout": 5}},
    }
    assert base == {"provider": {"limits": {"rpm": 1}}, "roles": {"verifier": {"timeout": 3}}}


def test_request_scheduler_config_is_target_side_only() -> None:
    assert RequestSchedulerConfig.__name__ == "RequestSchedulerConfig"


def test_build_ropd_client_config_helper_is_exported() -> None:
    assert callable(clients.build_ropd_client_config)
