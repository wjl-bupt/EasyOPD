from __future__ import annotations

from easyopd.methods.ropd.judge import provider, resolver


def test_judge_provider_module_imports_cleanly() -> None:
    assert hasattr(provider, "OpenAICompatibleProvider")
    assert hasattr(resolver, "JudgeProviderResolver")


def test_resolver_class_has_expected_surface() -> None:
    cls = resolver.JudgeProviderResolver
    assert hasattr(cls, "resolve")


def test_judge_provider_module_does_not_expose_legacy_names() -> None:
    for legacy in (
        "BlackOPDProvider",
        "BlackOPDResolver",
        "BlackOPDOpenAICompatibleProvider",
    ):
        assert not hasattr(provider, legacy)
        assert not hasattr(resolver, legacy)


def test_judge_provider_uses_target_side_module_paths() -> None:
    assert provider.__name__ == "easyopd.methods.ropd.judge.provider"
    assert resolver.__name__ == "easyopd.methods.ropd.judge.resolver"
