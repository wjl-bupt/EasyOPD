"""Registration / metadata structure tests for easyopd.methods.gad."""

from dataclasses import is_dataclass


def test_method_dataclass_is_frozen():
    from easyopd.methods.gad import METHOD, GADMethod

    assert is_dataclass(GADMethod)
    assert isinstance(METHOD, GADMethod)


def test_register_is_callable_and_idempotent():
    from easyopd.methods.gad import register

    register()
    register()  # must not raise on second call
