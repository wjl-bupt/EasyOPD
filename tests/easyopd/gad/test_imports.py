"""Importability smoke tests for easyopd.methods.gad."""


def test_package_imports():
    import easyopd.methods.gad  # noqa: F401


def test_method_metadata_present():
    from easyopd.methods.gad import METHOD

    assert METHOD.name == "gad"
    assert "verl/workers/critic/dp_critic.py" in METHOD.verl_modified_files
    assert METHOD.description  # non-empty
