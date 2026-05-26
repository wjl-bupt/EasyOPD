from __future__ import annotations

from importlib import resources

import easyopd.methods.ropd.prompt_utils as prompt_utils


PROMPT_RESOURCES = (
    "rubricator.txt",
    "rubricator_cn.txt",
    "verifier.txt",
    "verifier_cn.txt",
    "verifier_skywork.txt",
)


def test_all_five_prompt_resources_are_packaged() -> None:
    for name in PROMPT_RESOURCES:
        contents = (
            resources.files("easyopd.methods.ropd.prompts").joinpath(name).read_text(encoding="utf-8")
        )
        assert contents.strip(), f"prompt resource {name!r} is empty"


def test_load_prompt_template_returns_packaged_text() -> None:
    text = prompt_utils.load_prompt_template("rubricator.txt")
    assert text.strip()


def test_load_prompt_template_raises_on_missing_resource() -> None:
    import pytest

    with pytest.raises(Exception):
        prompt_utils.load_prompt_template("does_not_exist.txt")


def test_load_prompt_template_is_cached() -> None:
    assert (
        prompt_utils.load_prompt_template("rubricator.txt")
        is prompt_utils.load_prompt_template("rubricator.txt")
    )
