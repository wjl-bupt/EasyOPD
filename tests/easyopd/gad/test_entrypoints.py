"""Existence + shape smoke tests for examples/gad_trainer/."""

import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_DIR = REPO_ROOT / "examples/gad_trainer"


def test_train_script_exists_and_is_executable():
    script = TRAINER_DIR / "train_gad.sh"
    assert script.exists(), "examples/gad_trainer/train_gad.sh missing"
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR, "train_gad.sh must be executable"


def test_train_script_mentions_required_overrides():
    script = (TRAINER_DIR / "train_gad.sh").read_text(encoding="utf-8")
    # The user MUST set discriminator_init_path; the script should make that obvious.
    assert "discriminator_init_path" in script


def test_readme_exists():
    readme = TRAINER_DIR / "README.md"
    assert readme.exists(), "examples/gad_trainer/README.md missing"
    text = readme.read_text(encoding="utf-8")
    assert "discriminator_init_path" in text
    assert "teacher_response" in text
