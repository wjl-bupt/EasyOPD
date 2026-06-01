from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TARGET_PATHS = (
    PROJECT_ROOT / "easyopd" / "methods" / "ropd",
    PROJECT_ROOT / "easyopd" / "config" / "ropd",
    PROJECT_ROOT / "examples" / "ropd_trainer",
)

LEGACY_TOKENS = (
    "BLACK_OPD_",
    "shared_rubrics",
    "shared-rubrics",
    "SHARED_RUBRICS_",
)


def _iter_text_files() -> list[Path]:
    found: list[Path] = []
    for root in TARGET_PATHS:
        if not root.exists():
            continue
        if root.is_file():
            found.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {".py", ".yaml", ".yml", ".md", ".sh", ".txt"}:
                found.append(path)
    return found


def test_target_surface_has_no_legacy_tokens() -> None:
    bad: list[tuple[Path, str]] = []
    for path in _iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in LEGACY_TOKENS:
            if token in text:
                bad.append((path, token))
    assert not bad, f"legacy tokens leaked into target surface: {bad}"


def test_target_surface_uses_only_ropd_path_names() -> None:
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8")
        # `black_opd` should never appear in the public surface.
        assert "black_opd" not in text, f"{path} still references black_opd"


def test_no_pair_level_black_opd_paths_in_target() -> None:
    for path in _iter_text_files():
        text = path.read_text(encoding="utf-8")
        assert "pair_level_black_opd" not in text, f"{path} references pair_level_black_opd"
