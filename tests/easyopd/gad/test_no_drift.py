"""Verify the [EasyOPD:GAD] integration points remain exactly where we placed them."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _count_marker_lines(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return sum(1 for line in text.splitlines() if "# ============ [EasyOPD:GAD]" in line)


def test_dp_critic_has_exactly_three_wraps():
    # 3 wraps = 3 opening lines + 3 End lines = 6 marker lines.
    n = _count_marker_lines(REPO_ROOT / "verl/workers/critic/dp_critic.py")
    assert n == 6, f"expected 6 [EasyOPD:GAD] marker lines in dp_critic.py, found {n}"


def test_no_marker_in_ray_trainer():
    n = _count_marker_lines(REPO_ROOT / "verl/trainer/ppo/ray_trainer.py")
    assert n == 0, f"expected 0 [EasyOPD:GAD] markers in ray_trainer.py (plan honored), found {n}"
