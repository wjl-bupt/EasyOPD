"""GAD must not modify the actor file."""

from pathlib import Path


def test_dp_actor_has_no_gad_markers():
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "verl/workers/actor/dp_actor.py"
    text = target.read_text(encoding="utf-8")
    assert "[EasyOPD:GAD]" not in text, "GAD must not modify dp_actor.py"
