"""Evaluation package utilities shared across mainline and legacy paths."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
from pathlib import Path

SCHEMA_VERSION = "ropd.eval_package.v1"
GLOBAL_STEP_PATTERN = re.compile(r"global_step_(\d+)")


def _ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_train_step(resume_from_path: str | None) -> int:
    if not resume_from_path:
        return 0
    match = GLOBAL_STEP_PATTERN.search(resume_from_path)
    return int(match.group(1)) if match else 0


def _append_jsonl_with_lock(index_path: Path, entry: dict[str, object]) -> None:
    serialized_entry = json.dumps(entry, ensure_ascii=False) + "\n"
    with index_path.open("a", encoding="utf-8") as index_file:
        fcntl.flock(index_file.fileno(), fcntl.LOCK_EX)
        try:
            index_file.write(serialized_entry)
            index_file.flush()
            os.fsync(index_file.fileno())
        finally:
            fcntl.flock(index_file.fileno(), fcntl.LOCK_UN)


def build_eval_package(
    *,
    package_root: Path | str,
    experiment_name: str,
    source_machine_role: str,
    source_hostname: str,
    resume_from_path: str | None,
    checkpoint_origin: str,
    merged_model_path: str,
    eval_task: str,
    metrics_path: Path | str,
    config_overrides_path: Path | str,
    eval_log_path: Path | str,
    git_commit: str,
    created_at: str,
    schema_version: str = SCHEMA_VERSION,
) -> Path:
    package_root_path = Path(package_root)
    train_step = _extract_train_step(resume_from_path)
    target_dir = (
        package_root_path / experiment_name / f"step_{train_step}" / source_machine_role
    )
    _ensure_path(target_dir)

    shutil.copy2(Path(metrics_path), target_dir / "metrics.json")
    shutil.copy2(Path(config_overrides_path), target_dir / "config_overrides.txt")
    shutil.copy2(Path(eval_log_path), target_dir / "eval.log")

    git_commit_path = target_dir / "git_commit.txt"
    git_commit_path.write_text(git_commit, encoding="utf-8")

    manifest = {
        "schema_version": schema_version,
        "experiment_name": experiment_name,
        "source_machine_role": source_machine_role,
        "source_hostname": source_hostname,
        "train_step": train_step,
        "checkpoint_origin": checkpoint_origin,
        "merged_model_path": merged_model_path,
        "eval_task": eval_task,
        "created_at": created_at,
        "git_commit": git_commit,
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    return target_dir


def archive_eval_package(package_dir: Path | str, hub_root: Path | str) -> Path:
    package_path = Path(package_dir)
    manifest_path = package_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dest_dir = (
        Path(hub_root)
        / "eval-packages"
        / manifest["experiment_name"]
        / f"step_{manifest['train_step']}"
        / manifest["source_machine_role"]
    )

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(package_path, dest_dir)

    index_dir = Path(hub_root) / "index"
    _ensure_path(index_dir)
    index_path = index_dir / "eval_packages.jsonl"
    entry = dict(manifest)
    entry["archived_package_dir"] = str(dest_dir)
    _append_jsonl_with_lock(index_path, entry)

    return dest_dir
