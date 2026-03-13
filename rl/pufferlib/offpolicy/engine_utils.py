from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable


def init_run_artifacts(*, exp_dir: str, config_dict: dict[str, Any]) -> tuple[Path, Path, Any]:
    write_config = importlib.import_module("analysis.data_io").write_config
    checkpoint_manager_cls = importlib.import_module("rl.checkpointing").CheckpointManager
    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_path), config_dict)
    return (
        exp_path,
        exp_path / "metrics.jsonl",
        checkpoint_manager_cls(exp_dir=exp_path),
    )


def init_runtime(
    config: Any,
    *,
    build_env_setup_fn: Callable[[Any], Any],
    seed_everything_fn: Callable[[int], None],
    resolve_device_fn: Callable[[str], Any],
):
    global_seed_for_run = importlib.import_module("rl.core.envs").global_seed_for_run
    env_setup = build_env_setup_fn(config)
    run_seed = global_seed_for_run(int(env_setup.problem_seed))
    seed_everything_fn(int(run_seed))
    return (env_setup, resolve_device_fn(str(config.device)))


def checkpoint_mark_if_due(
    *,
    global_step: int,
    checkpoint_interval_steps: int | None,
    previous_mark: int,
    due_mark_fn: Callable[[int, int | None, int], int | None],
    save_fn: Callable[[], None],
) -> int:
    mark = due_mark_fn(int(global_step), checkpoint_interval_steps, int(previous_mark))
    if mark is None:
        return int(previous_mark)
    save_fn()
    return int(mark)
