"""RL logging facade: structured metrics + console output.

This module is RL-only. BO logging remains in experiments/common console paths.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from common.console import (
    PPO_METRICS,
    SAC_METRICS,
    print_iteration_log,
    print_iteration_simple,
    print_run_footer,
    print_run_header,
    register_algo_metrics,
)
from rl.checkpointing import append_jsonl

__all__ = [
    "PPO_METRICS",
    "SAC_METRICS",
    "append_metrics",
    "log_eval_iteration",
    "log_progress_iteration",
    "log_run_footer",
    "log_run_header",
    "log_run_header_basic",
    "register_algo_metrics",
]


def append_metrics(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)


def log_run_header(
    algo_name: str,
    config: Any,
    env: Any,
    training: Any,
    runtime: Any,
    *,
    prefix: str = "",
) -> None:
    print_run_header(algo_name, config, env, training, runtime, prefix=prefix)


def log_run_header_basic(
    *,
    algo_name: str,
    env_tag: str,
    seed: int,
    backbone_name: str,
    from_pixels: bool,
    obs_dim: int,
    act_dim: int,
    frames_per_batch: int,
    num_iterations: int,
    device_type: str,
    prefix: str = "",
) -> None:
    config = SimpleNamespace(
        env_tag=str(env_tag),
        seed=int(seed),
        backbone_name=str(backbone_name),
        total_timesteps=int(frames_per_batch) * int(num_iterations),
    )
    env = SimpleNamespace(
        env_conf=SimpleNamespace(from_pixels=bool(from_pixels)),
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
    )
    training = SimpleNamespace(
        frames_per_batch=int(frames_per_batch),
        num_iterations=int(num_iterations),
    )
    runtime = SimpleNamespace(device=SimpleNamespace(type=str(device_type)))
    print_run_header(algo_name, config, env, training, runtime, prefix=prefix)


def log_eval_iteration(
    iteration: int,
    num_iterations: int,
    frames_per_batch: int,
    *,
    eval_return: float | None = None,
    heldout_return: float | None = None,
    best_return: float = 0.0,
    algo_metrics: dict[str, float] | None = None,
    algo_name: str = "ppo",
    elapsed: float = 0.0,
    step_override: int | None = None,
    prefix: str = "",
) -> None:
    print_iteration_log(
        iteration,
        num_iterations,
        frames_per_batch,
        eval_return=eval_return,
        heldout_return=heldout_return,
        best_return=best_return,
        algo_metrics=algo_metrics,
        algo_name=algo_name,
        elapsed=elapsed,
        step_override=step_override,
        prefix=prefix,
    )


def log_progress_iteration(
    iteration: int,
    num_iterations: int,
    frames_per_batch: int,
    elapsed: float,
    *,
    algo_name: str = "ppo",
    step_override: int | None = None,
    prefix: str = "",
) -> None:
    print_iteration_simple(
        iteration,
        num_iterations,
        frames_per_batch,
        elapsed,
        algo_name=algo_name,
        step_override=step_override,
        prefix=prefix,
    )


def log_run_footer(
    best_return: float,
    total_iters_or_steps: int,
    total_time: float,
    *,
    algo_name: str = "ppo",
    step_label: str = "iters",
) -> None:
    print_run_footer(
        best_return,
        total_iters_or_steps,
        total_time,
        algo_name=algo_name,
        step_label=step_label,
    )
