from __future__ import annotations

import dataclasses
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from common.console import (
    PPO_METRICS,
    SAC_METRICS,
    print_iter_record,
    print_iteration_log,
    print_iteration_simple,
    print_run_footer,
    print_run_header,
    register_algo_metrics,
)
from common.console_core import _print_line
from rl.checkpointing import append_jsonl

__all__ = [
    "PPO_METRICS",
    "SAC_METRICS",
    "append_metrics",
    "configure_logging",
    "format_rl_iter_record",
    "infer_algo_name",
    "log_eval_iteration",
    "log_progress_iteration",
    "log_rl_iter",
    "print_rl_iter_record",
    "log_rl_status",
    "log_run_footer",
    "log_run_header",
    "log_run_header_basic",
    "register_algo_metrics",
]


_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
_ITER_LOGGER_NAME = "rl.iter"
_ITER_FIELD_ORDER = (
    "iter",
    "step",
    "elapsed",
    "iter_dt",
    "eval_dt",
    "fps",
    "ret_rollout",
    "ep_ret",
    "ep_len",
    "ret_eval",
    "ret_heldout",
    "ret_best",
    "rew",
    "done_frac",
    "kl",
    "clipfrac",
    "loss",
    "loss_pi",
    "loss_v",
    "entropy",
    "actor",
    "critic",
    "alpha",
    "alpha_loss",
)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a minimal RL-friendly stdlib logging handler once."""
    rl_log = logging.getLogger("rl")
    if not any(handler.get_name() == "yubo.rl" for handler in rl_log.handlers):
        handler = logging.StreamHandler()
        handler.set_name("yubo.rl")
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        rl_log.addHandler(handler)
    rl_log.setLevel(level)
    rl_log.propagate = False


def append_metrics(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)


def _iter_logger() -> logging.Logger:
    log = logging.getLogger(_ITER_LOGGER_NAME)
    if not any(handler.get_name() == "yubo.rl.iter" for handler in log.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name("yubo.rl.iter")
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in record.items() if not _is_missing(value)}


def _format_iter_value(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if key in {"elapsed", "iter_dt", "compile_dt"}:
            return f"{value:.2f}s"
        if key == "fps":
            return f"{value:.0f}"
        return f"{value:.6g}"
    return str(value)


def format_rl_iter_record(record: dict[str, Any]) -> str:
    clean = _clean_record(record)
    ordered_keys = [key for key in _ITER_FIELD_ORDER if key in clean]
    ordered_keys.extend(key for key in clean if key not in _ITER_FIELD_ORDER)
    parts = [f"{key} = {_format_iter_value(key, clean[key])}" for key in ordered_keys]
    return "ITER: " + " ".join(parts)


def infer_algo_name(record: dict[str, Any]) -> str:
    if "kl" in record or "clipfrac" in record or "loss_pi" in record:
        return "ppo"
    if "actor" in record or "critic" in record or "alpha_loss" in record:
        return "sac"
    return "ppo"


def print_rl_iter_record(
    record: dict[str, Any],
    *,
    algo_name: str | None = None,
    prefix: str = "",
) -> None:
    """Print one RL iteration as an aligned table row (standard console)."""
    print_iter_record(record, algo_name=algo_name or infer_algo_name(record), prefix=prefix)


def log_rl_iter(
    record: dict[str, Any],
    *,
    metrics_path: Path | None = None,
    algo_name: str | None = None,
    prefix: str = "",
) -> None:
    clean = _clean_record(record)
    if metrics_path is not None:
        append_metrics(metrics_path, clean)
    print_rl_iter_record(clean, algo_name=algo_name, prefix=prefix)


def log_rl_status(message: str) -> None:
    _print_line(str(message))


def log_run_header(
    algo_name: str,
    config: Any,
    env: Any,
    training: Any,
    runtime: Any,
    *,
    eval_label: str = "eval",
    prefix: str = "",
) -> None:
    print_run_header(algo_name, config, env, training, runtime, eval_label=eval_label, prefix=prefix)


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
    config_obj: Any | None = None,
    eval_label: str = "eval",
    prefix: str = "",
) -> None:
    config_data: dict[str, Any] = {}
    if config_obj is not None:
        if dataclasses.is_dataclass(config_obj):
            config_data.update(dataclasses.asdict(config_obj))
        elif hasattr(config_obj, "__dict__"):
            config_data.update(vars(config_obj))
    config_data.update(
        {
            "env_tag": str(env_tag),
            "seed": int(seed),
            "backbone_name": str(backbone_name),
            "total_timesteps": int(frames_per_batch) * int(num_iterations),
        }
    )
    config = SimpleNamespace(**config_data)
    env = SimpleNamespace(
        env_conf=SimpleNamespace(from_pixels=bool(from_pixels)),
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
    )
    training = SimpleNamespace(frames_per_batch=int(frames_per_batch), num_iterations=int(num_iterations))
    runtime = SimpleNamespace(device=SimpleNamespace(type=str(device_type)))
    print_run_header(algo_name, config, env, training, runtime, eval_label=eval_label, prefix=prefix)


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
    eval_return: float | None = None,
    best_return: float | None = None,
    algo_metrics: dict[str, float] | None = None,
    algo_name: str = "ppo",
    step_override: int | None = None,
    prefix: str = "",
) -> None:
    print_iteration_simple(
        iteration,
        num_iterations,
        frames_per_batch,
        elapsed,
        eval_return=eval_return,
        best_return=best_return,
        algo_metrics=algo_metrics,
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
