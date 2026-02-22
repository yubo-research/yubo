"""Metric construction and logging helpers for pufferlib PPO."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch.optim as optim

from .specs import _FlatBatch, _RuntimeState, _TrainPlan, _UpdateStats


def _explained_variance(batch: _FlatBatch) -> float:
    y_pred = batch.values.detach().cpu().numpy()
    y_true = batch.returns.detach().cpu().numpy()
    var_y = float(np.var(y_true))
    if var_y == 0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def _as_optional_finite(value: float | None) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return value_f


def _append_metrics_line(metrics_path: Path, payload: dict) -> None:
    from rl.algos import logger as rl_logger

    rl_logger.append_metrics(metrics_path, payload)


def _metric_payload(
    iteration: int,
    plan: _TrainPlan,
    optimizer: optim.Optimizer,
    state: _RuntimeState,
    update_stats: _UpdateStats,
    batch: _FlatBatch,
) -> dict:
    elapsed = time.time() - state.start_time
    sps = int(state.global_step / max(1e-6, elapsed))
    return {
        "iteration": int(iteration),
        "global_step": int(state.global_step),
        "frames_per_batch": int(plan.batch_size),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "approx_kl": float(update_stats.approx_kl),
        "clipfrac_mean": float(update_stats.clipfrac_mean),
        "explained_variance": float(_explained_variance(batch)),
        "episodic_return": _as_optional_finite(state.last_episode_return),
        "eval_return": _as_optional_finite(state.last_eval_return),
        "heldout_return": _as_optional_finite(state.last_heldout_return),
        "best_return": _as_optional_finite(state.best_return),
        "time_seconds": float(elapsed),
        "sps": int(sps),
        "num_iterations": int(plan.num_iterations),
    }


def _log_iteration(config, metric: dict) -> None:
    if int(config.log_interval) <= 0:
        return
    if int(metric["iteration"]) % int(config.log_interval) != 0:
        return
    from rl.algos import logger as rl_logger

    iteration = int(metric["iteration"])
    num_iterations = int(metric["num_iterations"])
    frames_per_batch = int(metric["frames_per_batch"])
    elapsed = float(metric.get("time_seconds", 0.0))
    eval_return = metric.get("eval_return")

    if eval_return is None:
        rl_logger.log_progress_iteration(
            iteration,
            num_iterations,
            frames_per_batch,
            elapsed,
            algo_name="ppo",
            prefix="[rl/ppo/puffer] ",
        )
        return

    rl_logger.log_eval_iteration(
        iteration,
        num_iterations,
        frames_per_batch,
        eval_return=eval_return,
        heldout_return=metric.get("heldout_return"),
        best_return=float(metric.get("best_return") or 0.0),
        algo_metrics={
            "kl": float(metric["approx_kl"]),
            "clipfrac": float(metric["clipfrac_mean"]),
        },
        algo_name="ppo",
        elapsed=elapsed,
        prefix="[rl/ppo/puffer] ",
    )


def _maybe_anneal_lr(config, plan: _TrainPlan, optimizer: optim.Optimizer, iteration: int) -> None:
    if not bool(config.anneal_lr):
        return
    frac = 1.0 - (iteration - 1.0) / float(plan.num_iterations)
    optimizer.param_groups[0]["lr"] = frac * float(config.learning_rate)
