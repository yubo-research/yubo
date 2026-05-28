from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from rl.core.progress import steps_per_second

_LOGGER = logging.getLogger(__name__)


def finite_mean(values: list[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if int(finite.size) else None


def build_algo_metrics(
    *,
    approx_kls: list[float],
    clipfracs: list[float],
) -> dict[str, float | None]:
    return {
        "kl": finite_mean(approx_kls),
        "clipfrac": finite_mean(clipfracs),
    }


def update_record_diagnostics(
    record: dict,
    *,
    rollout_metrics: dict[str, float | None],
    update_stats: dict[str, list[float]],
) -> None:
    record["nonfinite_reward_fraction"] = rollout_metrics.get("nonfinite_reward_fraction")
    record["nonfinite_gae_fraction"] = finite_mean(update_stats.get("nonfinite_gae_fractions", []))
    record["skipped_update_fraction"] = finite_mean(update_stats.get("skipped_updates", []))
    record["ess"] = finite_mean(update_stats.get("ess_values", []))
    record["loss_objective"] = finite_mean(update_stats.get("loss_objective", []))
    record["loss_critic"] = finite_mean(update_stats.get("loss_critic", []))
    record["loss_entropy"] = finite_mean(update_stats.get("loss_entropy", []))
    record["grad_norm"] = finite_mean(update_stats.get("grad_norm", []))


def log_record_diagnostics(record: dict, *, iteration: int) -> None:
    nonfinite_reward = float(record.get("nonfinite_reward_fraction") or 0.0)
    nonfinite_gae = float(record.get("nonfinite_gae_fraction") or 0.0)
    skipped_update = float(record.get("skipped_update_fraction") or 0.0)
    ess = record.get("ess")
    if nonfinite_reward > 0.0 or skipped_update > 0.0:
        _LOGGER.warning(
            "ppo diagnostics iteration=%s nonfinite_reward_fraction=%.4g nonfinite_gae_fraction=%.4g skipped_update_fraction=%.4g ess=%s",
            int(iteration),
            nonfinite_reward,
            nonfinite_gae,
            skipped_update,
            "-" if ess is None else f"{float(ess):.4g}",
        )
        return
    if nonfinite_gae > 0.0:
        _LOGGER.debug(
            "ppo diagnostics iteration=%s nonfinite_gae_fraction=%.4g ess=%s",
            int(iteration),
            nonfinite_gae,
            "-" if ess is None else f"{float(ess):.4g}",
        )


def build_eval_record(
    *,
    iteration: int,
    global_step: int,
    eval_return: float | None,
    heldout_return: float | None,
    best_return: float | None,
    approx_kl: float | None,
    clipfrac: float | None,
    started_at: float,
    rollout_reward: float | None = None,
    rollout_return: float | None = None,
    rollout_length: float | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    t_now = float(time.time()) if now is None else float(now)
    elapsed = float(t_now - float(started_at))
    return {
        "iteration": int(iteration),
        "global_step": int(global_step),
        "eval_return": eval_return,
        "heldout_return": heldout_return,
        "best_return": best_return,
        "rollout_reward": rollout_reward,
        "rollout_return": rollout_return,
        "rollout_length": rollout_length,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
        "time_seconds": elapsed,
        "steps_per_second": float(steps_per_second(int(global_step), float(started_at), now=t_now)),
    }
