from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from rl.iter_record import EvalRecordInputs

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


def build_eval_record(ctx: EvalRecordInputs) -> dict[str, Any]:
    from rl.iter_record import IterInputs
    from rl.torchrl_metrics import build_ppo_iter_record

    iteration = int(ctx.timing["iteration"])
    global_step = int(ctx.timing["global_step"])
    t_now = float(time.time()) if ctx.timing.get("now") is None else float(ctx.timing["now"])
    elapsed = float(t_now - float(ctx.started_at))
    frames = int(ctx.timing.get("frames_per_iter") or max(1, global_step // max(1, iteration)))
    dt = float(ctx.timing.get("iter_dt") or max(elapsed / max(1, iteration), 1e-9))
    return build_ppo_iter_record(
        IterInputs(
            iteration=iteration,
            step=global_step,
            frames_per_iter=frames,
            elapsed=elapsed,
            iter_dt=dt,
            metrics=dict(ctx.metrics),
        )
    )


def enrich_ppo_iter_record(
    record: dict[str, Any],
    *,
    rollout_metrics: dict[str, float | None],
    update_stats: dict[str, list[float]],
) -> None:
    update_record_diagnostics(record, rollout_metrics=rollout_metrics, update_stats=update_stats)
    for src, dst in (
        ("loss_objective", "loss_pi"),
        ("loss_critic", "loss_v"),
        ("loss_entropy", "entropy"),
    ):
        value = record.get(src)
        if value is not None:
            record[dst] = float(value)
