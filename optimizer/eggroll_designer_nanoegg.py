from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from optimizer.datum import Datum
from optimizer.designer_errors import NoSuchDesignerError
from optimizer.optimizer_types import IterateResult
from optimizer.trajectory import Trajectory


@dataclass
class _PopulationResult:
    directions: list[np.ndarray]
    raw_scores: list[float]
    dt_eval: float


def iterate_nanoegg(designer, _data, num_arms: int, *, telemetry=None) -> IterateResult:
    _ = _data
    population = _check_population(designer, int(num_arms))
    result = _collect_population(designer, population)
    prop_dt = _apply_update(designer, result.directions, result.raw_scores, population)
    datum, policy_eval_dt = _evaluate_updated_policy(designer, population)
    update_best_and_telemetry(designer, datum, prop_dt, telemetry)
    designer._epoch += 1
    return IterateResult(
        data=[datum],
        dt_prop=float(prop_dt),
        dt_eval=float(result.dt_eval + policy_eval_dt),
    )


def direction_seed(designer, pair_idx: int) -> int:
    seed = (0 if getattr(designer._policy, "problem_seed", None) is None else int(designer._policy.problem_seed)) + int(pair_idx)
    return (int(designer._epoch) << 16) ^ int(seed)


def standardize_scores(designer, raw_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if designer._rank_transform:
        ranks = np.argsort(np.argsort(scores)).astype(np.float64)
        scores = ranks / max(float(scores.size - 1), 1.0)
    centered = _center_scores(scores, int(designer._group_size))
    denom = float(np.sqrt(np.var(scores) + 1e-5))
    return centered / denom


def _check_population(designer, population: int) -> int:
    if population < 2 or population % 2 != 0:
        raise NoSuchDesignerError("EggRoll requires an even population >= 2 because EggRoll uses mirrored thread pairs.")
    if designer._noise_reuse < 0:
        raise NoSuchDesignerError("EggRoll option 'noise_reuse' must be >= 0.")
    return population


def _collect_population(designer, population: int) -> _PopulationResult:
    t_eval = time.time()
    directions: list[np.ndarray] = []
    raw_scores: list[float] = []
    pending_x: list[np.ndarray] = []
    pending_start = 0
    next_log_at = max(256, int(designer._nanoegg_batch_size))
    _log(
        designer,
        f"iter={designer._epoch} evaluating population={population} dim={designer._x.size} batch_size={designer._nanoegg_batch_size}",
    )

    for pair_idx in range(population // 2):
        direction = _sample_direction(designer, pair_idx)
        directions.append(direction)
        for sign in (1.0, -1.0):
            pending_x.append(designer._x + sign * designer._current_sigma() * direction)
            if len(pending_x) >= designer._nanoegg_batch_size:
                pending_start, next_log_at = _flush_pending(
                    designer,
                    pending_x,
                    raw_scores,
                    pending_start=pending_start,
                    next_log_at=next_log_at,
                    population=population,
                    t_eval=t_eval,
                )
    _flush_pending(
        designer,
        pending_x,
        raw_scores,
        pending_start=pending_start,
        next_log_at=next_log_at,
        population=population,
        t_eval=t_eval,
    )
    return _PopulationResult(directions=directions, raw_scores=raw_scores, dt_eval=time.time() - t_eval)


def _apply_update(designer, directions: list[np.ndarray], raw_scores: list[float], population: int) -> float:
    t_prop = time.time()
    grad = _mirrored_gradient(designer, directions, raw_scores, population)
    x_j = designer._objective.jnp.asarray(designer._x, dtype=designer._objective.jnp.float32)
    grad_j = designer._objective.jnp.asarray(-np.sqrt(float(population)) * grad, dtype=designer._objective.jnp.float32)
    updates, designer._nanoegg_opt_state = designer._nanoegg_tx.update(grad_j, designer._nanoegg_opt_state, x_j)
    updated_x = designer._nanoegg_apply_updates(x_j, updates)
    designer._x = np.asarray(designer._objective.jax.block_until_ready(updated_x), dtype=np.float64)
    return time.time() - t_prop


def _evaluate_updated_policy(designer, population: int):
    t_eval = time.time()
    eval_mean, eval_se = designer._objective.evaluate(designer._x, seed=(designer._epoch + 1) * 1_000_003)
    policy = designer._policy.with_snapshot(designer._objective.make_policy(designer._x))
    datum = Datum(
        designer,
        policy,
        None,
        Trajectory(
            rreturn=float(eval_mean),
            states=np.empty((0,)),
            actions=np.empty((0,)),
            rreturn_se=float(eval_se),
            num_steps=int((population + designer._num_envs) * designer._steps_per_episode),
        ),
    )
    return datum, time.time() - t_eval


def update_best_and_telemetry(designer, datum: Datum, prop_dt: float, telemetry) -> None:
    if designer._best_datum is None or datum.trajectory.get_decision_rreturn() > designer._best_datum.trajectory.get_decision_rreturn():
        designer._best_datum = datum
    if telemetry is not None:
        telemetry.set_dt_fit(prop_dt)
        telemetry.set_dt_select(0.0)


def _sample_direction(designer, pair_idx: int) -> np.ndarray:
    direction = designer._objective.sample_eggroll_noiser_noise(
        designer._x,
        seed=direction_seed(designer, pair_idx),
        noiser_name="eggroll",
        rank=designer._rank,
        group_size=designer._group_size,
        freeze_nonlora=designer._freeze_nonlora,
    )
    return np.asarray(direction, dtype=np.float64).reshape(designer._x.shape)


def _flush_pending(
    designer,
    pending_x: list[np.ndarray],
    raw_scores: list[float],
    *,
    pending_start: int,
    next_log_at: int,
    population: int,
    t_eval: float,
):
    if not pending_x:
        return pending_start, next_log_at
    batch = np.asarray(pending_x, dtype=np.float64)
    mus, _ses = designer._objective.evaluate_many(batch, seed=(designer._epoch * population) + pending_start)
    raw_scores.extend(float(v) for v in np.asarray(mus, dtype=np.float64).reshape(-1))
    done = len(raw_scores)
    pending_x.clear()
    return done, _maybe_log_progress(designer, done, next_log_at, population, t_eval)


def _maybe_log_progress(designer, done: int, next_log_at: int, population: int, t_eval: float) -> int:
    if done < next_log_at and done != population:
        return next_log_at
    _log(
        designer,
        f"iter={designer._epoch} evaluated={done}/{population} elapsed={time.time() - t_eval:.1f}s",
    )
    while next_log_at <= done:
        next_log_at += max(256, int(designer._nanoegg_batch_size))
    return next_log_at


def _mirrored_gradient(designer, directions: list[np.ndarray], raw_scores: list[float], population: int) -> np.ndarray:
    fitnesses = standardize_scores(designer, np.asarray(raw_scores, dtype=np.float64))
    grad = np.zeros_like(designer._x, dtype=np.float64)
    for pair_idx, direction in enumerate(directions):
        grad += (float(fitnesses[2 * pair_idx]) - float(fitnesses[(2 * pair_idx) + 1])) * direction
    return grad / max(float(population), 1.0)


def _center_scores(scores: np.ndarray, group_size: int) -> np.ndarray:
    if group_size < 0:
        raise NoSuchDesignerError("EggRoll option 'group_size' must be >= 0.")
    if group_size == 0:
        return scores - float(np.mean(scores))
    if scores.size % group_size != 0:
        raise NoSuchDesignerError("EggRoll option 'group_size' must divide the population.")
    grouped = scores.reshape((-1, group_size))
    return (grouped - np.mean(grouped, axis=1, keepdims=True)).reshape(-1)


def _log(designer, message: str) -> None:
    print(f"NANOEGG_EGGROLL: {message}", flush=True)
