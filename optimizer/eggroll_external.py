from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from optimizer.datum import Datum
from optimizer.designer_errors import NoSuchDesignerError
from optimizer.eggroll_designer_nanoegg import _check_population, update_best_and_telemetry
from optimizer.eggroll_options import eggroll_bool as _as_bool
from optimizer.eggroll_options import unit_decay as _as_unit_decay
from optimizer.optimizer_types import IterateResult
from optimizer.trajectory import Trajectory


@dataclass
class _ExternalPopulation:
    directions: list[np.ndarray]
    scores: list[float]
    num_steps: int
    dt_eval: float


class _AdamMax:
    def __init__(self, dim: int, *, b1: float, b2: float, weight_decay: float) -> None:
        self._m = np.zeros(int(dim), dtype=np.float64)
        self._v = np.zeros(int(dim), dtype=np.float64)
        self._t = 0
        self._b1 = float(b1)
        self._b2 = float(b2)
        self._weight_decay = float(weight_decay)

    def step(self, x: np.ndarray, grad: np.ndarray, *, lr: float) -> np.ndarray:
        self._t += 1
        g = np.asarray(grad, dtype=np.float64)
        if self._weight_decay:
            g = g - self._weight_decay * np.asarray(x, dtype=np.float64)
        self._m = self._b1 * self._m + (1.0 - self._b1) * g
        self._v = self._b2 * self._v + (1.0 - self._b2) * np.square(g)
        m_hat = self._m / (1.0 - self._b1**self._t)
        v_hat = self._v / (1.0 - self._b2**self._t)
        return np.asarray(x, dtype=np.float64) + float(lr) * m_hat / (np.sqrt(v_hat) + 1e-8)


def init_external(designer, policy, env_conf, cfg) -> None:
    from problems.isaaclab_env_adapters import is_isaaclab_env_tag
    from problems.isaaclab_score import IsaacLabScore, active_vector_slots

    env_name = str(getattr(env_conf, "env_name", ""))
    if not (is_isaaclab_env_tag(env_name) or env_name.startswith("dm_control/")):
        raise NoSuchDesignerError(f"Designer 'eggroll' external scoring only supports isaaclab: and dm_control/ envs, got {env_name!r}.")
    objective = IsaacLabScore(env_conf, policy, episodes=int(cfg.num_envs), steps_per_episode=int(cfg.steps))
    designer._is_external = True
    designer._policy = policy
    designer._objective = objective
    designer._active_vector_slots = active_vector_slots
    designer._state.x = objective.x0
    designer._sigma = float(cfg.sigma)
    designer._sigma_decay = _as_unit_decay(cfg.sigma_decay, name="sigma_decay")
    designer._lr = float(cfg.lr)
    designer._lr_decay = _as_unit_decay(cfg.lr_decay, name="lr_decay")
    designer._rank_transform = _as_bool(cfg.rank_transform, name="rank_transform")
    designer._rank = int(cfg.rank)
    designer._group_size = int(cfg.group_size)
    designer._freeze_nonlora = _as_bool(cfg.freeze_nonlora, name="freeze_nonlora")
    designer._noise_reuse = int(cfg.noise_reuse)
    designer._external_batch_size = int(cfg.batch_size)
    designer._steps_per_episode = int(cfg.steps)
    designer._num_envs = int(cfg.num_envs)
    designer._state.best_datum = None
    designer._state.epoch = 0
    if str(cfg.optax) not in {"adam", "adamw"}:
        raise NoSuchDesignerError("EggRoll external scoring currently supports optax='adam' or optax='adamw'.")
    designer._adam = _AdamMax(
        objective.dim,
        b1=float(cfg.b1),
        b2=float(cfg.b2),
        weight_decay=0.0 if str(cfg.optax) == "adam" else float(cfg.weight_decay),
    )
    _validate_external(designer)


def iterate_external(designer, state, _data, num_arms: int, *, telemetry=None) -> IterateResult:
    _ = _data
    population = _check_population(designer, int(num_arms))
    result = _evaluate_population(designer, state, population)
    prop_dt = _apply_update(designer, state, result.directions, result.scores, population)
    datum, eval_dt, eval_steps = _evaluate_current(designer, state, population)
    update_best_and_telemetry(designer, state, datum, prop_dt, telemetry)
    state.epoch += 1
    return IterateResult(
        data=[datum],
        dt_prop=float(prop_dt),
        dt_eval=float(result.dt_eval + eval_dt),
    )._replace(data=[_with_steps(datum, result.num_steps + eval_steps)])


def _validate_external(designer) -> None:
    if designer._sigma <= 0.0:
        raise NoSuchDesignerError("EggRoll option 'sigma' must be > 0.")
    if designer._lr <= 0.0:
        raise NoSuchDesignerError("EggRoll option 'lr' must be > 0.")
    if designer._rank < 1:
        raise NoSuchDesignerError("EggRoll option 'rank' must be >= 1.")
    if designer._group_size < 0:
        raise NoSuchDesignerError("EggRoll option 'group_size' must be >= 0.")
    if designer._external_batch_size < 1:
        raise NoSuchDesignerError("EggRoll option 'batch_size' must be >= 1.")


def _evaluate_population(designer, state, population: int) -> _ExternalPopulation:
    t0 = time.time()
    directions: list[np.ndarray] = []
    scores: list[float] = []
    total_steps = 0
    pending: list[np.ndarray] = []
    next_log_at = max(16, int(designer._external_batch_size))
    _log(designer, f"iter={state.epoch} evaluating population={population} dim={state.x.size}")
    for pair_idx in range(population // 2):
        direction = _sample_direction(designer, state, pair_idx)
        directions.append(direction)
        sigma = designer._current_sigma()
        pending.append(state.x + sigma * direction)
        pending.append(state.x - sigma * direction)
        if len(pending) >= int(designer._external_batch_size):
            next_log_at, total_steps = _flush(designer, state, pending, scores, population, next_log_at, total_steps, t0)
    next_log_at, total_steps = _flush(designer, state, pending, scores, population, next_log_at, total_steps, t0)
    _ = next_log_at
    return _ExternalPopulation(directions=directions, scores=scores, num_steps=int(total_steps), dt_eval=time.time() - t0)


def _flush(designer, state, pending: list[np.ndarray], scores: list[float], population: int, next_log_at: int, total_steps: int, t0: float):
    if not pending:
        return next_log_at, total_steps
    start = len(scores)
    _log(designer, f"iter={state.epoch} scoring candidates {start + 1}-{start + len(pending)}/{population}")
    mus, _ses = designer._objective.evaluate_many(np.asarray(pending, dtype=np.float64), seed=int(state.epoch * population) + start)
    scores.extend(float(v) for v in np.asarray(mus, dtype=np.float64).reshape(-1))
    total_steps += int(getattr(designer._objective, "last_num_steps", 0))
    pending.clear()
    done = len(scores)
    if done >= next_log_at or done == population:
        _log(designer, f"iter={state.epoch} evaluated={done}/{population} elapsed={time.time() - t0:.1f}s")
        while next_log_at <= done:
            next_log_at += max(16, int(designer._external_batch_size))
    return next_log_at, total_steps


def _apply_update(designer, state, directions: list[np.ndarray], raw_scores: list[float], population: int) -> float:
    t0 = time.time()
    grad = _mirrored_gradient(designer, state, directions, raw_scores, population)
    lr = designer._lr * (designer._lr_decay**state.epoch)
    state.x = np.clip(designer._adam.step(state.x, grad, lr=lr), -1.0, 1.0)
    return time.time() - t0


def _evaluate_current(designer, state, population: int):
    t0 = time.time()
    mu, se = _evaluate_current_reusing_active_vector_env(designer, state)
    steps = int(getattr(designer._objective, "last_num_steps", 0))
    datum = Datum(
        designer,
        designer._objective.make_policy(state.x),
        None,
        Trajectory(
            rreturn=float(mu),
            states=np.empty((0,)),
            actions=np.empty((0,)),
            rreturn_se=float(se),
            num_steps=int((population * designer._steps_per_episode) + steps),
        ),
    )
    return datum, time.time() - t0, steps


def _evaluate_current_reusing_active_vector_env(designer, state) -> tuple[float, float]:
    active_slots = designer._active_vector_slots(designer._objective)
    episodes = max(1, int(designer._objective.num_envs))
    repeat = 1 if active_slots is None else max(1, int(active_slots) // episodes)
    seed = (state.epoch + 1) * 1_000_003
    if repeat <= 1:
        return designer._objective.evaluate(state.x, seed=seed)
    xs = np.repeat(np.asarray(state.x, dtype=np.float64).reshape(1, -1), repeat, axis=0)
    mus, ses = designer._objective.evaluate_many(xs, seed=seed)
    return float(np.mean(mus)), float(np.sqrt(np.mean(np.square(ses))))


def _with_steps(datum: Datum, num_steps: int) -> Datum:
    datum.trajectory.num_steps = int(num_steps)
    return datum


def _sample_direction(designer, state, pair_idx: int) -> np.ndarray:
    base_seed = 0 if getattr(designer._policy, "problem_seed", None) is None else int(designer._policy.problem_seed)
    seed = (int(state.epoch) << 16) ^ (base_seed + int(pair_idx))
    direction = designer._objective.sample_eggroll_noiser_noise(
        state.x,
        seed=seed,
        noiser_name="eggroll",
        rank=int(designer._rank),
        group_size=int(designer._group_size),
        freeze_nonlora=bool(designer._freeze_nonlora),
    )
    return np.asarray(direction, dtype=np.float64).reshape(state.x.shape)


def _mirrored_gradient(designer, state, directions: list[np.ndarray], raw_scores: list[float], population: int) -> np.ndarray:
    scores = _standardize_scores(designer, np.asarray(raw_scores, dtype=np.float64))
    grad = np.zeros_like(state.x, dtype=np.float64)
    for pair_idx, direction in enumerate(directions):
        grad += (float(scores[2 * pair_idx]) - float(scores[(2 * pair_idx) + 1])) * direction
    return grad / math.sqrt(float(max(population, 1)))


def _standardize_scores(designer, raw_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if designer._rank_transform:
        ranks = np.argsort(np.argsort(scores)).astype(np.float64)
        scores = ranks / max(float(scores.size - 1), 1.0)
    if int(designer._group_size) > 0:
        if scores.size % int(designer._group_size) != 0:
            raise NoSuchDesignerError("EggRoll option 'group_size' must divide the population.")
        grouped = scores.reshape((-1, int(designer._group_size)))
        centered = grouped - np.mean(grouped, axis=1, keepdims=True)
        scores = centered.reshape(-1)
    else:
        scores = scores - float(np.mean(scores))
    return scores / float(np.sqrt(np.var(scores) + 1e-5))


def _log(designer, message: str) -> None:
    print(f"EGGROLL: {message}", flush=True)
