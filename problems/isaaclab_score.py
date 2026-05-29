from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from optimizer.eggroll_runtime_noise import EggRollNoiseSampler
from optimizer.trajectories import (
    _obs_for_policy,
    _resolve_max_episode_steps,
    _resolve_obs_bounds,
    _scale_action_to_space,
    _unpack_step_result,
)
from problems.isaaclab_tensor_score import try_evaluate_many_tensor_vectorized
from problems.torch_policy_batch import try_functional_policy_actions


@dataclass(frozen=True)
class Score:
    value: float
    stderr: float
    episodes: int
    num_steps: int


class _PolicyCodec:
    def __init__(self, policy: Any) -> None:
        if not all(hasattr(policy, name) for name in ("get_params", "set_params", "clone", "num_params")):
            raise TypeError("IsaacLab scoring requires a flat-parameter policy with get_params, set_params, clone, and num_params.")
        self.dim = int(policy.num_params())
        offsets: list[int] = []
        sizes: list[int] = []
        shapes: list[tuple[int, ...]] = []
        start = 0
        params = list(policy.parameters()) if hasattr(policy, "parameters") else []
        if params:
            for param in params:
                size = int(param.numel())
                offsets.append(start)
                sizes.append(size)
                shapes.append(tuple(int(v) for v in param.shape))
                start += size
        else:
            offsets.append(0)
            sizes.append(self.dim)
            shapes.append((self.dim,))
        self.offsets = tuple(offsets)
        self.sizes = tuple(sizes)
        self.shapes = tuple(shapes)

    def initial(self, policy: Any) -> np.ndarray:
        x = np.asarray(policy.get_params(), dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Policy returned {x.size} params, expected {self.dim}.")
        return x

    def load(self, policy: Any, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Candidate has {x.size} params, expected {self.dim}.")
        policy.set_params(np.clip(x, -1.0, 1.0))


class IsaacLabScore:
    """Score flat policy vectors by running IsaacLab rollouts."""

    def __init__(
        self,
        env_runtime: Any,
        policy: Any,
        *,
        episodes: int = 1,
        device: str | None = None,
        steps_per_episode: int | None = None,
        vectorize: bool = True,
    ) -> None:
        self._env_runtime = env_runtime
        self._policy = policy
        self._episodes = max(1, int(episodes))
        if device in (None, "auto", ""):
            self._device = None
        else:
            from problems.isaaclab_env_adapters import resolve_isaaclab_sim_device

            self._device = resolve_isaaclab_sim_device(str(device))
        self._steps_per_episode = None if steps_per_episode is None else int(steps_per_episode)
        self._vectorize = bool(vectorize)
        self._env = None
        self._vector_envs: dict[int, Any] = {}
        self._codec = _PolicyCodec(policy)
        self._noise = EggRollNoiseSampler(self._codec)
        self._x0 = self._codec.initial(policy)
        self.last_num_steps = 0
        self._embed_num_probes = 0
        self._embed_matrix: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(self._codec.dim)

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def steps_per_episode(self) -> int:
        if self._steps_per_episode is not None:
            return int(self._steps_per_episode)
        return int(_resolve_max_episode_steps(self._env_runtime))

    @property
    def num_envs(self) -> int:
        return int(self._episodes)

    def close(self) -> None:
        _close_single_env(self)
        _close_vector_envs(self)

    def make_policy(self, x: np.ndarray):
        return self._make_loaded_policy(x)

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        score = _score_candidate(self, x, seed=int(seed))
        self.last_num_steps = int(score.num_steps)
        return score.value, score.stderr

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        xs = np.asarray(x_batch, dtype=np.float64)
        if xs.ndim != 2:
            raise ValueError(f"x_batch must be rank-2, got shape {xs.shape}.")
        if self._vectorize and (xs.shape[0] > 1 or self._episodes > 1):
            vector_result = _try_evaluate_many_vectorized(self, xs, seed=int(seed))
            if vector_result is not None:
                means, ses, num_steps = vector_result
                self.last_num_steps = int(num_steps)
                return means, ses
        scores = [_score_candidate(self, x, seed=int(seed) + idx * self._episodes) for idx, x in enumerate(xs)]
        self.last_num_steps = int(sum(int(s.num_steps) for s in scores))
        return (
            np.asarray([s.value for s in scores], dtype=np.float64),
            np.asarray([s.stderr for s in scores], dtype=np.float64),
        )

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        return self._noise.sample(
            seed=int(seed),
            num_dim_target=num_dim_target,
            num_module_target=num_module_target,
        )

    def sample_eggroll_noiser_noise(
        self,
        _x: np.ndarray,
        *,
        seed: int,
        noiser_name: str = "eggroll",
        rank: int = 1,
        group_size: int = 0,
        freeze_nonlora: bool = False,
    ) -> np.ndarray:
        if str(noiser_name) != "eggroll":
            raise ValueError(f"IsaacLab scoring supports noiser='eggroll' only, got {noiser_name!r}.")
        return _sample_isaaclab_eggroll_noise(
            self._codec,
            seed=int(seed),
            rank=int(rank),
            group_size=int(group_size),
            freeze_nonlora=bool(freeze_nonlora),
        )

    def configure_embedding(self, num_probes: int) -> None:
        n = max(0, int(num_probes))
        self._embed_num_probes = n
        if n <= 0:
            self._embed_matrix = None
            return
        rng = np.random.default_rng(0)
        self._embed_matrix = rng.standard_normal((self.dim, n)).astype(np.float64) / math.sqrt(float(max(self.dim, 1)))

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        xs = np.asarray(x_batch, dtype=np.float64)
        if xs.ndim == 1:
            xs = xs.reshape(1, -1)
        if self._embed_matrix is None:
            return xs.copy()
        return xs @ self._embed_matrix

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embed_many(np.asarray(x, dtype=np.float64).reshape(1, -1))[0]

    def _make_loaded_policy(self, x: np.ndarray):
        policy = self._policy.clone()
        self._codec.load(policy, np.asarray(x, dtype=np.float64))
        if hasattr(policy, "reset_state"):
            policy.reset_state()
        return policy

    def _run_episode(self, seed: int) -> tuple[float, int]:
        env = _get_single_env(self)
        if hasattr(self._policy, "reset_state"):
            self._policy.reset_state()
        state, _info = env.reset(seed=int(seed))
        state_policy = np.asarray(_obs_for_policy(state), dtype=np.float32)
        lb, width = _resolve_obs_bounds(env.observation_space, state_policy)
        transform_state = bool(getattr(getattr(self._env_runtime, "gym_conf", None), "transform_state", False))
        max_steps = self.steps_per_episode
        total = 0.0
        num_steps = 0
        for _ in range(max_steps):
            state_policy = np.asarray(_obs_for_policy(state), dtype=np.float32)
            policy_input = (state_policy - lb) / width if transform_state else state_policy
            action = _scale_action_to_space(self._policy(policy_input), env.action_space)
            state, reward, terminated, truncated, _info = _unpack_step_result(env.step(action))
            total += float(reward)
            num_steps += 1
            if terminated or truncated:
                break
        return float(total), int(num_steps)


def _get_single_env(scorer: IsaacLabScore):
    if scorer._env is not None:
        return scorer._env
    _close_vector_envs(scorer)
    kwargs = {}
    if scorer._device is not None:
        kwargs["device"] = scorer._device
    try:
        scorer._env = scorer._env_runtime.make(**kwargs)
    except TypeError:
        kwargs.pop("device", None)
        scorer._env = scorer._env_runtime.make(**kwargs)
    return scorer._env


def active_vector_slots(scorer: IsaacLabScore) -> int | None:
    return next(iter(scorer._vector_envs), None)


def _sample_isaaclab_eggroll_noise(
    codec: _PolicyCodec,
    *,
    seed: int,
    rank: int,
    group_size: int,
    freeze_nonlora: bool,
) -> np.ndarray:
    if int(rank) < 1:
        raise ValueError("IsaacLab EggRoll rank must be >= 1.")
    if int(group_size) < 0:
        raise ValueError("IsaacLab EggRoll group_size must be >= 0.")
    noise = np.zeros(int(codec.dim), dtype=np.float64)
    leaf_seeds = np.random.SeedSequence(int(seed)).spawn(len(codec.sizes))
    leaves = zip(codec.offsets, codec.sizes, codec.shapes, leaf_seeds, strict=True)
    for start, size, shape, leaf_seed in leaves:
        end = int(start) + int(size)
        noise[int(start) : end] = _sample_param_eggroll_noise(
            np.random.default_rng(leaf_seed),
            shape=tuple(shape),
            rank=int(rank),
            freeze_nonlora=bool(freeze_nonlora),
        ).reshape(-1)
    return noise


def _sample_param_eggroll_noise(
    rng: np.random.Generator,
    *,
    shape: tuple[int, ...],
    rank: int,
    freeze_nonlora: bool,
) -> np.ndarray:
    if len(shape) < 2:
        if freeze_nonlora:
            return np.zeros(shape, dtype=np.float64)
        return rng.standard_normal(shape).astype(np.float64)
    rows = int(shape[0])
    cols = int(np.prod(shape[1:], dtype=np.int64))
    left = rng.standard_normal((rows, int(rank)))
    right = rng.standard_normal((cols, int(rank)))
    return (left @ right.T / math.sqrt(float(rank))).reshape(shape).astype(np.float64)


def _get_vector_env(scorer: IsaacLabScore, slots: int):
    slots = int(slots)
    if slots < 1:
        return None
    env = scorer._vector_envs.get(slots)
    if env is not None:
        return env
    _close_vector_envs(scorer)
    _close_single_env(scorer)
    kwargs = {"num_envs": slots, "batched": True}
    if scorer._device is not None:
        kwargs["device"] = scorer._device
    try:
        env = scorer._env_runtime.make(**kwargs)
    except TypeError:
        return None
    if not all(hasattr(env, name) for name in ("reset_batch", "step_batch")):
        try:
            env.close()
        except Exception:
            pass
        return None
    scorer._vector_envs[slots] = env
    return env


def _close_env(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass


def _close_single_env(scorer: IsaacLabScore) -> None:
    env = scorer._env
    scorer._env = None
    if env is not None:
        _close_env(env)


def _close_vector_envs(scorer: IsaacLabScore) -> None:
    envs = list(scorer._vector_envs.values())
    scorer._vector_envs.clear()
    for env in envs:
        _close_env(env)


def _policy_supports_vector_eval(policy: Any) -> bool:
    if bool(getattr(policy, "_recurrent", False)):
        return False
    return getattr(policy, "_rnn_hidden_size", None) is None


def _try_evaluate_many_vectorized(scorer: IsaacLabScore, xs: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray, int] | None:
    if not _policy_supports_vector_eval(scorer._policy):
        return None
    num_candidates = int(xs.shape[0])
    slots = num_candidates * scorer._episodes
    env = _get_vector_env(scorer, slots)
    if env is None:
        return None
    tensor_result = try_evaluate_many_tensor_vectorized(scorer, xs, env, seed=int(seed))
    if tensor_result is not None:
        return tensor_result
    obs, _info = env.reset_batch(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[0] != slots:
        return None

    candidate_idx = np.repeat(np.arange(num_candidates, dtype=np.int64), scorer._episodes)
    returns = np.zeros(slots, dtype=np.float64)
    steps = np.zeros(slots, dtype=np.int64)
    active = np.ones(slots, dtype=bool)
    zero_action = np.zeros(tuple(int(v) for v in env.action_space.shape), dtype=np.float32)

    for _ in range(scorer.steps_per_episode):
        raw_actions = _vector_policy_actions(scorer, xs, candidate_idx, obs, active, zero_action)
        actions = _scale_action_batch_to_space(raw_actions, env.action_space)
        next_obs, reward, terminated, truncated, _info = env.step_batch(actions)
        reward = np.asarray(reward, dtype=np.float64).reshape(slots)
        done = np.asarray(terminated, dtype=bool).reshape(slots) | np.asarray(truncated, dtype=bool).reshape(slots)

        returns[active] += reward[active]
        steps[active] += 1
        active &= ~done
        obs = np.asarray(next_obs, dtype=np.float32)
        if not np.any(active):
            break

    return (
        *_summarize_vector_returns(returns, scorer._episodes, num_candidates),
        int(np.sum(steps)),
    )


def _vector_policy_actions(
    scorer: IsaacLabScore,
    xs: np.ndarray,
    candidate_idx: np.ndarray,
    obs: np.ndarray,
    active: np.ndarray,
    zero_action: np.ndarray,
) -> np.ndarray:
    actions = try_functional_policy_actions(scorer._policy, scorer._codec, xs, candidate_idx, obs, active, zero_action)
    if actions is not None:
        return actions
    return _clone_policy_actions(scorer, xs, candidate_idx, obs, active, zero_action)


def _clone_policy_actions(
    scorer: IsaacLabScore,
    xs: np.ndarray,
    candidate_idx: np.ndarray,
    obs: np.ndarray,
    active: np.ndarray,
    zero_action: np.ndarray,
) -> np.ndarray:
    policies = [scorer._make_loaded_policy(xs[i]) for i in range(int(xs.shape[0]))]
    raw_actions = np.zeros((int(candidate_idx.size), *tuple(zero_action.shape)), dtype=np.float32)
    for cand_idx, policy in enumerate(policies):
        slots = np.flatnonzero(active & (candidate_idx == int(cand_idx)))
        if slots.size == 0:
            continue
        actions = np.asarray(policy(obs[slots]), dtype=np.float32)
        raw_actions[slots] = actions.reshape((slots.size, *tuple(zero_action.shape)))
    return raw_actions


def _scale_action_batch_to_space(actions: np.ndarray, action_space: Any) -> np.ndarray:
    if not hasattr(action_space, "low"):
        return np.asarray(actions, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        return np.clip(actions, -1.0, 1.0).astype(np.float32)
    return (low + (high - low) * (1.0 + actions) / 2.0).astype(np.float32)


def _summarize_vector_returns(returns: np.ndarray, episodes: int, num_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    returns_by_candidate = returns.reshape(num_candidates, int(episodes))
    means = np.mean(returns_by_candidate, axis=1).astype(np.float64)
    if int(episodes) <= 1:
        ses = np.zeros((num_candidates,), dtype=np.float64)
    else:
        ses = np.std(returns_by_candidate, axis=1, ddof=0) / math.sqrt(float(episodes))
    return means, np.asarray(ses, dtype=np.float64)


def build_isaaclab_evaluator(
    cfg,
    *,
    episodes: int | None = None,
    embed_num_probes: int = 0,
) -> IsaacLabScore:
    from problems.problem import build_problem

    if cfg.policy_tag is None:
        raise ValueError("IsaacLab scoring requires policy_tag.")
    problem = build_problem(
        cfg.env_tag,
        cfg.policy_tag,
        problem_seed=cfg.problem_seed,
        noise_seed_0=cfg.noise_seed_0,
    )
    policy = problem.build_policy()
    steps_per_episode = getattr(cfg, "steps_per_episode", None)
    if steps_per_episode is None:
        steps_per_episode = getattr(cfg, "steps", None)
    scorer = IsaacLabScore(
        problem.env,
        policy,
        episodes=cfg.num_envs if episodes is None else int(episodes),
        device=getattr(cfg, "runtime_device", None),
        steps_per_episode=steps_per_episode,
    )
    if int(embed_num_probes) > 0:
        scorer.configure_embedding(int(embed_num_probes))
    return scorer


def _score_candidate(scorer: IsaacLabScore, x: np.ndarray, *, seed: int) -> Score:
    if scorer._vectorize and scorer._episodes > 1:
        vector_result = _try_evaluate_many_vectorized(scorer, np.asarray(x, dtype=np.float64).reshape(1, -1), seed=int(seed))
        if vector_result is not None:
            means, ses, num_steps = vector_result
            return Score(
                value=float(means[0]),
                stderr=float(ses[0]),
                episodes=int(scorer._episodes),
                num_steps=int(num_steps),
            )
    scorer._codec.load(scorer._policy, np.asarray(x, dtype=np.float64))
    returns: list[float] = []
    total_steps = 0
    for episode_idx in range(scorer._episodes):
        value, num_steps = scorer._run_episode(int(seed) + episode_idx)
        returns.append(float(value))
        total_steps += int(num_steps)
    arr = np.asarray(returns, dtype=np.float64)
    stderr = 0.0 if arr.size <= 1 else float(np.std(arr, ddof=0) / math.sqrt(float(arr.size)))
    return Score(
        value=float(np.mean(arr)),
        stderr=stderr,
        episodes=int(arr.size),
        num_steps=int(total_steps),
    )
