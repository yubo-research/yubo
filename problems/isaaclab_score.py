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
        start = 0
        params = list(policy.parameters()) if hasattr(policy, "parameters") else []
        if params:
            for param in params:
                size = int(param.numel())
                offsets.append(start)
                sizes.append(size)
                start += size
        else:
            offsets.append(0)
            sizes.append(self.dim)
        self.offsets = tuple(offsets)
        self.sizes = tuple(sizes)

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
        self._device = None if device in (None, "auto", "") else str(device)
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
        env = self._env
        self._env = None
        if env is not None:
            env.close()
        for vector_env in self._vector_envs.values():
            vector_env.close()
        self._vector_envs.clear()

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
        if self._vectorize and xs.shape[0] > 1:
            vector_result = _try_evaluate_many_vectorized(self, xs, seed=int(seed))
            if vector_result is not None:
                return vector_result
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
        return self._noise.sample(seed=int(seed), num_dim_target=num_dim_target, num_module_target=num_module_target)

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
        _ = rank, group_size, freeze_nonlora
        if str(noiser_name) != "eggroll":
            raise ValueError(f"IsaacLab scoring supports noiser='eggroll' only, got {noiser_name!r}.")
        return self.sample_noise(seed=int(seed))

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
    kwargs = {}
    if scorer._device is not None:
        kwargs["device"] = scorer._device
    try:
        scorer._env = scorer._env_runtime.make(**kwargs)
    except TypeError:
        kwargs.pop("device", None)
        scorer._env = scorer._env_runtime.make(**kwargs)
    return scorer._env


def _get_vector_env(scorer: IsaacLabScore, slots: int):
    slots = int(slots)
    if slots < 1:
        return None
    env = scorer._vector_envs.get(slots)
    if env is not None:
        return env
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


def _policy_supports_vector_eval(policy: Any) -> bool:
    if bool(getattr(policy, "_recurrent", False)):
        return False
    return getattr(policy, "_rnn_hidden_size", None) is None


def _try_evaluate_many_vectorized(scorer: IsaacLabScore, xs: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not _policy_supports_vector_eval(scorer._policy):
        return None
    num_candidates = int(xs.shape[0])
    slots = num_candidates * scorer._episodes
    env = _get_vector_env(scorer, slots)
    if env is None:
        return None
    obs, _info = env.reset_batch(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[0] != slots:
        return None

    policies = [scorer._make_loaded_policy(xs[i]) for i in range(num_candidates)]
    candidate_idx = np.repeat(np.arange(num_candidates, dtype=np.int64), scorer._episodes)
    returns = np.zeros(slots, dtype=np.float64)
    steps = np.zeros(slots, dtype=np.int64)
    active = np.ones(slots, dtype=bool)
    zero_action = np.zeros(tuple(int(v) for v in env.action_space.shape), dtype=np.float32)

    for _ in range(scorer.steps_per_episode):
        raw_actions = _vector_policy_actions(policies, candidate_idx, obs, active, zero_action)
        actions = np.stack([_scale_action_to_space(action, env.action_space) for action in raw_actions], axis=0)
        next_obs, reward, terminated, truncated, _info = env.step_batch(actions)
        reward = np.asarray(reward, dtype=np.float64).reshape(slots)
        done = np.asarray(terminated, dtype=bool).reshape(slots) | np.asarray(truncated, dtype=bool).reshape(slots)

        returns[active] += reward[active]
        steps[active] += 1
        active &= ~done
        obs = np.asarray(next_obs, dtype=np.float32)
        if not np.any(active):
            break

    means, ses = _summarize_vector_returns(returns, scorer._episodes, num_candidates)
    scorer.last_num_steps = int(np.sum(steps))
    return means, ses


def _vector_policy_actions(policies: list[Any], candidate_idx: np.ndarray, obs: np.ndarray, active: np.ndarray, zero_action: np.ndarray) -> list[np.ndarray]:
    raw_actions = np.zeros((int(candidate_idx.size), *tuple(zero_action.shape)), dtype=np.float32)
    for cand_idx, policy in enumerate(policies):
        slots = np.flatnonzero(active & (candidate_idx == int(cand_idx)))
        if slots.size == 0:
            continue
        actions = np.asarray(policy(obs[slots]), dtype=np.float32)
        raw_actions[slots] = actions.reshape((slots.size, *tuple(zero_action.shape)))
    return [raw_actions[slot] if active[slot] else zero_action for slot in range(int(candidate_idx.size))]


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
    scorer._codec.load(scorer._policy, np.asarray(x, dtype=np.float64))
    returns: list[float] = []
    total_steps = 0
    for episode_idx in range(scorer._episodes):
        value, num_steps = scorer._run_episode(int(seed) + episode_idx)
        returns.append(float(value))
        total_steps += int(num_steps)
    arr = np.asarray(returns, dtype=np.float64)
    stderr = 0.0 if arr.size <= 1 else float(np.std(arr, ddof=0) / math.sqrt(float(arr.size)))
    return Score(value=float(np.mean(arr)), stderr=stderr, episodes=int(arr.size), num_steps=int(total_steps))
