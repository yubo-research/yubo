from __future__ import annotations

import atexit
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from .trajectories import _obs_for_policy, _resolve_max_episode_steps, _resolve_obs_bounds, _scale_action_to_space

__all__ = [
    "VectorizedPolicyEvaluator",
    "clear_cached_vectorized_policy_evaluators",
    "get_vectorized_policy_evaluator",
]

_VECTOR_EVAL_CACHE: dict[tuple[int, int, str | None], "VectorizedPolicyEvaluator"] = {}


def clear_cached_vectorized_policy_evaluators() -> None:
    for evaluator in _VECTOR_EVAL_CACHE.values():
        try:
            evaluator.close()
        except Exception:
            pass
    _VECTOR_EVAL_CACHE.clear()


atexit.register(clear_cached_vectorized_policy_evaluators)


def _clone_policy(policy: Any) -> Any:
    clone = getattr(policy, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(policy)


def _batch_obs_to_list(obs_batch: Any, num_envs: int) -> list[Any]:
    if isinstance(obs_batch, dict):
        items: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        for key, value in obs_batch.items():
            arr = np.asarray(value)
            for i in range(num_envs):
                items[i][key] = arr[i]
        return items
    arr = np.asarray(obs_batch)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [arr[i] for i in range(num_envs)]


def _supports_batched_forward(policy: Any) -> bool:
    return bool(
        getattr(policy, "_deterministic_eval", False) and hasattr(policy, "actor") and hasattr(policy, "_normalize") and hasattr(policy, "_postprocess_action")
    )


@dataclass
class _VectorEvalResult:
    mean: float
    se: float
    all_same: bool
    num_steps_total: int


@dataclass
class _VectorEvalContext:
    policies: list[Any]
    action_space: Any
    lb: np.ndarray
    width: np.ndarray
    transform_state: bool
    batch_forward: bool


class VectorizedPolicyEvaluator:
    """Reusable vectorized evaluator for repeated BO rollouts."""

    def __init__(self, env_conf: Any, *, num_envs: int, render_mode: str | None = None):
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self._env_conf = env_conf
        self._num_envs = int(num_envs)
        self._render_mode = render_mode
        self._vec_env = SyncVectorEnv([self._make_env_factory() for _ in range(self._num_envs)])
        self._max_steps = _resolve_max_episode_steps(env_conf)
        self._transform_state = bool(getattr(getattr(env_conf, "gym_conf", None), "transform_state", False))

    def _make_env_factory(self):
        render_mode = self._render_mode
        env_conf = self._env_conf

        def _make():
            return env_conf.make(render_mode=render_mode)

        return _make

    def close(self) -> None:
        self._vec_env.close()

    def __enter__(self) -> "VectorizedPolicyEvaluator":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _build_context(self, policies: list[Any], obs_items: list[Any]) -> _VectorEvalContext:
        sample_obs = np.asarray(_obs_for_policy(obs_items[0]), dtype=np.float32)
        obs_space = getattr(self._vec_env, "single_observation_space", getattr(self._vec_env, "observation_space", None))
        lb, width = _resolve_obs_bounds(obs_space, sample_obs)
        action_space = getattr(self._vec_env, "single_action_space", getattr(self._vec_env, "action_space", None))
        if action_space is None:
            raise ValueError("Vector env must expose an action space.")
        return _VectorEvalContext(
            policies=policies,
            action_space=action_space,
            lb=lb,
            width=width,
            transform_state=self._transform_state,
            batch_forward=self._num_envs > 1 and all(_supports_batched_forward(p) for p in policies),
        )

    def _actions_for_obs_items(self, ctx: _VectorEvalContext, obs_items: list[Any]) -> list[Any]:
        if ctx.batch_forward:
            state_args = []
            for i in range(self._num_envs):
                obs_i = np.asarray(_obs_for_policy(obs_items[i]), dtype=np.float32)
                state_i = (obs_i - ctx.lb) / ctx.width
                state_arg = state_i if ctx.transform_state else obs_i
                state_args.append(np.asarray(ctx.policies[i]._normalize(state_arg), dtype=np.float32))

            state_batch = torch.as_tensor(np.stack(state_args, axis=0), dtype=torch.float32)
            with torch.inference_mode():
                action_batch = ctx.policies[0].actor.forward(state_batch)
                action_batch = ctx.policies[0]._postprocess_action(action_batch)
            action_batch_np = action_batch.detach().cpu().numpy()
            return [_scale_action_to_space(action_batch_np[i], ctx.action_space) for i in range(self._num_envs)]

        actions = []
        for i in range(self._num_envs):
            obs_i = np.asarray(_obs_for_policy(obs_items[i]), dtype=np.float32)
            state_i = (obs_i - ctx.lb) / ctx.width
            state_arg = state_i if ctx.transform_state else obs_i
            action_p = ctx.policies[i](state_arg)
            actions.append(_scale_action_to_space(action_p, ctx.action_space))
        return actions

    def evaluate(self, policy: Any, *, base_seed: int = 0) -> _VectorEvalResult:
        policies = [_clone_policy(policy) for _ in range(self._num_envs)]
        seed_list = [int(base_seed) + i for i in range(self._num_envs)]
        try:
            obs_batch, _ = self._vec_env.reset(seed=seed_list)
        except TypeError:
            obs_batch, _ = self._vec_env.reset(seed=int(base_seed))

        obs_items = _batch_obs_to_list(obs_batch, self._num_envs)
        ctx = self._build_context(policies, obs_items)

        returns = np.zeros(self._num_envs, dtype=np.float64)
        steps_per_env = np.zeros(self._num_envs, dtype=np.int64)
        done = np.zeros(self._num_envs, dtype=bool)

        while True:
            actions = self._actions_for_obs_items(ctx, obs_items)

            obs_batch, reward, terminated, truncated, _info = self._vec_env.step(actions)
            obs_items = _batch_obs_to_list(obs_batch, self._num_envs)
            reward = np.asarray(reward, dtype=np.float64)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)

            active = ~done
            returns[active] += reward[active]
            steps_per_env[active] += 1
            done = done | terminated | truncated
            if max(steps_per_env) >= self._max_steps:
                done = done | (steps_per_env >= self._max_steps)
            if np.all(done):
                break

        total_steps = int(steps_per_env.sum())
        if len(returns) <= 1:
            mean_return = float(returns[0]) if len(returns) == 1 else 0.0
            std_return = 0.0
        else:
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))
        return _VectorEvalResult(
            mean=mean_return,
            se=float(std_return / np.sqrt(max(1, len(returns)))),
            all_same=bool(len(returns) > 1 and std_return == 0),
            num_steps_total=total_steps,
        )


def get_vectorized_policy_evaluator(env_conf: Any, *, num_envs: int, render_mode: str | None = None) -> VectorizedPolicyEvaluator:
    key = (id(env_conf), int(num_envs), render_mode)
    evaluator = _VECTOR_EVAL_CACHE.get(key)
    if evaluator is None:
        evaluator = VectorizedPolicyEvaluator(env_conf, num_envs=int(num_envs), render_mode=render_mode)
        _VECTOR_EVAL_CACHE[key] = evaluator
    return evaluator
