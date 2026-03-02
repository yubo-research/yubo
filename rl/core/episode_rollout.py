from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np


def _resolve_max_episode_steps(env_conf: Any) -> int:
    gym_conf = getattr(env_conf, "gym_conf", None)
    if gym_conf is not None:
        return int(getattr(gym_conf, "max_steps", 99999))
    return int(getattr(env_conf, "max_steps", 99999))


def _obs_for_policy(observation: Any) -> Any:
    if not isinstance(observation, dict):
        return observation
    if "observation" in observation:
        return observation["observation"]
    if "pixels" in observation:
        return observation["pixels"]
    if "state" in observation:
        return observation["state"]
    parts = [np.ravel(np.asarray(observation[key], dtype=np.float32)) for key in sorted(observation)]
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _bool_any(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    arr = np.asarray(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(arr.any())


def _float_sum(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr.item())
    return float(arr.sum())


def _scale_action_to_space(action: np.ndarray | int, action_space: Any) -> np.ndarray | int:
    if not hasattr(action_space, "low"):
        if hasattr(action_space, "n"):
            if isinstance(action, (int, float, np.integer)):
                return int(action)
            arr = np.asarray(action).ravel()
            return int(arr.item()) if arr.size == 1 else int(arr[0])
        return action
    action_f64 = np.asarray(action, dtype=np.float64)
    return action_space.low + (action_space.high - action_space.low) * (1 + action_f64) / 2


def _unpack_reset_result(reset_out: Any) -> tuple[Any, Any]:
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
        return obs, info
    return reset_out, {}


def _unpack_step_result(step_out: Any) -> tuple[Any, Any, bool, bool, Any]:
    if len(step_out) == 5:
        state, reward, terminated, truncated, info = step_out
        return state, reward, _bool_any(terminated), _bool_any(truncated), info
    if len(step_out) == 4:
        state, reward, done, info = step_out
        return state, reward, _bool_any(done), False, info
    raise ValueError(f"Unsupported env.step return arity: {len(step_out)}")


@dataclass(frozen=True)
class Trajectory:
    rreturn: float
    rreturn_se: float | None = None


class MeanReturnResult(NamedTuple):
    mean: float
    se: float
    all_same: bool


def collect_episode_return(env_conf: Any, policy: Any, *, noise_seed: int | None) -> float:
    with closing(env_conf.make()) as env:
        observation, _ = _unpack_reset_result(env.reset(seed=noise_seed))
        episode_return = 0.0
        max_steps = _resolve_max_episode_steps(env_conf)
        for _ in range(int(max_steps)):
            policy_observation = _obs_for_policy(observation)
            action = policy(np.asarray(policy_observation))
            env_action = _scale_action_to_space(action, env.action_space)
            observation, reward, terminated, truncated, _ = _unpack_step_result(env.step(env_action))
            episode_return += _float_sum(reward)
            if terminated or truncated:
                break
        return float(episode_return)


def _resolve_noise_seed(env_conf: Any, *, i_noise: int | None, denoise_seed: int) -> int:
    base = 0 if i_noise is None else int(i_noise)
    noise_seed_0 = getattr(env_conf, "noise_seed_0", 0)
    noise_seed_0_i = 0 if noise_seed_0 is None else int(noise_seed_0)
    return int(base + noise_seed_0_i + int(denoise_seed))


def collect_trajectory_with_noise(
    env_conf: Any,
    policy: Any,
    *,
    i_noise: int | None = None,
    denoise_seed: int = 0,
) -> tuple[Trajectory, int]:
    noise_seed = _resolve_noise_seed(env_conf, i_noise=i_noise, denoise_seed=int(denoise_seed))
    episode_return = collect_episode_return(env_conf, policy, noise_seed=int(noise_seed))
    return Trajectory(float(episode_return)), int(noise_seed)


def mean_return_over_runs(env_conf: Any, policy: Any, num_denoise: int, *, i_noise: int | None = None) -> MeanReturnResult:
    n = int(num_denoise)
    if n <= 0:
        raise ValueError("num_denoise must be > 0")
    returns = [collect_trajectory_with_noise(env_conf, policy, i_noise=i_noise, denoise_seed=i)[0].rreturn for i in range(n)]
    std = float(np.std(returns))
    all_same = len(returns) > 1 and std == 0.0
    se = float(std / np.sqrt(len(returns)))
    return MeanReturnResult(
        mean=float(np.mean(returns)),
        se=se,
        all_same=bool(all_same),
    )


def collect_denoised_trajectory(
    env_conf: Any,
    policy: Any,
    *,
    num_denoise: int | None,
    i_noise: int | None = None,
) -> tuple[Trajectory, int | None]:
    if num_denoise is None:
        return collect_trajectory_with_noise(env_conf, policy, i_noise=i_noise, denoise_seed=0)

    if int(num_denoise) == 1:
        return collect_trajectory_with_noise(env_conf, policy, i_noise=i_noise, denoise_seed=0)

    mean_ret, se_ret, _ = mean_return_over_runs(env_conf, policy, int(num_denoise), i_noise=i_noise)
    rreturn_se = None if bool(getattr(env_conf, "frozen_noise", False)) else float(se_ret)
    return Trajectory(float(mean_ret), rreturn_se=rreturn_se), None


def evaluate_for_best(env_conf: Any, policy: Any, num_denoise_passive_eval: int, *, i_noise: int = 99999) -> float:
    mean_ret, _, _ = mean_return_over_runs(env_conf, policy, int(num_denoise_passive_eval), i_noise=int(i_noise))
    return float(mean_ret)
