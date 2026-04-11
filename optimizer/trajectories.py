import atexit
import os
from typing import Any

import numpy as np

from .trajectory import Trajectory

__all__ = ["collect_trajectory"]

_REUSE_EVAL_ENV_ENVVAR = "YUBO_TURBO_ENN_REUSE_ENV"
_EVAL_ENV_CACHE: dict[tuple[int, str | None], Any] = {}


def _reuse_eval_env_enabled() -> bool:
    value = str(os.getenv(_REUSE_EVAL_ENV_ENVVAR, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _clear_cached_eval_envs() -> None:
    for env in _EVAL_ENV_CACHE.values():
        try:
            env.close()
        except Exception:
            pass
    _EVAL_ENV_CACHE.clear()


atexit.register(_clear_cached_eval_envs)


def _resolve_max_episode_steps(env_conf: Any) -> int:
    """Inlined from common.video to avoid pulling common.video into optimizer transitive deps."""
    if getattr(env_conf, "gym_conf", None) is not None:
        return int(env_conf.gym_conf.max_steps)
    return int(getattr(env_conf, "max_steps", 99999))


def _scale_action_to_space(action: np.ndarray | int, action_space: Any) -> np.ndarray | int:
    """Inlined from common.video to avoid pulling common.video into optimizer transitive deps."""
    if not hasattr(action_space, "low"):
        if hasattr(action_space, "n"):
            if isinstance(action, (int, float, np.integer)):
                return int(action)
            arr = np.asarray(action).ravel()
            return int(arr.item()) if arr.size == 1 else int(arr[0])
        return action
    action = np.asarray(action, dtype=np.float64)
    return action_space.low + (action_space.high - action_space.low) * (1 + action) / 2


def _unpack_step_result(step_out: Any) -> tuple[Any, Any, bool, bool, Any]:
    # Support both Gymnasium (5-tuple) and older Gym-style (4-tuple) env.step outputs.
    if len(step_out) == 5:
        state, reward, terminated, truncated, info = step_out
        return state, reward, bool(terminated), bool(truncated), info
    if len(step_out) == 4:
        state, reward, done, info = step_out
        return state, reward, bool(done), False, info
    raise ValueError(f"Unsupported env.step return arity: {len(step_out)}")


def _get_bounds_safe(arr, default_fn):
    return default_fn(arr.shape) if np.all(np.isinf(arr)) else arr


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


def _resolve_obs_bounds(observation_space: Any, sample_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(observation_space, "low") and hasattr(observation_space, "high"):
        low = np.asarray(observation_space.low, dtype=np.float32)
        high = np.asarray(observation_space.high, dtype=np.float32)
        if low.shape == sample_obs.shape and high.shape == sample_obs.shape:
            lb = _get_bounds_safe(low, np.zeros)
            width = _get_bounds_safe(high - lb, np.ones)
            if not np.any(np.isinf(lb)) and not np.any(np.isinf(width)):
                return lb, width
    return np.zeros_like(sample_obs, dtype=np.float32), np.ones_like(sample_obs, dtype=np.float32)


def _make_return(policy, return_trajectory):
    if not (hasattr(policy, "wants_vector_return") and bool(policy.wants_vector_return())):
        return return_trajectory
    assert hasattr(policy, "metrics")
    mets = np.asarray(policy.metrics(), dtype=np.float64)
    return np.concatenate([np.asarray([float(return_trajectory)], dtype=np.float64), mets], axis=0)


def _collect_ppo_step(policy, ppo, reward, done):
    ppo["rewards"].append(reward)
    ppo["dones"].append(done)
    if ppo["has_log_probs"]:
        ppo["log_probs"].append(policy.last_log_probs())
    if ppo["has_values"]:
        ppo["values"].append(policy.last_values())


def _make_eval_env(env_conf, render_mode: str | None):
    return env_conf.make(render_mode=render_mode)


def _get_eval_env(env_conf, render_mode: str | None, *, allow_reuse: bool):
    if not allow_reuse:
        return _make_eval_env(env_conf, render_mode), False
    cache_key = (id(env_conf), render_mode)
    env = _EVAL_ENV_CACHE.get(cache_key)
    if env is None:
        env = _make_eval_env(env_conf, render_mode)
        _EVAL_ENV_CACHE[cache_key] = env
    return env, True


def _policy_allows_env_reuse(policy: Any) -> bool:
    return bool(getattr(policy, "_turbo_enn_eval_reuse_ok", False))


def collect_trajectory(env_conf, policy, noise_seed=None, show_frames=False):
    b_gym = env_conf.gym_conf is not None
    num_frames_skip = env_conf.gym_conf.num_frames_skip if b_gym and show_frames else 1

    render_mode = "rgb_array" if show_frames else None
    reuse_env = _reuse_eval_env_enabled() and not show_frames and _policy_allows_env_reuse(policy)
    env, keep_open = _get_eval_env(env_conf, render_mode, allow_reuse=reuse_env)
    return_trajectory = 0
    traj_states, traj_actions = [], []
    ppo = {
        "rewards": [],
        "log_probs": [],
        "values": [],
        "dones": [],
        "has_log_probs": hasattr(policy, "last_log_probs") and callable(policy.last_log_probs),
        "has_values": hasattr(policy, "last_values") and callable(policy.last_values),
    }

    def _draw():
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        clear_output(wait=True)
        plt.imshow(env.render())
        plt.title(f"i_iter = {i_iter} return = {return_trajectory:.2f}")
        plt.show()

    if hasattr(policy, "reset_state"):
        policy.reset_state()

    state, _ = env.reset(seed=noise_seed)
    state_policy = np.asarray(_obs_for_policy(state), dtype=np.float32)
    lb, width = _resolve_obs_bounds(env.observation_space, state_policy)
    max_steps = _resolve_max_episode_steps(env_conf)
    transform_state = b_gym and env_conf.gym_conf.transform_state
    prev_action, prev_reward = 0, 0.0
    recurrent = getattr(policy, "_recurrent", False)

    step_count = 0
    for i_iter in range(max_steps):
        state_policy = np.asarray(_obs_for_policy(state), dtype=np.float32)
        state_p = (state_policy - lb) / width
        if recurrent:
            action_p = policy(
                state_p if transform_state else state_policy,
                prev_action=prev_action,
                prev_reward=prev_reward,
            )
        else:
            action_p = policy(state_p if transform_state else state_policy)
        action = _scale_action_to_space(action_p, env.action_space)
        traj_states.append(state_p)
        traj_actions.append(action_p)

        state, reward, terminated, truncated, _ = _unpack_step_result(env.step(action))
        step_count += 1
        _collect_ppo_step(policy, ppo, reward, terminated or truncated)

        if recurrent:
            prev_action = int(action) if isinstance(action, (int, np.integer)) else int(np.asarray(action).item())
            prev_reward = float(reward)
        return_trajectory += reward
        if show_frames and i_iter % max(1, max_steps // num_frames_skip) == 0:
            _draw()
        if terminated or truncated:
            break

    if show_frames:
        _draw()
    if not keep_open:
        env.close()
    rreturn = _make_return(policy, return_trajectory)
    return Trajectory(
        rreturn,
        np.array(traj_states).T,
        np.array(traj_actions).T,
        rreturn_se=None,
        num_steps=int(step_count),
        rewards=np.array(ppo["rewards"], dtype=np.float32),
        log_probs=np.array(ppo["log_probs"], dtype=np.float32) if ppo["log_probs"] else None,
        values=np.array(ppo["values"], dtype=np.float32) if ppo["values"] else None,
        dones=np.array(ppo["dones"], dtype=bool),
    )
