from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _callback_or_direct(jnp, result_shape, callback, *args):
    if getattr(jnp, "__name__", "") == "numpy":
        return callback(*args)
    import jax

    return jax.pure_callback(
        callback,
        result_shape,
        *args,
        vmap_method="sequential",
    )


@dataclass(frozen=True)
class GymnasiumMujocoSpec:
    env_id: str
    family: str
    frame_skip: int
    max_episode_steps: int
    reset_noise_scale: float
    qvel_noise_type: str
    obs_dim_value: int
    oracle: Any
    bindings: dict[str, Any] = field(default_factory=dict)

    def obs(self, data, model, jnp):
        del model
        if getattr(jnp, "__name__", "") == "numpy":
            return self.oracle.obs(data.qpos, data.qvel)
        import jax

        return _callback_or_direct(
            jnp,
            jax.ShapeDtypeStruct((self.obs_dim_value,), jnp.float32),
            self.oracle.obs,
            data.qpos,
            data.qvel,
        )

    def obs_dim(self, model) -> int:
        del model
        return int(self.obs_dim_value)

    def reward_info(self, data_before, data_after, action, model, jnp):
        del model
        if getattr(jnp, "__name__", "") == "numpy":
            _obs, reward, _terminated, _truncated, info_values = self.oracle.step(
                data_before.qpos,
                data_before.qvel,
                data_after.qpos,
                data_after.qvel,
                action,
            )
        else:
            import jax

            obs_shape = jax.ShapeDtypeStruct((self.obs_dim_value,), jnp.float32)
            scalar = jax.ShapeDtypeStruct((), jnp.float32)
            boolean = jax.ShapeDtypeStruct((), bool)
            _obs, reward, _terminated, _truncated, info_values = _callback_or_direct(
                jnp,
                (obs_shape, scalar, boolean, boolean, (scalar, scalar, scalar, scalar)),
                self.oracle.step,
                data_before.qpos,
                data_before.qvel,
                data_after.qpos,
                data_after.qvel,
                action,
            )
        x_position, x_velocity, reward_forward, reward_ctrl = info_values
        return jnp.asarray(reward, dtype=jnp.float32), {
            "x_position": jnp.asarray(x_position, dtype=jnp.float32),
            "x_velocity": jnp.asarray(x_velocity, dtype=jnp.float32),
            "reward_forward": jnp.asarray(reward_forward, dtype=jnp.float32),
            "reward_ctrl": jnp.asarray(reward_ctrl, dtype=jnp.float32),
        }

    def step_semantics(self, data_before, data_after, action, model, jnp):
        del model
        if getattr(jnp, "__name__", "") == "numpy":
            obs, reward, terminated, truncated, info_values = self.oracle.step(
                data_before.qpos,
                data_before.qvel,
                data_after.qpos,
                data_after.qvel,
                action,
            )
        else:
            import jax

            obs_shape = jax.ShapeDtypeStruct((self.obs_dim_value,), jnp.float32)
            scalar = jax.ShapeDtypeStruct((), jnp.float32)
            boolean = jax.ShapeDtypeStruct((), bool)
            obs, reward, terminated, truncated, info_values = _callback_or_direct(
                jnp,
                (obs_shape, scalar, boolean, boolean, (scalar, scalar, scalar, scalar)),
                self.oracle.step,
                data_before.qpos,
                data_before.qvel,
                data_after.qpos,
                data_after.qvel,
                action,
            )
        x_position, x_velocity, reward_forward, reward_ctrl = info_values
        return (
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(terminated, dtype=bool),
            jnp.asarray(truncated, dtype=bool),
            {
                "x_position": jnp.asarray(x_position, dtype=jnp.float32),
                "x_velocity": jnp.asarray(x_velocity, dtype=jnp.float32),
                "reward_forward": jnp.asarray(reward_forward, dtype=jnp.float32),
                "reward_ctrl": jnp.asarray(reward_ctrl, dtype=jnp.float32),
            },
        )

    def terminated(self, data, jnp):
        del data
        return jnp.asarray(self.oracle.terminated(), dtype=bool)


class _HalfCheetahGymnasiumOracle:
    def __init__(self, env) -> None:
        self._env = env
        self._unwrapped = env.unwrapped
        self._lock = threading.Lock()
        self.obs_dim = int(np.prod(tuple(int(v) for v in env.observation_space.shape)))
        self.dt = float(self._unwrapped.dt)

    def obs(self, qpos, qvel):
        with self._lock:
            self._set_state(qpos, qvel)
            return np.asarray(self._unwrapped._get_obs(), dtype=np.float32)

    def step(self, qpos_before, qvel_before, qpos_after, qvel_after, action):
        qpos_before = np.asarray(qpos_before, dtype=np.float64)
        qvel_before = np.asarray(qvel_before, dtype=np.float64)
        qpos_after = np.asarray(qpos_after, dtype=np.float64)
        qvel_after = np.asarray(qvel_after, dtype=np.float64)
        action = np.asarray(action, dtype=np.float32)
        with self._lock:
            self._set_state(qpos_before, qvel_before)
            original_do_simulation = self._unwrapped.do_simulation

            def set_mjx_next_state(_action, _frame_skip):
                self._set_state(qpos_after, qvel_after)

            self._unwrapped.do_simulation = set_mjx_next_state
            try:
                obs, reward, terminated, truncated, info = self._unwrapped.step(action)
            finally:
                self._unwrapped.do_simulation = original_do_simulation
        info_values = (
            np.asarray(info["x_position"], dtype=np.float32),
            np.asarray(info["x_velocity"], dtype=np.float32),
            np.asarray(info["reward_forward"], dtype=np.float32),
            np.asarray(info["reward_ctrl"], dtype=np.float32),
        )
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            info_values,
        )

    @staticmethod
    def terminated():
        return np.asarray(False, dtype=bool)

    def close(self) -> None:
        self._env.close()

    def _set_state(self, qpos, qvel) -> None:
        import mujoco

        self._unwrapped.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self._unwrapped.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        try:
            mujoco.mj_forward(self._unwrapped.model, self._unwrapped.data)
        except (AttributeError, TypeError):
            pass


def _require_halfcheetah_oracle(env):
    unwrapped = env.unwrapped
    missing = [name for name in ("_get_obs", "_get_rew", "data", "do_simulation", "model", "step") if not hasattr(unwrapped, name)]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Gymnasium HalfCheetah MJX semantics require Gymnasium-owned methods/fields; missing: {missing_text}.")
    return _HalfCheetahGymnasiumOracle(env)


def _bind_halfcheetah(env) -> GymnasiumMujocoSpec:
    unwrapped = env.unwrapped
    env_id = str(getattr(getattr(unwrapped, "spec", None), "id", "HalfCheetah-v5"))
    max_episode_steps = int(getattr(getattr(unwrapped, "spec", None), "max_episode_steps", 1000) or 1000)
    oracle = _require_halfcheetah_oracle(env)
    return GymnasiumMujocoSpec(
        env_id=env_id,
        family="gymnasium_mujoco",
        frame_skip=int(getattr(unwrapped, "frame_skip", 5)),
        max_episode_steps=max_episode_steps,
        reset_noise_scale=float(getattr(unwrapped, "_reset_noise_scale", 0.1)),
        qvel_noise_type="normal",
        obs_dim_value=oracle.obs_dim,
        oracle=oracle,
        bindings={
            "semantics_owner": "gymnasium",
            "obs_owner": f"{type(unwrapped).__module__}.{type(unwrapped).__name__}._get_obs",
            "reward_owner": f"{type(unwrapped).__module__}.{type(unwrapped).__name__}._get_rew",
        },
    )


def resolve_gymnasium_mujoco_spec(env) -> GymnasiumMujocoSpec | None:
    root_env = env if hasattr(env, "unwrapped") else None
    unwrapped = getattr(env, "unwrapped", env)
    env_id = str(getattr(getattr(unwrapped, "spec", None), "id", ""))
    if env_id.startswith("HalfCheetah-"):
        if root_env is None:
            import gymnasium as gym

            root_env = gym.make(env_id)
        return _bind_halfcheetah(root_env)
    return None


def supported_gymnasium_mujoco_specs() -> tuple[str, ...]:
    return ("HalfCheetah-v5",)
