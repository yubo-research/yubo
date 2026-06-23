from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from problems.mujoco_gl import normalize_mujoco_gl_for_platform


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


@dataclass(frozen=True)
class _HalfCheetahJaxSemantics:
    env_id: str
    family: str
    frame_skip: int
    max_episode_steps: int
    reset_noise_scale: float
    qvel_noise_type: str
    obs_dim_value: int
    obs_qpos_start: int
    dt: float
    forward_reward_weight: float
    ctrl_cost_weight: float
    bindings: dict[str, Any] = field(default_factory=dict)

    def obs(self, data, model, jnp):
        del model
        return jnp.concatenate(
            [
                jnp.ravel(data.qpos[self.obs_qpos_start :]),
                jnp.ravel(data.qvel),
            ],
            axis=0,
        ).astype(jnp.float32)

    def obs_dim(self, model) -> int:
        del model
        return int(self.obs_dim_value)

    def reward_info(self, data_before, data_after, action, model, jnp):
        del model
        x_position = data_after.qpos[0]
        x_velocity = (x_position - data_before.qpos[0]) / jnp.asarray(
            self.dt,
            dtype=jnp.float32,
        )
        reward_forward = self.forward_reward_weight * x_velocity
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(action * action)
        reward_ctrl = -ctrl_cost
        return jnp.asarray(reward_forward + reward_ctrl, dtype=jnp.float32), {
            "x_position": x_position.astype(jnp.float32),
            "x_velocity": x_velocity.astype(jnp.float32),
            "reward_forward": reward_forward.astype(jnp.float32),
            "reward_ctrl": reward_ctrl.astype(jnp.float32),
        }

    def step_semantics(self, data_before, data_after, action, model, jnp):
        obs = self.obs(data_after, model, jnp)
        reward, info = self.reward_info(data_before, data_after, action, model, jnp)
        return (
            obs,
            reward,
            jnp.asarray(False, dtype=bool),
            jnp.asarray(False, dtype=bool),
            info,
        )

    def terminated(self, data, jnp):
        del data
        return jnp.asarray(False, dtype=bool)

    @property
    def oracle(self):
        return None


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
        normalize_mujoco_gl_for_platform()
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


def _halfcheetah_fast_spec(env) -> _HalfCheetahJaxSemantics:
    unwrapped = env.unwrapped
    env_id = str(getattr(getattr(unwrapped, "spec", None), "id", "HalfCheetah-v5"))
    max_episode_steps = int(getattr(getattr(unwrapped, "spec", None), "max_episode_steps", 1000) or 1000)
    obs_dim = int(np.prod(tuple(int(v) for v in env.observation_space.shape)))
    exclude_current_positions = bool(getattr(unwrapped, "_exclude_current_positions_from_observation", True))
    return _HalfCheetahJaxSemantics(
        env_id=env_id,
        family="gymnasium_mujoco",
        frame_skip=int(getattr(unwrapped, "frame_skip", 5)),
        max_episode_steps=max_episode_steps,
        reset_noise_scale=float(getattr(unwrapped, "_reset_noise_scale", 0.1)),
        qvel_noise_type="normal",
        obs_dim_value=obs_dim,
        obs_qpos_start=1 if exclude_current_positions else 0,
        dt=float(unwrapped.dt),
        forward_reward_weight=float(getattr(unwrapped, "_forward_reward_weight", 1.0)),
        ctrl_cost_weight=float(getattr(unwrapped, "_ctrl_cost_weight", 0.1)),
        bindings={
            "semantics_owner": "yubo_jax_fast",
            "obs_owner": "problems.gymnasium_mujoco_specs._HalfCheetahJaxSemantics.obs",
            "reward_owner": "problems.gymnasium_mujoco_specs._HalfCheetahJaxSemantics.reward_info",
        },
    )


def _bind_halfcheetah(env, *, fast: bool) -> GymnasiumMujocoSpec | _HalfCheetahJaxSemantics:
    unwrapped = env.unwrapped
    if fast:
        return _halfcheetah_fast_spec(env)
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


def resolve_gymnasium_mujoco_spec(env, *, fast: bool = False):
    root_env = env if hasattr(env, "unwrapped") else None
    unwrapped = getattr(env, "unwrapped", env)
    env_id = str(getattr(getattr(unwrapped, "spec", None), "id", ""))
    if env_id.startswith("HalfCheetah-"):
        if root_env is None:
            import gymnasium as gym

            normalize_mujoco_gl_for_platform()
            root_env = gym.make(env_id)
        return _bind_halfcheetah(root_env, fast=fast)
    return None


def supported_gymnasium_mujoco_specs() -> tuple[str, ...]:
    return ("HalfCheetah-v5",)
