from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class GymnasiumMujocoSpec:
    env_id: str
    family: str
    frame_skip: int
    max_episode_steps: int
    reset_noise_scale: float
    qvel_noise_type: str
    obs_fn: Callable
    obs_dim_fn: Callable
    reward_info_fn: Callable
    terminated_fn: Callable
    bindings: dict[str, Any] = field(default_factory=dict)

    def obs(self, data, model, jnp):
        return self.obs_fn(data=data, model=model, jnp=jnp, spec=self)

    def obs_dim(self, model) -> int:
        return int(self.obs_dim_fn(model=model, spec=self))

    def reward_info(self, data_before, data_after, action, model, jnp):
        return self.reward_info_fn(
            data_before=data_before,
            data_after=data_after,
            action=action,
            model=model,
            jnp=jnp,
            spec=self,
        )

    def terminated(self, data, jnp):
        return self.terminated_fn(data=data, jnp=jnp, spec=self)


def _qpos_qvel_obs(*, data, model, jnp, spec: GymnasiumMujocoSpec):
    del model
    obs_qpos_start = int(spec.bindings["obs_qpos_start"])
    return jnp.concatenate(
        [jnp.ravel(data.qpos[obs_qpos_start:]), jnp.ravel(data.qvel)],
        axis=0,
    ).astype(jnp.float32)


def _qpos_qvel_obs_dim(*, model, spec: GymnasiumMujocoSpec) -> int:
    obs_qpos_start = int(spec.bindings["obs_qpos_start"])
    return max(int(model.nq) - obs_qpos_start, 0) + int(model.nv)


def _halfcheetah_reward_info(*, data_before, data_after, action, model, jnp, spec: GymnasiumMujocoSpec):
    del model
    bindings = spec.bindings
    dt = jnp.maximum(data_after.time - data_before.time, jnp.asarray(1e-6, dtype=jnp.float32))
    x_velocity = (data_after.qpos[0] - data_before.qpos[0]) / dt
    forward_reward = float(bindings["forward_reward_weight"]) * x_velocity
    ctrl_cost = float(bindings["ctrl_cost_weight"]) * jnp.sum(action * action)
    reward = forward_reward - ctrl_cost
    return jnp.asarray(reward, dtype=jnp.float32), {
        "x_position": data_after.qpos[0],
        "x_velocity": x_velocity,
        "reward_forward": forward_reward,
        "reward_ctrl": -ctrl_cost,
    }


def _never_terminated(*, data, jnp, spec: GymnasiumMujocoSpec):
    del data, spec
    return jnp.asarray(False, dtype=bool)


def _bind_halfcheetah(env) -> GymnasiumMujocoSpec:
    env_id = str(getattr(getattr(env, "spec", None), "id", "HalfCheetah-v5"))
    max_episode_steps = int(getattr(getattr(env, "spec", None), "max_episode_steps", 1000) or 1000)
    exclude_current_positions = bool(getattr(env, "_exclude_current_positions_from_observation", True))
    return GymnasiumMujocoSpec(
        env_id=env_id,
        family="gymnasium_mujoco",
        frame_skip=int(getattr(env, "frame_skip", 5)),
        max_episode_steps=max_episode_steps,
        reset_noise_scale=float(getattr(env, "_reset_noise_scale", 0.1)),
        qvel_noise_type="normal",
        obs_fn=_qpos_qvel_obs,
        obs_dim_fn=_qpos_qvel_obs_dim,
        reward_info_fn=_halfcheetah_reward_info,
        terminated_fn=_never_terminated,
        bindings={
            "obs_qpos_start": 1 if exclude_current_positions else 0,
            "forward_reward_weight": float(getattr(env, "_forward_reward_weight", 1.0)),
            "ctrl_cost_weight": float(getattr(env, "_ctrl_cost_weight", 0.1)),
        },
    )


def resolve_gymnasium_mujoco_spec(env) -> GymnasiumMujocoSpec | None:
    env_id = str(getattr(getattr(env, "spec", None), "id", ""))
    if env_id.startswith("HalfCheetah-"):
        return _bind_halfcheetah(env)
    return None


def supported_gymnasium_mujoco_specs() -> tuple[str, ...]:
    return ("HalfCheetah-v5",)
