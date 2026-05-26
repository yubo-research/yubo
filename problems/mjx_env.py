from __future__ import annotations

from problems import jax_env_core as core

GYMNASIUM_ENV_PREFIX = "gymnasium:"


def is_gymnasium_env_tag(env_name: str) -> bool:
    return str(env_name).startswith(GYMNASIUM_ENV_PREFIX)


def parse_gymnasium_env_id(env_name: str) -> str:
    tag = str(env_name)
    if not is_gymnasium_env_tag(tag):
        raise ValueError(f"Unsupported Gymnasium env tag {tag!r}. Expected 'gymnasium:<EnvId>'.")
    env_id = tag.split(":", 1)[1].strip()
    if not env_id:
        raise ValueError(f"Gymnasium env tag {tag!r} is missing an env id.")
    return env_id


def _load_gymnasium_model(env_id: str):
    import gymnasium as gym

    env = gym.make(env_id)
    try:
        model = getattr(env.unwrapped, "model", None)
        if model is None:
            raise TypeError(f"Gymnasium env {env_id!r} does not expose an unwrapped MuJoCo model.")
        return model
    finally:
        env.close()


def _qpos_obs(qpos, *, nq: int, qpos0, jnp):
    if int(nq) <= 1:
        return qpos
    # Gymnasium MuJoCo tasks commonly hide absolute x position.
    return qpos[1:] if qpos0.shape[0] == int(nq) else qpos


def _flat_obs(data, model, jnp):
    qpos = _qpos_obs(data.qpos, nq=int(model.nq), qpos0=jnp.asarray(model.qpos0), jnp=jnp)
    parts = [jnp.ravel(qpos), jnp.ravel(data.qvel)]
    if int(model.nsensor) > 0:
        parts.append(jnp.ravel(data.sensordata))
    return jnp.concatenate(parts, axis=0).astype(jnp.float32)


def _obs_dim(model) -> int:
    qpos_dim = max(int(model.nq) - 1, 0)
    return int(qpos_dim + int(model.nv) + int(model.nsensordata))


def _action_bounds(model, jnp):
    if int(model.nu) <= 0:
        return jnp.zeros((0,), dtype=jnp.float32), jnp.zeros((0,), dtype=jnp.float32)
    ctrlrange = jnp.asarray(model.actuator_ctrlrange, dtype=jnp.float32)
    limited = jnp.asarray(model.actuator_ctrllimited, dtype=bool)
    low = jnp.where(limited, ctrlrange[:, 0], -1.0)
    high = jnp.where(limited, ctrlrange[:, 1], 1.0)
    return low.astype(jnp.float32), high.astype(jnp.float32)


class GymnasiumMJXAdapter:
    """MJX execution backend for Gymnasium MuJoCo registry envs."""

    def __init__(self, env_name: str, *, jax, jnp) -> None:
        from gymnax.environments import spaces
        from mujoco import mjx

        self._jax = jax
        self._jnp = jnp
        self._mjx = mjx
        self.env_id = parse_gymnasium_env_id(env_name)
        self.model = _load_gymnasium_model(self.env_id)
        self.mjx_model = mjx.put_model(self.model)
        obs_shape = (_obs_dim(self.model),)
        low, high = _action_bounds(self.model, jnp)
        self.observation_space = core._gymnax_box_from_shape(spaces, jnp, obs_shape)
        self.action_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=jnp.float32)

    def reset(self, key):
        data = self._mjx.make_data(self.mjx_model)
        qpos_noise, qvel_noise = self._reset_noise(key)
        data = data.replace(
            qpos=self._jnp.asarray(self.model.qpos0, dtype=self._jnp.float32) + qpos_noise,
            qvel=self._jnp.zeros((int(self.model.nv),), dtype=self._jnp.float32) + qvel_noise,
        )
        return _flat_obs(data, self.model, self._jnp), data

    def step(self, _key, state, action):
        action = self.clip_action(action)
        next_state = self._mjx.step(self.mjx_model, state.replace(ctrl=action))
        obs = _flat_obs(next_state, self.model, self._jnp)
        reward = self._generic_reward(state, next_state)
        done = self._jnp.asarray(False, dtype=self._jnp.float32)
        return obs, next_state, reward, done, {}

    def clip_action(self, action):
        return core._clip_box_action(self.action_space, self._jnp, action)

    def _reset_noise(self, key):
        qpos_key, qvel_key = self._jax.random.split(key)
        qpos_noise = self._jax.random.uniform(
            qpos_key,
            (int(self.model.nq),),
            minval=-0.01,
            maxval=0.01,
            dtype=self._jnp.float32,
        )
        qvel_noise = self._jax.random.normal(qvel_key, (int(self.model.nv),), dtype=self._jnp.float32) * 0.01
        return qpos_noise, qvel_noise

    def _generic_reward(self, state, next_state):
        if int(self.model.nq) <= 0:
            return self._jnp.asarray(0.0, dtype=self._jnp.float32)
        dt = self._jnp.asarray(float(self.model.opt.timestep), dtype=self._jnp.float32)
        forward = (next_state.qpos[0] - state.qpos[0]) / self._jnp.maximum(dt, 1e-6)
        ctrl_cost = 1e-3 * self._jnp.sum(next_state.ctrl * next_state.ctrl)
        return self._jnp.asarray(forward - ctrl_cost, dtype=self._jnp.float32)
