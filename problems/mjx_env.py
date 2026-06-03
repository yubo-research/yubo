from __future__ import annotations

from typing import Any, NamedTuple, TypeAlias

from jaxtyping import Array, Float, PRNGKeyArray

from problems import jax_env_core as core
from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

GYMNASIUM_ENV_PREFIX = "gymnasium:"
GYMNASIUM_FAST_ENV_PREFIX = "gymnasium_fast:"


def is_gymnasium_env_tag(env_name: str) -> bool:
    return str(env_name).startswith((GYMNASIUM_ENV_PREFIX, GYMNASIUM_FAST_ENV_PREFIX))


def is_gymnasium_fast_env_tag(env_name: str) -> bool:
    return str(env_name).startswith(GYMNASIUM_FAST_ENV_PREFIX)


def parse_gymnasium_env_id(env_name: str) -> str:
    tag = str(env_name)
    if not is_gymnasium_env_tag(tag):
        raise ValueError(f"Unsupported Gymnasium env tag {tag!r}. Expected 'gymnasium:<EnvId>' or 'gymnasium_fast:<EnvId>'.")
    env_id = tag.split(":", 1)[1].strip()
    if not env_id:
        raise ValueError(f"Gymnasium env tag {tag!r} is missing an env id.")
    return env_id


class GymnasiumMJXState(NamedTuple):
    data: Any
    steps: Any


Obs: TypeAlias = Any
State: TypeAlias = Any
Action: TypeAlias = Any


def _load_gymnasium_env_spec(env_id: str, *, fast: bool = False):
    import gymnasium as gym

    env = gym.make(env_id)
    try:
        model = getattr(env.unwrapped, "model", None)
        if model is None:
            raise TypeError(f"Gymnasium env {env_id!r} does not expose an unwrapped MuJoCo model.")
        spec = resolve_gymnasium_mujoco_spec(env, fast=fast)
        if spec is None:
            raise ValueError(f"Gymnasium MJX adapter has no verified MuJoCo semantics for {env_id!r}. Add a spec before routing this env through MJX.")
        return model, spec
    except Exception:
        env.close()
        raise


def _action_bounds(model, jnp):
    if int(model.nu) <= 0:
        return jnp.zeros((0,), dtype=jnp.float32), jnp.zeros((0,), dtype=jnp.float32)
    ctrlrange = jnp.asarray(model.actuator_ctrlrange, dtype=jnp.float32)
    limited = jnp.asarray(model.actuator_ctrllimited, dtype=bool)
    low = jnp.where(limited, ctrlrange[:, 0], -1.0)
    high = jnp.where(limited, ctrlrange[:, 1], 1.0)
    return low.astype(jnp.float32), high.astype(jnp.float32)


def _resolve_mjx_device(jax):
    try:
        return jax.devices()[0]
    except TypeError:
        try:
            cuda_devices = jax.devices("cuda")
        except (RuntimeError, AttributeError):
            cuda_devices = []
        if cuda_devices:
            return cuda_devices[0]
        return jax.devices("cpu")[0]


def _mjx_kwargs_for_device(mjx, device):
    if _mjx_device_is_mps(device):
        # MJX's JAX implementation can run on Apple MPS, but the automatic
        # implementation resolver in some MuJoCo builds does not classify the
        # MPS platform. Pin the implementation explicitly for that backend.
        return {"device": device, "impl": mjx.Impl.JAX}
    return {}


def _mjx_device_is_mps(device) -> bool:
    return str(getattr(device, "platform", "")).lower() == "mps"


class GymnasiumMJXAdapter:
    """MJX execution backend for Gymnasium MuJoCo registry envs."""

    def __init__(self, env_name: str, *, jax, jnp) -> None:
        from gymnax.environments import spaces
        from mujoco import mjx

        self._jax = jax
        self._jnp = jnp
        self._mjx = mjx
        self._mjx_device = _resolve_mjx_device(jax)
        self._mjx_kwargs = _mjx_kwargs_for_device(mjx, self._mjx_device)
        self.fast = is_gymnasium_fast_env_tag(env_name)
        self.env_id = parse_gymnasium_env_id(env_name)
        spec_fast = self.fast or _mjx_device_is_mps(self._mjx_device)
        self.model, self.spec = _load_gymnasium_env_spec(self.env_id, fast=spec_fast)
        self.mjx_model = mjx.put_model(self.model, **self._mjx_kwargs)
        obs_shape = (self.spec.obs_dim(self.model),)
        low, high = _action_bounds(self.model, jnp)
        self.observation_space = core._gymnax_box_from_shape(spaces, jnp, obs_shape)
        self.action_space = spaces.Box(low=low, high=high, shape=low.shape, dtype=jnp.float32)

    def reset(self, key: PRNGKeyArray) -> tuple[Obs, GymnasiumMJXState]:
        data = self._reset_data(key)
        return self.spec.obs(data, self.model, self._jnp), GymnasiumMJXState(
            data=data,
            steps=self._jnp.asarray(0, dtype=self._jnp.int32),
        )

    def step(self, key: PRNGKeyArray, state: GymnasiumMJXState, action: Action) -> core.JaxStepResult:
        step_key, reset_key = self._jax.random.split(key)
        del step_key
        action = self.clip_action(action)
        next_data = self._step_data(state.data, action)
        next_obs, reward, terminated, gym_truncated, info = self.spec.step_semantics(state.data, next_data, action, self.model, self._jnp)
        next_steps = state.steps + self._jnp.asarray(1, dtype=self._jnp.int32)
        time_limit_truncated = next_steps >= self._jnp.asarray(
            int(self.spec.max_episode_steps),
            dtype=self._jnp.int32,
        )
        truncated = self._jnp.logical_or(gym_truncated, time_limit_truncated)
        done_bool = self._jnp.logical_or(terminated, truncated)
        reset_data = self._reset_data(reset_key)
        reset_obs = self.spec.obs(reset_data, self.model, self._jnp)
        keep_state = GymnasiumMJXState(data=next_data, steps=next_steps)
        reset_state = GymnasiumMJXState(
            data=reset_data,
            steps=self._jnp.asarray(0, dtype=self._jnp.int32),
        )
        out_state = self._where_state(done_bool, reset_state, keep_state)
        obs = self._jnp.where(done_bool, reset_obs, next_obs)
        return core.JaxStepResult(
            obs=obs,
            state=out_state,
            reward=reward,
            terminated=terminated.astype(self._jnp.float32),
            truncated=truncated.astype(self._jnp.float32),
            info=info,
        )

    def clip_action(self, action: Float[Array, "..."]) -> Float[Array, "..."]:
        return core._clip_box_action(self.action_space, self._jnp, action)

    def close(self) -> None:
        close = getattr(getattr(self, "spec", None), "oracle", None)
        close_fn = getattr(close, "close", None)
        if callable(close_fn):
            close_fn()

    def _reset_data(self, key):
        data = self._mjx.make_data(self.mjx_model, **self._mjx_kwargs)
        qpos_noise, qvel_noise = self._reset_noise(key)
        data = data.replace(
            qpos=self._jnp.asarray(self.model.qpos0, dtype=self._jnp.float32) + qpos_noise,
            qvel=self._jnp.zeros((int(self.model.nv),), dtype=self._jnp.float32) + qvel_noise,
        )
        return self._mjx.forward(self.mjx_model, data)

    def _step_data(self, data, action):
        data = data.replace(ctrl=action)

        def step_once(carry, _):
            return self._mjx.step(self.mjx_model, carry), None

        next_data, _ = self._jax.lax.scan(step_once, data, xs=None, length=int(self.spec.frame_skip))
        return next_data

    def _where_state(self, condition, reset_state, keep_state):
        data = self._jax.tree_util.tree_map(
            lambda reset_value, keep_value: self._jnp.where(condition, reset_value, keep_value),
            reset_state.data,
            keep_state.data,
        )
        steps = self._jnp.where(condition, reset_state.steps, keep_state.steps)
        return GymnasiumMJXState(data=data, steps=steps)

    def _reset_noise(self, key):
        qpos_key, qvel_key = self._jax.random.split(key)
        reset_noise_scale = float(self.spec.reset_noise_scale)
        qpos_noise = self._jax.random.uniform(
            qpos_key,
            (int(self.model.nq),),
            minval=-reset_noise_scale,
            maxval=reset_noise_scale,
            dtype=self._jnp.float32,
        )
        qvel_noise = self._jax.random.normal(qvel_key, (int(self.model.nv),), dtype=self._jnp.float32) * reset_noise_scale
        return qpos_noise, qvel_noise
