from __future__ import annotations

from typing import Any

from problems import jax_env_core as core


def _set_headless_mujoco_gl_default() -> None:
    import os
    import sys

    if not sys.platform.startswith("linux"):
        return
    if os.environ.get("MUJOCO_GL") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    os.environ["MUJOCO_GL"] = "egl"


class GymnaxAdapter:
    def __init__(self, env_name: str, *, jax, jnp, gymnax=None) -> None:
        if gymnax is None:
            import gymnax

        self._jnp = jnp
        self.env, self.env_params = gymnax.make(env_name.split(":", 1)[1])
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_space = self.env.action_space(self.env_params)

    def reset(self, key):
        return self.env.reset(key, self.env_params)

    def step(self, key, state, action):
        return self.env.step(key, state, action, self.env_params)

    def clip_action(self, action):
        return core._clip_box_action(self.action_space, self._jnp, action)


class GymnaxLikeAdapter:
    def __init__(self, env_name: str, *, jax, jnp, env: Any, env_params: Any | None = None) -> None:
        self._jnp = jnp
        self.env = env
        self.env_params = core._default_env_params(env) if env_params is None else env_params
        self.observation_space, self.action_space = core._make_gymnax_like_spaces(
            env,
            self.env_params,
            jax=jax,
            jnp=jnp,
        )

    def reset(self, key):
        if self.env_params is None:
            return self.env.reset(key)
        return self.env.reset(key, self.env_params)

    def step(self, key, state, action):
        if self.env_params is None:
            return self.env.step(key, state, action)
        return self.env.step(key, state, action, self.env_params)

    def clip_action(self, action):
        return core._clip_box_action(self.action_space, self._jnp, action)


class BraxAdapter:
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        _set_headless_mujoco_gl_default()
        from brax import envs
        from gymnax.environments import spaces

        self._jnp = jnp
        brax_name = env_name.split(":", 1)[1]
        self.env = envs.get_environment(brax_name)
        obs_shape = tuple(int(v) for v in jax.eval_shape(self.env.reset, jax.random.key(0)).obs.shape)
        act_shape = (int(self.env.action_size),)
        self.observation_space = core._gymnax_box_from_shape(spaces, jnp, obs_shape)
        self.action_space = core._gymnax_box_from_shape(spaces, jnp, act_shape, low=-1.0, high=1.0)

    def reset(self, key):
        state = self.env.reset(key)
        return self._jnp.asarray(state.obs, dtype=self._jnp.float32), state

    def step(self, _key, state, action):
        action = self.clip_action(action)
        next_state = self.env.step(state, action)
        result = (
            self._jnp.asarray(next_state.obs, dtype=self._jnp.float32),
            next_state,
            next_state.reward,
            next_state.done,
            {},
        )
        return result

    def clip_action(self, action):
        return core._clip_box_action(self.action_space, self._jnp, action)
