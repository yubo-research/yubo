from __future__ import annotations

from typing import Any

from problems import jax_env_core as core


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
