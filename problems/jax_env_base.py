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
        from gymnax.environments import spaces

        # Robust action handling: squeeze if environment is discrete or action is 1-element
        if isinstance(self.action_space, spaces.Discrete) or (hasattr(action, "shape") and action.shape == (1,)):
            action = self._jnp.squeeze(action).astype(self._jnp.int32)

        obs, next_state, reward, done, info = self.env.step(key, state, action, self.env_params)
        return core.JaxStepResult(
            obs=obs,
            state=next_state,
            reward=reward.astype(self._jnp.float32),
            terminated=done.astype(self._jnp.float32),
            truncated=self._jnp.zeros_like(done, dtype=self._jnp.float32),
            info=info,
        )

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
            obs, next_state, reward, done, info = self.env.step(key, state, action)
        else:
            obs, next_state, reward, done, info = self.env.step(key, state, action, self.env_params)
        return core.JaxStepResult(
            obs=obs,
            state=next_state,
            reward=reward,
            terminated=done.astype(self._jnp.float32),
            truncated=self._jnp.zeros_like(done, dtype=self._jnp.float32),
            info=info,
        )

    def clip_action(self, action):
        return core._clip_box_action(self.action_space, self._jnp, action)
