from __future__ import annotations

from typing import Any, TypeAlias

from jaxtyping import Array, Float, PRNGKeyArray

from problems import jax_env_core as core

Obs: TypeAlias = Any
State: TypeAlias = Any
Action: TypeAlias = Any


class GymnaxAdapter:
    def __init__(self, env_name: str, *, jax, jnp, gymnax=None) -> None:
        if gymnax is None:
            import gymnax

        self._jnp = jnp
        self.env, self.env_params = gymnax.make(env_name.split(":", 1)[1])
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_space = self.env.action_space(self.env_params)

    def reset(self, key: PRNGKeyArray) -> tuple[Obs, State]:
        obs, state = self.env.reset(key, self.env_params)
        return obs, state

    def step(self, key: PRNGKeyArray, state: State, action: Action) -> core.JaxStepResult:
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

    def clip_action(self, action: Float[Array, "..."]) -> Float[Array, "..."]:
        return core._clip_box_action(self.action_space, self._jnp, action)


class GymnaxLikeAdapter:
    def __init__(self, env_name: str, *, jax, jnp, env: Any, env_params: Any | None = None) -> None:
        from gymnax.environments import spaces

        self._jnp = jnp
        self._jax = jax
        self.env = env
        self.env_params = core._default_env_params(env) if env_params is None else env_params
        if hasattr(env, "observation_space") and hasattr(env, "action_space"):
            self.observation_space = core._call_space(env.observation_space, self.env_params)
            self.action_space = core._call_space(env.action_space, self.env_params)
        else:
            obs, _state = self.reset(jax.random.key(0))
            self.observation_space = core._space_from_sample(obs, spaces, jax, jnp)
            if hasattr(env, "action_spec"):
                self.action_space = core._spec_to_space(core._action_spec(env), spaces, jnp)
            else:
                self.action_space = spaces.Discrete(2)

    def reset(self, key: PRNGKeyArray) -> tuple[Obs, State]:
        # Canonical interface is reset(key)->(obs,state). If the wrapped env does not
        # accept params (or returns state only), we normalize here (not in core).
        try:
            if self.env_params is None:
                out = self.env.reset(key)
            else:
                out = self.env.reset(key, self.env_params)
        except TypeError:
            out = self.env.reset(key)

        if isinstance(out, tuple) and len(out) == 2:
            return out
        # Some envs return only state with an `.obs` field.
        if hasattr(out, "obs"):
            return out.obs, out
        raise TypeError(f"Unsupported reset() return type for GymnaxLikeAdapter: {type(out).__name__}")

    def step(self, key: PRNGKeyArray, state: State, action: Action) -> core.JaxStepResult:
        try:
            if self.env_params is None:
                obs, next_state, reward, done, info = self.env.step(key, state, action)
            else:
                obs, next_state, reward, done, info = self.env.step(key, state, action, self.env_params)
        except TypeError:
            obs, next_state, reward, done, info = self.env.step(key, state, action)
        return core.JaxStepResult(
            obs=obs,
            state=next_state,
            reward=reward,
            terminated=done.astype(self._jnp.float32),
            truncated=self._jnp.zeros_like(done, dtype=self._jnp.float32),
            info=info,
        )

    def clip_action(self, action: Float[Array, "..."]) -> Float[Array, "..."]:
        return core._clip_box_action(self.action_space, self._jnp, action)
