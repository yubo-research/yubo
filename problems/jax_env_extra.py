from __future__ import annotations

from problems import jax_env_core as core
from problems.jax_env_base import GymnaxLikeAdapter


class CraftaxAdapter(GymnaxLikeAdapter):
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        raw_name = env_name.split(":", 1)[1]
        try:
            env, params = core._make_gymnax_env(raw_name)
        except Exception:
            env, params = self._make_craftax(raw_name)
        super().__init__(env_name, jax=jax, jnp=jnp, env=env, env_params=params)

    @staticmethod
    def _make_craftax(raw_name: str):
        import craftax

        for factory_name in ("make", "make_craftax_env_from_name"):
            factory = getattr(craftax, factory_name, None)
            if callable(factory):
                made = factory(raw_name)
                if isinstance(made, tuple) and len(made) == 2:
                    return made
                return made, core._default_env_params(made)

        if "Classic" in raw_name:
            from craftax.craftax.envs.craftax_classic_symbolic_env import (
                CraftaxClassicSymbolicEnv,
            )

            env = CraftaxClassicSymbolicEnv()
        else:
            from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

            env = CraftaxSymbolicEnv()
        return env, core._default_env_params(env)


class JumanjiAdapter:
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        import jumanji
        from gymnax.environments import spaces

        self._jax = jax
        self._jnp = jnp
        self.env = jumanji.make(env_name.split(":", 1)[1])
        key = jax.random.key(0)
        _state, timestep = self.env.reset(key)
        obs = core._flat_obs(timestep.observation, jax, jnp)
        self.observation_space = core._gymnax_box_from_shape(spaces, jnp, tuple(int(v) for v in obs.shape))
        self.action_space = core._spec_to_space(core._action_spec(self.env), spaces, jnp)

    def reset(self, key):
        state, timestep = self.env.reset(key)
        return core._flat_obs(timestep.observation, self._jax, self._jnp), state

    def step(self, _key, state, action):
        next_state, timestep = self.env.step(state, action)
        terminated = timestep.last().astype(self._jnp.float32)
        truncated = self._jnp.zeros_like(terminated)
        return core.JaxStepResult(
            obs=core._flat_obs(timestep.observation, self._jax, self._jnp),
            state=next_state,
            reward=timestep.reward,
            terminated=terminated,
            truncated=truncated,
            info={},
        )

    def clip_action(self, action):
        return core._clip_box_action(self.action_space, self._jnp, action)


class KinetixAdapter(GymnaxLikeAdapter):
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        raw_name = env_name.split(":", 1)[1]
        env, params = self._make_kinetix(raw_name)
        super().__init__(env_name, jax=jax, jnp=jnp, env=env, env_params=params)

    @staticmethod
    def _make_kinetix(raw_name: str):
        import kinetix

        for factory_name in (
            "make",
            "make_kinetix_env_from_name",
            "make_env_from_name",
        ):
            factory = getattr(kinetix, factory_name, None)
            if callable(factory):
                made = factory(raw_name)
                if isinstance(made, tuple) and len(made) == 2:
                    return made
                return made, core._default_env_params(made)
        try:
            from kinetix.environment.env import make_kinetix_env_from_name
        except ImportError:
            from kinetix.environment import make_kinetix_env_from_name
        made = make_kinetix_env_from_name(raw_name)
        if isinstance(made, tuple) and len(made) == 2:
            return made
        return made, core._default_env_params(made)
