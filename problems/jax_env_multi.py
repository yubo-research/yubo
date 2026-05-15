from __future__ import annotations

import math

from problems import jax_env_core as core
from problems.jax_env_base import GymnaxLikeAdapter


class NavixAdapter(GymnaxLikeAdapter):
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        raw_name = env_name.split(":", 1)[1]
        env, params = self._make_navix(raw_name)
        super().__init__(env_name, jax=jax, jnp=jnp, env=env, env_params=params)

    @staticmethod
    def _make_navix(raw_name: str):
        import navix

        for factory_name in ("make", "make_env"):
            factory = getattr(navix, factory_name, None)
            if callable(factory):
                made = factory(raw_name)
                if isinstance(made, tuple) and len(made) == 2:
                    return made
                return made, core._default_env_params(made)
        raise ImportError("Navix is installed, but no supported environment factory was found.")


class JaxMARLAdapter:
    _NAME_ALIASES = {
        "mpe-simple-reference-v3": "MPE_simple_reference_v3",
        "mpe-simple-speaker-listener-v4": "MPE_simple_speaker_listener_v4",
        "mpe-simple-spread-v3": "MPE_simple_spread_v3",
    }

    def __init__(self, env_name: str, *, jax, jnp) -> None:
        import jaxmarl
        from gymnax.environments import spaces

        self._jax = jax
        self._jnp = jnp
        raw_name = env_name.split(":", 1)[1]
        candidates = [
            raw_name,
            self._NAME_ALIASES.get(raw_name, raw_name),
            raw_name.replace("-", "_"),
        ]
        last_exc = None
        for candidate in dict.fromkeys(candidates):
            try:
                self.env = jaxmarl.make(candidate)
                break
            except Exception as exc:  # noqa: BLE001 - try documented and normalized names.
                last_exc = exc
        else:
            raise last_exc or ValueError(f"Unable to construct JaxMARL env {raw_name!r}.")

        self.agents = tuple(getattr(self.env, "agents", ()))
        if not self.agents:
            self.agents = tuple(getattr(self.env, "possible_agents", ()))
        if not self.agents:
            raise ValueError(f"JaxMARL env {raw_name!r} did not expose agents.")

        key = jax.random.key(0)
        obs, _state = self.env.reset(key)
        self.observation_space = core._space_from_sample(self._ordered_obs(obs), spaces, jax, jnp)
        self._action_sizes = tuple(self._agent_action_size(agent) for agent in self.agents)
        max_action = max(self._action_sizes) - 1
        self.action_space = spaces.Box(
            low=jnp.zeros((len(self.agents),), dtype=jnp.float32),
            high=jnp.full((len(self.agents),), max_action, dtype=jnp.float32),
            shape=(len(self.agents),),
            dtype=jnp.float32,
        )

    def _ordered_obs(self, obs):
        return tuple(obs[agent] for agent in self.agents)

    def _agent_action_size(self, agent) -> int:
        space = self.env.action_space(agent)
        if hasattr(space, "n"):
            return int(space.n)
        if hasattr(space, "num_values"):
            return int(space.num_values)
        if hasattr(space, "shape"):
            return int(math.prod(tuple(int(v) for v in space.shape)))
        raise TypeError(f"Unsupported JaxMARL action space for agent {agent!r}: {type(space).__name__}")

    def reset(self, key):
        obs, state = self.env.reset(key)
        return core._flat_obs(self._ordered_obs(obs), self._jax, self._jnp), state

    def step(self, key, state, action):
        actions = action if isinstance(action, dict) else self.clip_action(action)
        obs, next_state, rewards, dones, infos = self.env.step(key, state, actions)
        reward_values = [self._jnp.asarray(rewards[agent], dtype=self._jnp.float32) for agent in self.agents]
        reward = self._jnp.mean(self._jnp.stack(reward_values))
        done = dones.get("__all__", None) if hasattr(dones, "get") else None
        if done is None:
            done_values = [self._jnp.asarray(dones[agent]) for agent in self.agents]
            done = self._jnp.all(self._jnp.stack(done_values))
        result = (
            core._flat_obs(self._ordered_obs(obs), self._jax, self._jnp),
            next_state,
            reward,
            done,
            infos,
        )
        return result

    def clip_action(self, action):
        if isinstance(action, dict):
            return {
                agent: self._jnp.clip(
                    self._jnp.rint(action[agent]).astype(self._jnp.int32),
                    0,
                    int(size) - 1,
                )
                for agent, size in zip(self.agents, self._action_sizes, strict=True)
            }
        action = self._jnp.ravel(self._jnp.asarray(action))
        action = self._jnp.rint(action).astype(self._jnp.int32)
        out = {}
        for idx, (agent, size) in enumerate(zip(self.agents, self._action_sizes, strict=True)):
            out[agent] = self._jnp.clip(action[idx], 0, int(size) - 1)
        return out
