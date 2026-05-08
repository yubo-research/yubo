from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


EGGROLL_JAX_ENV_PREFIXES = (
    "gymnax:",
    "brax:",
    "craftax:",
    "jaxmarl:",
    "jumanji:",
    "kinetix:",
    "navix:",
)
EGGROLL_SURROGATE_ENV_PREFIXES = (
    "passk:",
    "qwen:",
    "rwkv-int8-distill:",
    "jaxlob:",
    "synthetic:linear-speed",
)
EGGROLL_ADAPTER_ENV_PREFIXES = EGGROLL_JAX_ENV_PREFIXES + EGGROLL_SURROGATE_ENV_PREFIXES
EGGROLL_ENV_PREFIXES = EGGROLL_ADAPTER_ENV_PREFIXES


def supports_eggroll_env_adapter(env_name: str) -> bool:
    return str(env_name).startswith(EGGROLL_ADAPTER_ENV_PREFIXES)


def supports_eggroll_env(env_name: str) -> bool:
    return str(env_name).startswith(EGGROLL_ENV_PREFIXES)


def _stable_scale(text: str) -> float:
    total = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(text)))
    return float((total % 997) + 1) / 997.0


def _space_bounds(space: Any, jnp) -> tuple[Any, Any]:
    low = space.low if isinstance(space.low, jnp.ndarray) else space.low * jnp.ones(space.shape, dtype=space.dtype)
    high = space.high if isinstance(space.high, jnp.ndarray) else space.high * jnp.ones(space.shape, dtype=space.dtype)
    return jnp.asarray(low, dtype=jnp.float32), jnp.asarray(high, dtype=jnp.float32)


def _flat_obs(obs: Any, jax, jnp):
    leaves = jax.tree_util.tree_leaves(obs)
    flat = [jnp.ravel(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in leaves if hasattr(leaf, "shape")]
    if not flat:
        return jnp.asarray(obs, dtype=jnp.float32)
    if len(flat) == 1:
        return flat[0]
    return jnp.concatenate(flat, axis=0)


def _gymnax_box_from_shape(spaces, jnp, shape: tuple[int, ...], *, low=None, high=None):
    low = -jnp.inf if low is None else low
    high = jnp.inf if high is None else high
    return spaces.Box(
        low=jnp.full(shape, low, dtype=jnp.float32),
        high=jnp.full(shape, high, dtype=jnp.float32),
        shape=shape,
        dtype=jnp.float32,
    )


def _space_from_sample(sample: Any, spaces, jax, jnp):
    obs = _flat_obs(sample, jax, jnp)
    return _gymnax_box_from_shape(spaces, jnp, tuple(int(v) for v in obs.shape))


def _spec_to_space(spec: Any, spaces, jnp):
    if hasattr(spec, "num_values"):
        return spaces.Discrete(int(spec.num_values))
    if hasattr(spec, "maximum") and hasattr(spec, "minimum") and hasattr(spec, "shape"):
        shape = tuple(int(v) for v in spec.shape)
        minimum = jnp.asarray(spec.minimum)
        maximum = jnp.asarray(spec.maximum)
        if shape == () and jnp.issubdtype(minimum.dtype, jnp.integer) and jnp.issubdtype(maximum.dtype, jnp.integer):
            return spaces.Discrete(int(maximum) + 1)
        return spaces.Box(
            low=jnp.broadcast_to(minimum, shape).astype(jnp.float32),
            high=jnp.broadcast_to(maximum, shape).astype(jnp.float32),
            shape=shape,
            dtype=jnp.float32,
        )
    raise TypeError(f"Unsupported action spec type for EggRoll adapter: {type(spec).__name__}")


def _action_spec(env: Any):
    spec = getattr(env, "action_spec")
    return spec() if callable(spec) else spec


def _call_space(fn_or_space: Any, params: Any):
    if callable(fn_or_space):
        try:
            return fn_or_space(params)
        except TypeError:
            return fn_or_space()
    return fn_or_space


def _default_env_params(env: Any):
    for name in ("default_params", "default_env_params", "env_params"):
        if hasattr(env, name):
            value = getattr(env, name)
            return value() if callable(value) else value
    return None


def _make_gymnax_like_spaces(env: Any, params: Any, *, jax, jnp):
    if hasattr(env, "observation_space") and hasattr(env, "action_space"):
        return _call_space(env.observation_space, params), _call_space(env.action_space, params)
    from gymnax.environments import spaces

    key = jax.random.key(0)
    try:
        obs, _state = env.reset(key, params)
    except TypeError:
        obs, _state = env.reset(key)
    observation_space = _space_from_sample(obs, spaces, jax, jnp)
    if hasattr(env, "action_spec"):
        action_space = _spec_to_space(_action_spec(env), spaces, jnp)
    else:
        action_space = spaces.Discrete(2)
    return observation_space, action_space


def _make_gymnax_env(env_id: str):
    import gymnax

    return gymnax.make(env_id)


@dataclass(frozen=True)
class EggRollEnvSpaces:
    observation_space: Any
    action_space: Any


class GymnaxEggRollAdapter:
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
        if not (hasattr(self.action_space, "low") and hasattr(self.action_space, "high")):
            return action
        low, high = _space_bounds(self.action_space, self._jnp)
        return self._jnp.clip(action, low, high)


class GymnaxLikeEggRollAdapter:
    def __init__(self, env_name: str, *, jax, jnp, env: Any, env_params: Any | None = None) -> None:
        self._jnp = jnp
        self.env = env
        self.env_params = _default_env_params(env) if env_params is None else env_params
        self.observation_space, self.action_space = _make_gymnax_like_spaces(env, self.env_params, jax=jax, jnp=jnp)

    def reset(self, key):
        if self.env_params is None:
            return self.env.reset(key)
        return self.env.reset(key, self.env_params)

    def step(self, key, state, action):
        if self.env_params is None:
            return self.env.step(key, state, action)
        return self.env.step(key, state, action, self.env_params)

    def clip_action(self, action):
        if not (hasattr(self.action_space, "low") and hasattr(self.action_space, "high")):
            return action
        low, high = _space_bounds(self.action_space, self._jnp)
        return self._jnp.clip(action, low, high)


class BraxEggRollAdapter:
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        from brax import envs
        from gymnax.environments import spaces

        self._jnp = jnp
        brax_name = env_name.split(":", 1)[1]
        self.env = envs.get_environment(brax_name)
        obs_shape = (int(self.env.observation_size),)
        act_shape = (int(self.env.action_size),)
        self.observation_space = _gymnax_box_from_shape(spaces, jnp, obs_shape)
        self.action_space = _gymnax_box_from_shape(spaces, jnp, act_shape, low=-1.0, high=1.0)

    def reset(self, key):
        state = self.env.reset(key)
        return self._jnp.asarray(state.obs, dtype=self._jnp.float32), state

    def step(self, _key, state, action):
        action = self.clip_action(action)
        next_state = self.env.step(state, action)
        return self._jnp.asarray(next_state.obs, dtype=self._jnp.float32), next_state, next_state.reward, next_state.done, {}

    def clip_action(self, action):
        low, high = _space_bounds(self.action_space, self._jnp)
        return self._jnp.clip(action, low, high)


class CraftaxEggRollAdapter(GymnaxLikeEggRollAdapter):
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        raw_name = env_name.split(":", 1)[1]
        try:
            env, params = _make_gymnax_env(raw_name)
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
                return made, _default_env_params(made)

        if "Classic" in raw_name:
            from craftax.craftax.envs.craftax_classic_symbolic_env import CraftaxClassicSymbolicEnv

            env = CraftaxClassicSymbolicEnv()
        else:
            from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

            env = CraftaxSymbolicEnv()
        return env, _default_env_params(env)


class JumanjiEggRollAdapter:
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        import jumanji
        from gymnax.environments import spaces

        self._jax = jax
        self._jnp = jnp
        self.env = jumanji.make(env_name.split(":", 1)[1])
        key = jax.random.key(0)
        state, timestep = self.env.reset(key)
        obs = _flat_obs(timestep.observation, jax, jnp)
        self.observation_space = _gymnax_box_from_shape(spaces, jnp, tuple(int(v) for v in obs.shape))
        self.action_space = _spec_to_space(_action_spec(self.env), spaces, jnp)

    def reset(self, key):
        state, timestep = self.env.reset(key)
        return _flat_obs(timestep.observation, self._jax, self._jnp), state

    def step(self, _key, state, action):
        next_state, timestep = self.env.step(state, action)
        done = timestep.last()
        return _flat_obs(timestep.observation, self._jax, self._jnp), next_state, timestep.reward, done, {}

    def clip_action(self, action):
        if not (hasattr(self.action_space, "low") and hasattr(self.action_space, "high")):
            return action
        low, high = _space_bounds(self.action_space, self._jnp)
        return self._jnp.clip(action, low, high)


class KinetixEggRollAdapter(GymnaxLikeEggRollAdapter):
    def __init__(self, env_name: str, *, jax, jnp) -> None:
        raw_name = env_name.split(":", 1)[1]
        env, params = self._make_kinetix(raw_name)
        super().__init__(env_name, jax=jax, jnp=jnp, env=env, env_params=params)

    @staticmethod
    def _make_kinetix(raw_name: str):
        import kinetix

        for factory_name in ("make", "make_kinetix_env_from_name", "make_env_from_name"):
            factory = getattr(kinetix, factory_name, None)
            if callable(factory):
                made = factory(raw_name)
                if isinstance(made, tuple) and len(made) == 2:
                    return made
                return made, _default_env_params(made)
        try:
            from kinetix.environment.env import make_kinetix_env_from_name
        except ImportError:
            from kinetix.environment import make_kinetix_env_from_name
        made = make_kinetix_env_from_name(raw_name)
        if isinstance(made, tuple) and len(made) == 2:
            return made
        return made, _default_env_params(made)


class NavixEggRollAdapter(GymnaxLikeEggRollAdapter):
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
                return made, _default_env_params(made)
        raise ImportError("Navix is installed, but no supported environment factory was found.")


class JaxMARLEggRollAdapter:
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
        candidates = [raw_name, self._NAME_ALIASES.get(raw_name, raw_name), raw_name.replace("-", "_")]
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
        self.observation_space = _space_from_sample(self._ordered_obs(obs), spaces, jax, jnp)
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
        return _flat_obs(self._ordered_obs(obs), self._jax, self._jnp), state

    def step(self, key, state, action):
        actions = action if isinstance(action, dict) else self.clip_action(action)
        obs, next_state, rewards, dones, infos = self.env.step(key, state, actions)
        reward_values = [self._jnp.asarray(rewards[agent], dtype=self._jnp.float32) for agent in self.agents]
        reward = self._jnp.mean(self._jnp.stack(reward_values))
        done = dones.get("__all__", None) if hasattr(dones, "get") else None
        if done is None:
            done_values = [self._jnp.asarray(dones[agent]) for agent in self.agents]
            done = self._jnp.all(self._jnp.stack(done_values))
        return _flat_obs(self._ordered_obs(obs), self._jax, self._jnp), next_state, reward, done, infos

    def clip_action(self, action):
        if isinstance(action, dict):
            return {
                agent: self._jnp.clip(self._jnp.rint(action[agent]).astype(self._jnp.int32), 0, int(size) - 1)
                for agent, size in zip(self.agents, self._action_sizes, strict=True)
            }
        action = self._jnp.ravel(self._jnp.asarray(action))
        action = self._jnp.rint(action).astype(self._jnp.int32)
        out = {}
        for idx, (agent, size) in enumerate(zip(self.agents, self._action_sizes, strict=True)):
            out[agent] = self._jnp.clip(action[idx], 0, int(size) - 1)
        return out


class SurrogateObjectiveEggRollAdapter:
    _SPECS = {
        "synthetic:linear-speed": (256, 256, 1.0),
        "passk:": (64, 64, 4.0),
        "qwen:": (64, 64, 3.0),
        "jaxlob:": (128, 32, 2.0),
        "rwkv-int8-distill:": (64, 64, 1.5),
    }

    def __init__(self, env_name: str, *, jax, jnp) -> None:
        from gymnax.environments import spaces

        self._jnp = jnp
        self.env_name = str(env_name)
        obs_dim, action_dim, reward_scale = self._resolve_spec(self.env_name)
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        self._reward_scale = float(reward_scale)
        self._tag_scale = _stable_scale(self.env_name)
        self.observation_space = _gymnax_box_from_shape(spaces, jnp, (self._obs_dim,), low=-1.0, high=1.0)
        self.action_space = _gymnax_box_from_shape(spaces, jnp, (self._action_dim,), low=-1.0, high=1.0)

    @classmethod
    def _resolve_spec(cls, env_name: str) -> tuple[int, int, float]:
        for prefix, spec in cls._SPECS.items():
            if env_name.startswith(prefix):
                return spec
        raise ValueError(env_name)

    def _obs(self):
        idx = self._jnp.arange(self._obs_dim, dtype=self._jnp.float32)
        return self._jnp.sin((idx + 1.0) * self._tag_scale)

    def _target(self):
        idx = self._jnp.arange(self._action_dim, dtype=self._jnp.float32)
        return self._jnp.tanh(self._jnp.cos((idx + 1.0) * (self._tag_scale + 0.25)))

    def reset(self, _key):
        state = {"t": self._jnp.array(0, dtype=self._jnp.int32)}
        return self._obs(), state

    def step(self, _key, state, action):
        action = self.clip_action(action)
        target = self._target()
        mse = self._jnp.mean((action - target) ** 2)
        alignment = self._jnp.mean(action * target)
        reward = self._reward_scale * (alignment - mse)
        next_state = {"t": state["t"] + self._jnp.array(1, dtype=self._jnp.int32)}
        done = self._jnp.array(True)
        return self._obs(), next_state, reward.astype(self._jnp.float32), done, {}

    def clip_action(self, action):
        action = self._jnp.ravel(self._jnp.asarray(action, dtype=self._jnp.float32))
        if action.shape[0] != self._action_dim:
            action = self._jnp.resize(action, (self._action_dim,))
        return self._jnp.clip(action, -1.0, 1.0)


def make_eggroll_env_adapter(env_name: str, *, jax, jnp):
    env_name = str(env_name)
    if env_name.startswith("gymnax:"):
        return GymnaxEggRollAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("brax:"):
        return BraxEggRollAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("craftax:"):
        return CraftaxEggRollAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("jaxmarl:"):
        return JaxMARLEggRollAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("jumanji:"):
        return JumanjiEggRollAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("kinetix:"):
        return KinetixEggRollAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("navix:"):
        return NavixEggRollAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith(EGGROLL_SURROGATE_ENV_PREFIXES):
        return SurrogateObjectiveEggRollAdapter(env_name, jax=jax, jnp=jnp)
    raise ValueError(f"Unsupported EggRoll env tag: {env_name}")


def resolve_eggroll_env_spaces(env_name: str) -> EggRollEnvSpaces:
    import jax
    import jax.numpy as jnp

    adapter = make_eggroll_env_adapter(str(env_name), jax=jax, jnp=jnp)
    return EggRollEnvSpaces(adapter.observation_space, adapter.action_space)
