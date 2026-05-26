from __future__ import annotations

from dataclasses import dataclass
from typing import Any

JAX_ENV_PREFIXES = (
    "gymnax:",
    "brax:",
    "gymnasium:",
    "craftax:",
    "jaxmarl:",
    "jumanji:",
    "kinetix:",
    "navix:",
)
SURROGATE_OBJECTIVE_PREFIXES = (
    "llm:deepscaler:passk4",
    "llm:deepscaler:rlvr",
    "rwkv:distill:",
    "hft:",
)
JAX_OBJECTIVE_PREFIXES = JAX_ENV_PREFIXES + SURROGATE_OBJECTIVE_PREFIXES

_SUPPORTED_JAX_ENV_EXAMPLES = (
    "gymnax:CartPole-v1",
    "gymnax:Swimmer-misc",
    "brax:ant",
    "brax:humanoid",
    "brax:inverted_double_pendulum",
    "gymnasium:HalfCheetah-v5",
    "craftax:Craftax-Classic-Symbolic-v1",
    "craftax:Craftax-Symbolic-AutoReset-v1",
    "jaxmarl:mpe-simple-reference-v3",
    "jaxmarl:mpe-simple-speaker-listener-v4",
    "jaxmarl:mpe-simple-spread-v3",
    "jumanji:Game2048-v1",
    "jumanji:Knapsack-v1",
    "jumanji:Snake-v1",
    "kinetix:s/h1_thrust_over_ball",
    "kinetix:s/hard_pinball",
    "kinetix:s/thrustcontrol_left",
    "navix:Navix-DoorKey-8x8-v0",
    "navix:Navix-Dynamic-Obstacles-6x6-Random-v0",
    "navix:Navix-FourRooms-v0",
)


@dataclass(frozen=True)
class JaxEnvSpaces:
    observation_space: Any
    action_space: Any


def supports_jax_env_tag(env_name: str) -> bool:
    return str(env_name).startswith(JAX_ENV_PREFIXES)


def supports_jax_objective_tag(env_name: str) -> bool:
    return str(env_name).startswith(JAX_OBJECTIVE_PREFIXES)


def supported_jax_env_tags() -> tuple[str, ...]:
    return tuple(sorted(_SUPPORTED_JAX_ENV_EXAMPLES))


def _stable_scale(text: str) -> float:
    total = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(text)))
    return float((total % 997) + 1) / 997.0


def _space_bounds(space: Any, jnp) -> tuple[Any, Any]:
    low = space.low if isinstance(space.low, jnp.ndarray) else space.low * jnp.ones(space.shape, dtype=space.dtype)
    high = space.high if isinstance(space.high, jnp.ndarray) else space.high * jnp.ones(space.shape, dtype=space.dtype)
    return jnp.asarray(low, dtype=jnp.float32), jnp.asarray(high, dtype=jnp.float32)


def _clip_box_action(action_space: Any, jnp, action: Any):
    if not (hasattr(action_space, "low") and hasattr(action_space, "high")):
        return action
    low, high = _space_bounds(action_space, jnp)
    return jnp.clip(action, low, high)


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
    raise TypeError(f"Unsupported action spec type for JAX env adapter: {type(spec).__name__}")


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
