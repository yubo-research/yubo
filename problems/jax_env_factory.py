from __future__ import annotations

from problems.jax_env import (
    BraxAdapter,
    CraftaxAdapter,
    GymnaxAdapter,
    JaxMARLAdapter,
    JumanjiAdapter,
    KinetixAdapter,
    NavixAdapter,
)
from problems.jax_env_core import SURROGATE_OBJECTIVE_PREFIXES, JaxEnvSpaces
from problems.surrogate_objective_env import SurrogateObjectiveAdapter


def make_jax_env_adapter(env_name: str, *, jax, jnp):
    env_name = str(env_name)
    if env_name.startswith("gymnax:"):
        return GymnaxAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("brax:"):
        return BraxAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("craftax:"):
        return CraftaxAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("jaxmarl:"):
        return JaxMARLAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("jumanji:"):
        return JumanjiAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("kinetix:"):
        return KinetixAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith("navix:"):
        return NavixAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith(SURROGATE_OBJECTIVE_PREFIXES):
        return SurrogateObjectiveAdapter(env_name, jax=jax, jnp=jnp)
    raise ValueError(f"Unsupported JAX env tag: {env_name}")


def resolve_jax_env_spaces(env_name: str) -> JaxEnvSpaces:
    import jax
    import jax.numpy as jnp

    adapter = make_jax_env_adapter(str(env_name), jax=jax, jnp=jnp)
    return JaxEnvSpaces(adapter.observation_space, adapter.action_space)
