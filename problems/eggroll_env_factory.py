from __future__ import annotations

from problems.eggroll_env_core import EGGROLL_SURROGATE_ENV_PREFIXES, EggRollEnvSpaces
from problems.eggroll_env_jax import (
    BraxEggRollAdapter,
    CraftaxEggRollAdapter,
    GymnaxEggRollAdapter,
    JaxMARLEggRollAdapter,
    JumanjiEggRollAdapter,
    KinetixEggRollAdapter,
    NavixEggRollAdapter,
)
from problems.eggroll_env_surrogate import SurrogateObjectiveEggRollAdapter


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
