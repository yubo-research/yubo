from __future__ import annotations

from typing import Any

from problems.jax_env import (
    CraftaxAdapter,
    GymnasiumMJXAdapter,
    GymnaxAdapter,
    JaxMARLAdapter,
    JumanjiAdapter,
    KinetixAdapter,
    MujocoPlaygroundAdapter,
    NavixAdapter,
)
from problems.jax_env_core import SURROGATE_OBJECTIVE_PREFIXES, JaxEnvSpaces
from problems.mjx_env import is_gymnasium_env_tag
from problems.surrogate_objective_env import SurrogateObjectiveAdapter

_DIRECT_ADAPTERS = (
    ("gymnax:", GymnaxAdapter),
    ("craftax:", CraftaxAdapter),
    ("jaxmarl:", JaxMARLAdapter),
    ("jumanji:", JumanjiAdapter),
    ("kinetix:", KinetixAdapter),
    ("navix:", NavixAdapter),
    ("mujoco_playground:", MujocoPlaygroundAdapter),
)


def make_jax_env_adapter(env_name: str, *, jax, jnp, env_runtime: Any | None = None):
    env_name = str(env_name)
    from problems.isaaclab_env_adapters import is_isaaclab_env_tag

    if is_isaaclab_env_tag(env_name):
        from problems.isaaclab_jax_env import make_isaaclab_jax_adapter

        runtime = env_runtime
        kwargs = getattr(runtime, "kwargs", {}) if runtime is not None else {}
        max_steps = getattr(runtime, "max_steps", None) if runtime is not None else None
        return make_isaaclab_jax_adapter(
            env_name,
            jax=jax,
            jnp=jnp,
            env_runtime=runtime,
            problem_seed=int(getattr(runtime, "problem_seed", 0) or 0),
            device=getattr(runtime, "runtime_device", None) or kwargs.get("device"),
            max_steps=int(max_steps or 1000),
            headless=bool(kwargs.get("headless", True)),
            launcher_kwargs=kwargs.get("launcher_kwargs"),
        )
    for prefix, adapter_cls in _DIRECT_ADAPTERS:
        if env_name.startswith(prefix):
            return adapter_cls(env_name, jax=jax, jnp=jnp)
    if is_gymnasium_env_tag(env_name):
        return GymnasiumMJXAdapter(env_name, jax=jax, jnp=jnp)
    if env_name.startswith(SURROGATE_OBJECTIVE_PREFIXES):
        return SurrogateObjectiveAdapter(env_name, jax=jax, jnp=jnp)
    raise ValueError(f"Unsupported JAX env tag: {env_name}")


def resolve_jax_env_spaces(env_name: str) -> JaxEnvSpaces:
    import jax
    import jax.numpy as jnp

    adapter = make_jax_env_adapter(str(env_name), jax=jax, jnp=jnp)
    return JaxEnvSpaces(adapter.observation_space, adapter.action_space)
