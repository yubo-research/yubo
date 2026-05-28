from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class MJXRuntime(NamedTuple):
    jax: Any
    jnp: Any
    optax: Any
    adapter: Any
    obs_dim: int
    act_dim: int
    low: Any
    high: Any


def require_mjx_stack():
    import jax
    import jax.numpy as jnp
    import optax

    return jax, jnp, optax


def make_mjx_runtime(config) -> MJXRuntime:
    jax, jnp, optax = require_mjx_stack()
    from problems.jax_env_core import _space_bounds
    from problems.jax_env_factory import make_jax_env_adapter

    adapter = make_jax_env_adapter(config.env_tag, jax=jax, jnp=jnp)
    obs_dim = int(np.prod(tuple(adapter.observation_space.shape)))
    act_dim = int(np.prod(tuple(adapter.action_space.shape)))
    low, high = _space_bounds(adapter.action_space, jnp)
    return MJXRuntime(jax, jnp, optax, adapter, obs_dim, act_dim, low, high)
