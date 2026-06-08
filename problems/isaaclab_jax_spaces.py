from __future__ import annotations

from typing import Any

import numpy as np

from problems import jax_env_core as core


def gymnax_spaces_from_host(host: Any, *, jax, jnp) -> tuple[Any, Any]:
    from gymnax.environments import spaces

    obs_shape = (int(np.prod(host.observation_space.shape)),)
    act_shape = tuple(int(v) for v in host.action_space.shape)
    low = np.ravel(np.asarray(host.action_space.low, dtype=np.float32))
    high = np.ravel(np.asarray(host.action_space.high, dtype=np.float32))
    if low.size != act_shape[0] if act_shape else low.size:
        low = np.full(act_shape, -1.0, dtype=np.float32)
        high = np.full(act_shape, 1.0, dtype=np.float32)
    return (
        core._gymnax_box_from_shape(spaces, jnp, obs_shape),
        spaces.Box(
            low=jnp.asarray(low, dtype=jnp.float32),
            high=jnp.asarray(high, dtype=jnp.float32),
            shape=act_shape,
            dtype=jnp.float32,
        ),
    )
