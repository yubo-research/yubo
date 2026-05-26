from __future__ import annotations

from typing import Any, NamedTuple

import warp as wp


class UnifiedState(NamedTuple):
    """A GPU-native state that can be viewed by PyTorch or JAX via DLPack."""

    obs: wp.array
    reward: wp.array
    done: wp.array

    def torch(self) -> dict[str, Any]:
        """Return a dictionary of PyTorch tensor views."""
        return {
            "obs": wp.to_torch(self.obs),
            "reward": wp.to_torch(self.reward),
            "done": wp.to_torch(self.done),
        }

    def jax(self) -> dict[str, Any]:
        """Return a dictionary of JAX array views."""
        return {
            "obs": wp.to_jax(self.obs),
            "reward": wp.to_jax(self.reward),
            "done": wp.to_jax(self.done),
        }


class UnifiedMJXWarpAdapter:
    """A high-performance bridge that can switch between MJX and Warp backends."""

    def __init__(self, env_name: str, backend: str = "warp", num_envs: int = 1) -> None:
        self.env_name = env_name
        self.backend_name = backend
        self.num_envs = int(num_envs)

        if backend == "warp":
            from problems.warp_env import GymnasiumWarpAdapter

            self.impl = GymnasiumWarpAdapter(env_name, num_envs=self.num_envs)
        elif backend == "mjx":
            # For MJX, we would need jax/jnp, usually passed at runtime in this repo
            # Keeping it KISS for now by focusing on the Warp path
            raise NotImplementedError("Unified MJX backend requires JAX context.")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.observation_space = self.impl.observation_space
        self.action_space = self.impl.action_space

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], Any]:
        obs, data = self.impl.reset(seed)
        # Return as a dict of Torch tensors by default for non-JAX runners
        return {"obs": wp.to_torch(obs)}, data

    def step(self, data: Any, action: Any) -> UnifiedState:
        res = self.impl.step(data, action)
        return UnifiedState(obs=res.obs, reward=res.reward, done=res.done)

    def close(self):
        pass


def save_unified_checkpoint(path: str, state: Any):
    """Save a unified checkpoint using Orbax."""
    import jax
    import orbax.checkpoint as ocp

    # Convert everything to JAX arrays via DLPack for Orbax
    def _to_jax(x):
        if isinstance(x, wp.array):
            return wp.to_jax(x)
        if hasattr(x, "__dlpack__"):  # Torch and others
            import jax.dlpack as jdl

            return jdl.from_dlpack(x)
        return x

    jax_state = jax.tree_util.tree_map(_to_jax, state)
    # Use PyTreeCheckpointHandler for more direct control
    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    mngr = ocp.CheckpointManager(path, ocp.PyTreeCheckpointer(), options=options)
    mngr.save(0, jax_state)
    mngr.wait_until_finished()
