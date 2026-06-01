from __future__ import annotations

from typing import Any

import numpy as np


class EggRollRuntimeVectorOps:
    """Flat-vector conversions shared by EggRollJAXRuntime."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def dim(self) -> int:
        return self._runtime.dim

    def to_vector(self, x):
        return self._runtime.jnp.asarray(x, dtype=self._runtime.jnp.float32)

    def to_vector_batch(self, x_batch):
        x = self._runtime.jnp.asarray(x_batch, dtype=self._runtime.jnp.float32)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x_batch must have shape (n, {self.dim}), got {x.shape}.")
        return x

    def copy_vector(self, x):
        return self._runtime.jnp.array(x, dtype=self._runtime.jnp.float32, copy=True)

    def stack_vectors(self, xs):
        return self._runtime.jnp.stack([self.to_vector(x) for x in xs], axis=0)

    def zeros_vector(self, dim: int):
        return self._runtime.jnp.zeros((int(dim),), dtype=self._runtime.jnp.float32)

    def vector_to_numpy(self, x) -> np.ndarray:
        return np.asarray(self._runtime.jax.device_get(x), dtype=np.float64)

    def decode_vector_params(self, x):
        x = self.to_vector(x)
        if self._runtime._vector_mode == "offset":
            return self._runtime.codec.decode_offset(x, scale=self._runtime._param_scale)
        return self._runtime.codec.decode_absolute(x)


__all__ = ["EggRollRuntimeVectorOps"]
