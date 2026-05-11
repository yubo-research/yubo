from __future__ import annotations

import numpy as np


_MAX_INT8 = 127
_PARAM_STANDARD = 0
_PARAM_MM = 1
_PARAM_EMB = 2


class _NanoEggSubspaceCodec:
    def __init__(
        self,
        jax,
        jnp,
        params,
        es_map,
        *,
        dim: int,
        delta_scale: float,
        seed: int,
        lora_only: bool,
        basis_max_leaves: int | None,
    ) -> None:
        self._jax = jax
        self._jnp = jnp
        self._params = params
        self.dim = int(dim)
        self.delta_scale = float(delta_scale)
        leaves, treedef = jax.tree_util.tree_flatten(params)
        map_leaves, map_treedef = jax.tree_util.tree_flatten(es_map)
        if map_treedef != treedef:
            raise ValueError("NanoEgg es_map tree does not match params tree.")
        self._leaves = tuple(leaves)
        self._treedef = treedef
        self._leaf_shapes = tuple(tuple(int(v) for v in leaf.shape) for leaf in leaves)
        sizes = np.asarray([int(leaf.size) for leaf in leaves], dtype=np.int64)
        self._leaf_kind = np.asarray([int(np.max(np.asarray(leaf))) for leaf in map_leaves], dtype=np.int64)
        eligible = (
            np.asarray([bool(np.any(np.isin(np.asarray(leaf), (_PARAM_MM, _PARAM_EMB)))) for leaf in map_leaves], dtype=bool)
            if lora_only
            else np.ones(len(leaves), dtype=bool)
        )
        valid = np.flatnonzero((sizes > 0) & eligible)
        if valid.size == 0:
            raise ValueError("NanoEgg params tree has no eligible leaves for UHD subspace.")
        if basis_max_leaves is not None and int(basis_max_leaves) < valid.size:
            rng_for_leaves = np.random.default_rng(int(seed) ^ 0xA5A5A5A5)
            probs = sizes[valid].astype(np.float64)
            probs = probs / probs.sum()
            valid = np.sort(rng_for_leaves.choice(valid, size=int(basis_max_leaves), replace=False, p=probs).astype(np.int64))
        probs = sizes[valid].astype(np.float64)
        probs = probs / probs.sum()
        rng = np.random.default_rng(int(seed))
        self._basis_leaf = rng.choice(valid, size=self.dim, replace=True, p=probs).astype(np.int64)
        self._basis_index = np.asarray([rng.integers(sizes[leaf]) for leaf in self._basis_leaf], dtype=np.int64)
        self._basis_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=self.dim).astype(np.float32)
        self._base_noise_seed = int(seed)
        self.x0 = np.zeros((self.dim,), dtype=np.float64)

    def decode(self, x: np.ndarray):
        coeffs = np.asarray(x, dtype=np.float32).reshape(-1)
        if coeffs.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},), got {coeffs.shape}.")
        active = np.flatnonzero(coeffs != 0.0)
        if active.size == 0:
            return self._params
        leaves = list(self._leaves)
        active_basis_leaf = self._basis_leaf[active]
        for leaf_idx in np.unique(active_basis_leaf):
            positions = active[np.flatnonzero(active_basis_leaf == leaf_idx)]
            leaf = leaves[int(leaf_idx)]
            flat = self._jnp.reshape(leaf, (-1,))
            idx = self._jnp.asarray(self._basis_index[positions], dtype=self._jnp.int32)
            values = self._jnp.rint(self._jnp.asarray(coeffs[positions] * self._basis_sign[positions] * self.delta_scale, dtype=self._jnp.float32)).astype(
                self._jnp.int32
            )
            updated = self._jnp.clip(flat.astype(self._jnp.int32).at[idx].add(values), -_MAX_INT8, _MAX_INT8).astype(leaf.dtype)
            leaves[int(leaf_idx)] = updated.reshape(leaf.shape)
        return self._jax.tree_util.tree_unflatten(self._treedef, leaves)

    def sample_eggroll_direction(
        self,
        *,
        seed: int,
        rank: int,
        group_size: int,
        freeze_nonlora: bool,
    ) -> np.ndarray:
        _ = group_size
        rank = max(int(rank), 1)
        direction = np.zeros(self.dim, dtype=np.float64)
        for leaf_idx in np.unique(self._basis_leaf):
            positions = np.flatnonzero(self._basis_leaf == int(leaf_idx))
            kind = int(self._leaf_kind[int(leaf_idx)])
            if kind == _PARAM_EMB:
                continue
            if freeze_nonlora and kind == _PARAM_STANDARD:
                continue
            shape = self._leaf_shapes[int(leaf_idx)]
            leaf_seed = _mix_seed(self._base_noise_seed, int(seed), int(leaf_idx), kind)
            if kind == _PARAM_MM and len(shape) >= 2:
                direction[positions] = _eggroll_low_rank_values(
                    self._jax,
                    self._jnp,
                    seed=leaf_seed,
                    flat_indices=self._basis_index[positions],
                    shape=shape,
                    rank=rank,
                )
            else:
                direction[positions] = _jax_flat_normal_values(
                    self._jax,
                    self._jnp,
                    seed=leaf_seed,
                    flat_indices=self._basis_index[positions],
                    shape=shape,
                )
        return direction.astype(np.float64)


def _mix_seed(*values: int) -> int:
    state = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        state ^= int(value) & mask
        state = (state * 0xBF58476D1CE4E5B9) & mask
        state ^= state >> 30
    return int(state & 0xFFFFFFFF)


def _eggroll_low_rank_values(jax, jnp, *, seed: int, flat_indices: np.ndarray, shape: tuple[int, ...], rank: int) -> np.ndarray:
    cols = max(int(shape[-1]), 1)
    rows_count = max(int(np.prod(shape[:-1], dtype=np.int64)), 1)
    rows = np.asarray(flat_indices, dtype=np.int64) // max(cols, 1)
    col_ids = np.asarray(flat_indices, dtype=np.int64) % max(cols, 1)
    key = jax.random.key(int(seed) & 0xFFFFFFFF)
    lora_params = jax.random.normal(key, (rows_count + cols, int(rank)), dtype=jnp.float32)
    b = lora_params[:cols]
    a = lora_params[cols:]
    values = jnp.sum(a[jnp.asarray(rows, dtype=jnp.int32)] * b[jnp.asarray(col_ids, dtype=jnp.int32)], axis=1)
    return np.asarray(values / np.sqrt(float(rank)), dtype=np.float64)


def _jax_flat_normal_values(jax, jnp, *, seed: int, flat_indices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    key = jax.random.key(int(seed) & 0xFFFFFFFF)
    values = jax.random.normal(key, shape, dtype=jnp.float32).reshape((-1,))
    return np.asarray(values[jnp.asarray(flat_indices, dtype=jnp.int32)], dtype=np.float64)


__all__ = ["_NanoEggSubspaceCodec"]
