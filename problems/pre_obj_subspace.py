from __future__ import annotations

from typing import Any

import numpy as np


class _SubspaceParamCodec:
    """Map a small UHD vector into deterministic sparse deltas on a real param tree."""

    def __init__(
        self,
        jax,
        jnp,
        params,
        *,
        es_map=None,
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
        self._leaves = tuple(leaves)
        self._treedef = treedef
        sizes = np.asarray([int(leaf.size) for leaf in leaves], dtype=np.int64)
        eligible = np.ones(len(leaves), dtype=bool)
        if lora_only:
            map_leaves, map_treedef = jax.tree_util.tree_flatten(es_map)
            if map_treedef != treedef:
                raise ValueError("HyperscaleES es_map tree does not match params tree; cannot build LoRA-only UHD subspace.")
            eligible = np.asarray([_is_lora_leaf(map_leaf) for map_leaf in map_leaves], dtype=bool)

        valid = np.flatnonzero((sizes > 0) & eligible)
        if valid.size == 0:
            raise ValueError("HyperscaleES params tree has no eligible trainable leaves for the UHD pretraining subspace.")
        if basis_max_leaves is not None and int(basis_max_leaves) < valid.size:
            rng_for_leaves = np.random.default_rng(int(seed) ^ 0x5F3759DF)
            leaf_probs = sizes[valid].astype(np.float64)
            leaf_probs = leaf_probs / leaf_probs.sum()
            valid = np.sort(rng_for_leaves.choice(valid, size=int(basis_max_leaves), replace=False, p=leaf_probs).astype(np.int64))
        probs = sizes[valid].astype(np.float64)
        probs = probs / probs.sum()
        rng = np.random.default_rng(int(seed))
        self._basis_leaf = rng.choice(valid, size=self.dim, replace=True, p=probs).astype(np.int64)
        self._basis_index = np.asarray([rng.integers(sizes[leaf]) for leaf in self._basis_leaf], dtype=np.int64)
        self._basis_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=self.dim).astype(np.float32)
        self._basis_leaf_positions = tuple(
            (
                int(leaf_idx),
                self._jnp.asarray(np.flatnonzero(self._basis_leaf == leaf_idx), dtype=self._jnp.int32),
            )
            for leaf_idx in np.unique(self._basis_leaf)
        )
        self._basis_index_device = self._jnp.asarray(self._basis_index, dtype=self._jnp.int32)
        self._basis_sign_device = self._jnp.asarray(self._basis_sign, dtype=self._jnp.float32)
        self.num_total_leaves = int(len(leaves))
        self.num_candidate_leaves = int(valid.size)
        self.num_candidate_params = int(sizes[valid].sum())
        self.num_basis_leaves = int(np.unique(self._basis_leaf).size)
        self.x0 = np.zeros((self.dim,), dtype=np.float64)

    def decode(self, x: np.ndarray):
        coeffs = np.asarray(x, dtype=np.float32).reshape(-1)
        if coeffs.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},), got {coeffs.shape}.")
        active = np.flatnonzero(coeffs != 0.0)
        if active.size == 0:
            return self._params
        leaves = self._decode_active_leaves(coeffs, active)
        return self._jax.tree_util.tree_unflatten(self._treedef, leaves)

    def decode_device(self, x, params=None):
        coeffs = self._jnp.asarray(x, dtype=self._jnp.float32).reshape(-1)
        if coeffs.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},), got {coeffs.shape}.")
        leaves = list(self._leaves if params is None else self._jax.tree_util.tree_leaves(params))
        for leaf_idx, positions in self._basis_leaf_positions:
            leaf = leaves[leaf_idx]
            flat = self._jnp.reshape(leaf, (-1,))
            idx = self._basis_index_device[positions]
            values = coeffs[positions] * self._basis_sign_device[positions] * self.delta_scale
            leaves[leaf_idx] = flat.at[idx].add(values.astype(flat.dtype)).reshape(leaf.shape)
        return self._jax.tree_util.tree_unflatten(self._treedef, leaves)

    def _decode_active_leaves(self, coeffs: np.ndarray, active: np.ndarray):
        leaves = list(self._leaves)
        for leaf_idx in np.unique(self._basis_leaf[active]):
            positions = active[self._basis_leaf[active] == leaf_idx]
            leaf = leaves[int(leaf_idx)]
            flat = self._jnp.reshape(leaf, (-1,))
            idx = self._jnp.asarray(self._basis_index[positions], dtype=self._jnp.int32)
            values = self._jnp.asarray(
                coeffs[positions] * self._basis_sign[positions] * self.delta_scale,
                dtype=flat.dtype,
            )
            leaves[int(leaf_idx)] = flat.at[idx].add(values).reshape(leaf.shape)
        return leaves


def _is_lora_leaf(map_leaf: Any) -> bool:
    arr = np.asarray(map_leaf)
    return bool(np.any(arr == 1))
