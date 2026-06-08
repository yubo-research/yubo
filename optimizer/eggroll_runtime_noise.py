from __future__ import annotations

import importlib
import math

import numpy as np

from optimizer.eggroll_runtime_core import _DEFAULT_STACK_ERROR


def sample_leaf_mask(rng: np.random.Generator, *, num_leaves: int, target: float) -> np.ndarray:
    if 0 < target < 1:
        mask = rng.random(int(num_leaves)) < target
        if not np.any(mask):
            mask[int(rng.integers(int(num_leaves)))] = True
        return mask
    k = min(max(int(target), 1), int(num_leaves))
    mask = np.zeros(int(num_leaves), dtype=bool)
    mask[rng.choice(int(num_leaves), size=k, replace=False)] = True
    return mask


class EggRollNoiseSampler:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._codec = runtime.codec

    def sample(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ):
        key = self._runtime.jax.random.key(int(seed) & 0xFFFFFFFF)
        if num_module_target is not None:
            return self._sample_leaf_noise(key, float(num_module_target))
        if num_dim_target is not None:
            return self._sample_dim_noise(key, float(num_dim_target))
        return self._runtime.jax.random.normal(key, (self._codec.dim,), dtype=self._runtime.jnp.float32)

    def _sample_dim_noise(self, key, target: float):
        if target <= 0:
            raise ValueError("dim perturb target must be > 0.")
        if 0 < target < 1:
            return self._sample_dim_fraction(key, target)
        k = min(max(int(target), 1), self._codec.dim)
        idx_key, value_key = self._runtime.jax.random.split(key)
        idx = self._runtime.jax.random.choice(idx_key, self._codec.dim, shape=(k,), replace=False)
        values = self._runtime.jax.random.normal(value_key, (k,), dtype=self._runtime.jnp.float32)
        return self._runtime.jnp.zeros((self._codec.dim,), dtype=self._runtime.jnp.float32).at[idx].set(values)

    def _sample_leaf_noise(self, key, target: float):
        if target <= 0:
            raise ValueError("module perturb target must be > 0.")
        mask_key, noise_key = self._runtime.jax.random.split(key)
        mask_seed = int(np.asarray(self._runtime.jax.random.randint(mask_key, (), 0, 2**31 - 1)))
        mask = self._leaf_mask(mask_seed, target)
        full_noise = self._runtime.jax.random.normal(noise_key, (self._codec.dim,), dtype=self._runtime.jnp.float32)
        flat_mask = np.zeros(self._codec.dim, dtype=bool)
        for selected, start, size in zip(mask, self._codec.offsets, self._codec.sizes, strict=True):
            if selected:
                flat_mask[int(start) : int(start) + int(size)] = True
        return self._runtime.jnp.where(self._runtime.jnp.asarray(flat_mask), full_noise, 0.0)

    def _sample_dim_fraction(self, key, target: float):
        mask_key, value_key, fallback_key = self._runtime.jax.random.split(key, 3)
        mask = self._runtime.jax.random.uniform(mask_key, (self._codec.dim,)) < float(target)
        fallback_idx = self._runtime.jax.random.randint(fallback_key, (), 0, self._codec.dim)
        fallback = self._runtime.jnp.arange(self._codec.dim) == fallback_idx
        mask = self._runtime.jnp.where(self._runtime.jnp.any(mask), mask, fallback)
        noise = self._runtime.jax.random.normal(value_key, (self._codec.dim,), dtype=self._runtime.jnp.float32)
        return self._runtime.jnp.where(mask, noise, 0.0)

    def _leaf_mask(self, seed: int, target: float) -> np.ndarray:
        return sample_leaf_mask(np.random.default_rng(int(seed)), num_leaves=len(self._codec.sizes), target=float(target))


class EggRollNoiserMaterializer:
    """Materialize upstream HyperscaleES noiser perturbations as flat UHD directions."""

    def __init__(
        self,
        runtime,
        *,
        noiser_name: str,
        rank: int,
        group_size: int,
        freeze_nonlora: bool,
    ) -> None:
        try:
            import optax
            from hyperscalees.noiser import all_noisers
        except ImportError as exc:
            raise ImportError(_DEFAULT_STACK_ERROR) from exc
        if str(noiser_name) not in all_noisers:
            raise ValueError(f"Unknown HyperscaleES noiser {noiser_name!r}. Available: {sorted(all_noisers)}")
        if int(rank) < 1:
            raise ValueError("eggroll_rank must be >= 1.")
        if int(group_size) < 0:
            raise ValueError("eggroll_group_size must be >= 0.")

        self._runtime = runtime
        self._noiser_name = str(noiser_name)
        self._noiser = all_noisers[self._noiser_name]
        self._noiser_module = importlib.import_module(self._noiser.__module__)
        self._frozen_noiser_params, self._noiser_params = self._noiser.init_noiser(
            runtime.policy.params,
            1.0,
            1.0,
            solver=optax.sgd,
            group_size=int(group_size),
            freeze_nonlora=bool(freeze_nonlora),
            rank=int(rank),
            use_batched_update=True,
        )

    def sample(self, x: np.ndarray, *, seed: int):
        rt = self._runtime
        if rt.vector_mode != "absolute":
            raise ValueError("EggRoll noiser perturbations require EggRollJAXRuntime vector_mode='absolute'.")
        x_arr = rt.to_vector(x)
        params = rt.decode_vector_params(x_arr)
        iterinfo = (
            rt.jnp.asarray(0, dtype=rt.jnp.int32),
            rt.jnp.asarray(2 * int(seed), dtype=rt.jnp.int32),
        )
        noised = rt.jax.tree.map(
            lambda p, k, m: self._materialize_leaf(p, k, m, iterinfo),
            params,
            rt.es_tree_key,
            rt.policy.es_map,
        )
        noised_x = rt.codec.flatten_device(noised)
        return noised_x - x_arr

    def _materialize_leaf(self, param, key, map_class, iterinfo):
        map_id = int(np.asarray(map_class))
        if map_id == 0:
            return self._noiser.get_noisy_standard(self._frozen_noiser_params, self._noiser_params, param, key, iterinfo)
        if map_id == 1:
            return self._materialize_structured_leaf(param, key, iterinfo)
        return param

    def _materialize_structured_leaf(self, param, key, iterinfo):
        if hasattr(self._noiser_module, "get_lora_update_params"):
            return self._materialize_lora_leaf(param, key, iterinfo)
        if hasattr(self._noiser_module, "get_sparse_update_params"):
            return self._materialize_sparse_leaf(param, key, iterinfo)
        raise ValueError(f"No materializer for map_class=1 with noiser {self._noiser_name!r}.")

    def _materialize_lora_leaf(self, param, key, iterinfo):
        rank = int(self._frozen_noiser_params.get("rank", 1))
        base_sigma = self._noiser_params["sigma"] / math.sqrt(float(rank))
        a, b = self._noiser_module.get_lora_update_params(self._frozen_noiser_params, base_sigma, iterinfo, param, key)
        return param + a @ b.T

    def _materialize_sparse_leaf(self, param, key, iterinfo):
        values, idx_a, idx_b = self._noiser_module.get_sparse_update_params(
            self._frozen_noiser_params,
            self._noiser_params["sigma"],
            iterinfo,
            param,
            key,
        )
        return param + self._runtime.jnp.zeros_like(param).at[idx_a, idx_b].add(values)
