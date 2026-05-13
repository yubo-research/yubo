from __future__ import annotations

import importlib
import math

import numpy as np

from optimizer.eggroll_runtime_core import _DEFAULT_STACK_ERROR


class EggRollNoiseSampler:
    def __init__(self, codec) -> None:
        self._codec = codec

    def sample(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        if num_module_target is not None:
            return self._sample_module_noise(rng, float(num_module_target))
        if num_dim_target is not None:
            return self._sample_dim_noise(rng, float(num_dim_target))
        return rng.standard_normal(self._codec.dim).astype(np.float64)

    def _sample_dim_noise(self, rng: np.random.Generator, target: float) -> np.ndarray:
        if target <= 0:
            raise ValueError("dim perturb target must be > 0.")
        if 0 < target < 1:
            return self._sample_dim_fraction(rng, target)
        k = min(max(int(target), 1), self._codec.dim)
        idx = rng.choice(self._codec.dim, size=k, replace=False)
        noise = np.zeros(self._codec.dim, dtype=np.float64)
        noise[idx] = rng.standard_normal(k)
        return noise

    def _sample_module_noise(self, rng: np.random.Generator, target: float) -> np.ndarray:
        if target <= 0:
            raise ValueError("module perturb target must be > 0.")
        mask = self._module_mask(rng, target)
        noise = np.zeros(self._codec.dim, dtype=np.float64)
        for selected, start, size in zip(mask, self._codec.offsets, self._codec.sizes, strict=True):
            if not selected:
                continue
            end = int(start) + int(size)
            noise[int(start) : end] = rng.standard_normal(int(size))
        return noise

    def _sample_dim_fraction(self, rng: np.random.Generator, target: float) -> np.ndarray:
        mask = rng.random(self._codec.dim) < target
        if not np.any(mask):
            mask[int(rng.integers(self._codec.dim))] = True
        noise = rng.standard_normal(self._codec.dim).astype(np.float64)
        noise[~mask] = 0.0
        return noise

    def _module_mask(self, rng: np.random.Generator, target: float) -> np.ndarray:
        num_leaves = len(self._codec.sizes)
        if 0 < target < 1:
            mask = rng.random(num_leaves) < target
            if not np.any(mask):
                mask[int(rng.integers(num_leaves))] = True
            return mask
        k = min(max(int(target), 1), num_leaves)
        mask = np.zeros(num_leaves, dtype=bool)
        mask[rng.choice(num_leaves, size=k, replace=False)] = True
        return mask


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

    def sample(self, x: np.ndarray, *, seed: int) -> np.ndarray:
        rt = self._runtime
        if rt.vector_mode != "absolute":
            raise ValueError("EggRoll noiser perturbations require EggRollJAXRuntime vector_mode='absolute'.")
        x_arr = np.asarray(x, dtype=np.float64)
        params = rt.decode_vector_params(rt.jnp.asarray(x_arr, dtype=rt.jnp.float32))
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
        noised_x = rt.codec.flatten(noised)
        return np.asarray(noised_x - x_arr, dtype=np.float64)

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
