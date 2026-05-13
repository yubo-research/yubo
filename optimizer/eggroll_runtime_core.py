from __future__ import annotations

from typing import Any

import numpy as np


_DEFAULT_STACK_ERROR = (
    "EggRoll JAX vector evaluation requires the separate HyperscaleES environment. "
    "Run admin/setup-hyperscalees.sh first, then use the plain python CLI from that environment."
)


def require_eggroll_jax_stack(message: str | None = None):
    try:
        import jax
        import jax.numpy as jnp
        from hyperscalees.models.common import simple_es_tree_key
    except ImportError as exc:
        raise ImportError(_DEFAULT_STACK_ERROR if message is None else message) from exc
    return jax, jnp, simple_es_tree_key


def as_bool(
    value: Any,
    *,
    name: str,
    error_cls=ValueError,
    option_label: str = "EggRoll JAX option",
) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "t", "1", "yes"}:
            return True
        if lower in {"false", "f", "0", "no"}:
            return False
    raise error_cls(f"{option_label} '{name}' must be a bool.")


class IdentityNoiser:
    @staticmethod
    def get_noisy_standard(_frozen_noiser_params, _noiser_params, params, _es_tree_key, _iterinfo):
        return params

    @staticmethod
    def do_mm(_frozen_noiser_params, _noiser_params, params, _es_tree_key, _iterinfo, x):
        return x @ params.T

    @staticmethod
    def do_Tmm(_frozen_noiser_params, _noiser_params, params, _es_tree_key, _iterinfo, x):
        return x @ params

    @staticmethod
    def do_emb(_frozen_noiser_params, _noiser_params, params, _es_tree_key, _iterinfo, x):
        return params[x]


class EggRollParamCodec:
    def __init__(self, jax, jnp, params_init) -> None:
        self._jax = jax
        self._jnp = jnp
        leaves, treedef = jax.tree_util.tree_flatten(params_init)
        self._leaves_init = tuple(leaves)
        self._treedef = treedef
        self.shapes = tuple(tuple(int(v) for v in leaf.shape) for leaf in leaves)
        self.dtypes = tuple(leaf.dtype for leaf in leaves)
        self.sizes = tuple(int(leaf.size) for leaf in leaves)
        self.offsets = tuple(np.cumsum((0,) + self.sizes[:-1]).astype(np.int64).tolist())
        self.dim = int(sum(self.sizes))
        self.x0 = self.flatten(params_init)

    def flatten(self, params) -> np.ndarray:
        leaves, _treedef = self._jax.tree_util.tree_flatten(params)
        flat = [np.asarray(leaf).reshape(-1).astype(np.float64) for leaf in leaves]
        if not flat:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(flat, axis=0)

    def decode_absolute(self, x):
        leaves = []
        start = 0
        for shape, dtype, size in zip(self.shapes, self.dtypes, self.sizes, strict=True):
            leaf = self._jnp.reshape(x[start : start + size], shape).astype(dtype)
            leaves.append(leaf)
            start += size
        return self._jax.tree_util.tree_unflatten(self._treedef, leaves)

    def decode_offset(self, x, *, scale: float):
        leaves = []
        start = 0
        for init_leaf, shape, dtype, size in zip(
            self._leaves_init,
            self.shapes,
            self.dtypes,
            self.sizes,
            strict=True,
        ):
            raw = self._jnp.reshape(x[start : start + size], shape).astype(dtype)
            leaves.append(init_leaf + raw * float(scale))
            start += size
        return self._jax.tree_util.tree_unflatten(self._treedef, leaves)


class EggRollActionSelector:
    def __init__(self, jax, jnp, *, deterministic_policy: bool) -> None:
        self._jax = jax
        self._jnp = jnp
        self._deterministic_policy = bool(deterministic_policy)

    def select_action(self, policy_dist, action_key):
        if isinstance(policy_dist, tuple):
            policy_dist = policy_dist[0]
        if not self._deterministic_policy:
            return policy_dist.sample(seed=action_key)
        mode = getattr(policy_dist, "mode", None)
        if callable(mode):
            return mode()
        mean = getattr(policy_dist, "mean", None)
        if callable(mean):
            return mean()
        if mean is not None:
            return mean
        return policy_dist.sample(seed=action_key)

    def distribution_features(self, policy_dist):
        if isinstance(policy_dist, tuple):
            policy_dist = policy_dist[0]
        features = self._distribution_features(policy_dist)
        if features is not None:
            return features
        sample = policy_dist.sample(seed=self._jax.random.key(0))
        return self._jnp.ravel(self._jnp.asarray(sample, dtype=self._jnp.float32))

    def _distribution_features(self, policy_dist):
        for name in ("logits", "mean", "probs", "mode"):
            value = getattr(policy_dist, name, None)
            if callable(value):
                value = value()
            if value is not None:
                return self._jnp.ravel(self._jnp.asarray(value, dtype=self._jnp.float32))
        return None
