from __future__ import annotations

import numpy as np


class EggRollRuntimeEmbedder:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._probe_obs = None
        self._embed_batch_jit = None
        self._num_probes = 0

    def configure(self, num_probes: int) -> None:
        if int(num_probes) < 1:
            raise ValueError("num_probes must be >= 1.")
        self._num_probes = int(num_probes)
        self._probe_obs = None
        self._embed_batch_jit = None

    def _ensure_built(self) -> None:
        if self._embed_batch_jit is not None:
            return
        if self._num_probes < 1:
            raise ValueError("Behavior embedding is not configured.")
        keys = self._runtime.jax.random.split(self._runtime.embed_key_base, self._num_probes)
        self._probe_obs = self._runtime.jax.vmap(lambda k: self._runtime.env_adapter.reset(k)[0])(keys)
        self._embed_batch_jit = self._build()

    def _build(self):
        rt = self._runtime
        jax = rt.jax
        jnp = rt.jnp
        model_cls = rt.policy.model_cls
        noiser = rt.identity_noiser
        frozen_params = rt.policy.frozen_params
        es_tree_key = rt.es_tree_key
        probe_obs = self._probe_obs

        def embed_candidate(x):
            params = rt.decode_vector_params(x)

            def features_for_obs(obs):
                policy_dist = model_cls.forward(noiser, None, None, frozen_params, params, es_tree_key, None, obs)
                return rt.action_selector.distribution_features(policy_dist)

            z = jax.vmap(features_for_obs)(probe_obs)
            return jnp.ravel(z)

        @jax.jit
        def embed_batch(x_batch):
            return jax.vmap(embed_candidate)(x_batch)

        return embed_batch

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        self._ensure_built()
        x_batch = np.asarray(x_batch, dtype=np.float64)
        assert self._embed_batch_jit is not None
        z = self._embed_batch_jit(self._runtime.jnp.asarray(x_batch, dtype=self._runtime.jnp.float32))
        return np.asarray(self._runtime.jax.block_until_ready(z), dtype=np.float64)

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embed_many(np.asarray([x], dtype=np.float64))[0]
