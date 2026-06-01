from __future__ import annotations

import numpy as np

from optimizer.eggroll_runtime import EggRollJAXRuntime

_RUNTIME_ATTRS = frozenset(
    {
        "dim",
        "x0",
        "steps_per_episode",
        "num_envs",
        "copy_vector",
        "stack_vectors",
        "zeros_vector",
        "embed",
        "embed_many",
        "sample_noise",
        "sample_eggroll_noiser_noise",
    }
)


class EggRollJAXVectorObjective:
    """Functional flat-vector objective for EggRoll/HyperscaleES policies."""

    def __init__(
        self,
        policy,
        env_conf,
        *,
        steps_per_episode: int = 200,
        num_envs: int = 1,
        deterministic_policy: bool = False,
        seed_offset: int = 0,
        embed_num_probes: int = 0,
    ) -> None:
        self._runtime = EggRollJAXRuntime(
            policy,
            env_conf,
            steps_per_episode=steps_per_episode,
            num_envs=num_envs,
            deterministic_policy=deterministic_policy,
            seed_offset=seed_offset,
            embed_num_probes=embed_num_probes,
            vector_mode="absolute",
            es_key_fold=31,
            eval_key_fold=32,
            embed_key_fold=33,
            error_cls=ValueError,
            option_label="EggRoll JAX option",
        )

    def __getattr__(self, name: str):
        if name in _RUNTIME_ATTRS:
            return getattr(self._runtime, name)
        raise AttributeError(name)

    def flatten_params(self, params) -> np.ndarray:
        return self._runtime.codec.flatten(params)

    def decode_params(self, x):
        return self._runtime.codec.decode_absolute(x)

    def make_policy(self, x: np.ndarray):
        return self._runtime.make_policy(x, attr_name="_eggroll_uhd_x")

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        return self._runtime.evaluate(x, seed=seed)

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return self._runtime.evaluate_many(x_batch, seed=seed)

    def configure_embedding(self, num_probes: int) -> None:
        self._runtime.configure_embedding(num_probes)
