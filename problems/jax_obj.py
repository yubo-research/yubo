from __future__ import annotations

import numpy as np

from optimizer.eggroll_runtime import EggRollJAXRuntime


class EggRollJAXVectorObjective:
    """Functional flat-vector objective for EggRoll/HyperscaleES policies."""

    def __init__(
        self,
        policy,
        env_conf,
        *,
        steps_per_episode: int = 200,
        eval_episodes: int = 1,
        deterministic_policy: bool = False,
        seed_offset: int = 0,
        embed_num_probes: int = 0,
    ) -> None:
        self._runtime = EggRollJAXRuntime(
            policy,
            env_conf,
            steps_per_episode=steps_per_episode,
            eval_episodes=eval_episodes,
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

    @property
    def dim(self) -> int:
        return self._runtime.dim

    @property
    def x0(self) -> np.ndarray:
        return self._runtime.x0

    @property
    def steps_per_episode(self) -> int:
        return self._runtime.steps_per_episode

    @property
    def eval_episodes(self) -> int:
        return self._runtime.eval_episodes

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

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        return self._runtime.embed_many(x_batch)

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self._runtime.embed(x)

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        return self._runtime.sample_noise(
            seed=seed,
            num_dim_target=num_dim_target,
            num_module_target=num_module_target,
        )

    def sample_eggroll_noiser_noise(
        self,
        x: np.ndarray,
        *,
        seed: int,
        noiser_name: str = "eggroll",
        rank: int = 1,
        group_size: int = 0,
        freeze_nonlora: bool = False,
    ) -> np.ndarray:
        return self._runtime.sample_eggroll_noiser_noise(
            x,
            seed=seed,
            noiser_name=noiser_name,
            rank=rank,
            group_size=group_size,
            freeze_nonlora=freeze_nonlora,
        )
