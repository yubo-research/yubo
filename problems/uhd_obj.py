from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ops.uhd_config import UHDConfig


class UHDVectorObjective(Protocol):
    @property
    def dim(self) -> int: ...

    @property
    def x0(self) -> np.ndarray: ...

    @property
    def steps_per_episode(self) -> int: ...

    @property
    def num_envs(self) -> int: ...

    def make_policy(self, x: np.ndarray): ...

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]: ...

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]: ...

    def configure_embedding(self, num_probes: int) -> None: ...

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray: ...

    def embed(self, x: np.ndarray) -> np.ndarray: ...

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class BuiltUHDVectorObjective:
    objective: UHDVectorObjective
    source: str


def supports_uhd_vector_objective(env_tag: str) -> bool:
    from problems.eggroll_env_adapters import supports_eggroll_env_adapter
    from problems.pre_obj import is_hyperscalees_pretrain_env, is_nanoegg_pretrain_env
    from problems.text_obj import is_text_env

    return supports_eggroll_env_adapter(env_tag) or is_hyperscalees_pretrain_env(env_tag) or is_nanoegg_pretrain_env(env_tag) or is_text_env(env_tag)


def build_uhd_vector_objective(cfg: UHDConfig, *, embed_num_probes: int = 0) -> BuiltUHDVectorObjective:
    from problems.eggroll_env_adapters import supports_eggroll_env_adapter
    from problems.pre_obj import (
        HyperscaleESLLMVectorObjective,
        NanoEggPretrainVectorObjective,
        is_hyperscalees_pretrain_env,
        is_nanoegg_pretrain_env,
    )
    from problems.text_obj import TextObjective, is_text_env

    env_tag = str(cfg.env_tag)
    if is_text_env(env_tag):
        objective = TextObjective(cfg)
        if int(embed_num_probes) > 0:
            objective.configure_embedding(int(embed_num_probes))
        return BuiltUHDVectorObjective(objective=objective, source="text")

    if is_nanoegg_pretrain_env(env_tag):
        objective = NanoEggPretrainVectorObjective(cfg)
        if int(embed_num_probes) > 0:
            objective.configure_embedding(int(embed_num_probes))
        return BuiltUHDVectorObjective(objective=objective, source="nanoegg-pretrain")

    if supports_eggroll_env_adapter(env_tag):
        if cfg.policy_tag is None:
            raise ValueError("UHD vector JAX env objectives require policy_tag.")
        from problems.jax_obj import EggRollJAXVectorObjective
        from problems.problem import build_problem

        problem = build_problem(
            env_tag,
            cfg.policy_tag,
            problem_seed=cfg.problem_seed,
            noise_seed_0=cfg.noise_seed_0,
        )
        objective = EggRollJAXVectorObjective(
            problem.build_policy(),
            problem.env,
            steps_per_episode=cfg.steps_per_episode,
            num_envs=cfg.num_envs,
            deterministic_policy=cfg.deterministic_policy,
            seed_offset=cfg.seed_offset,
            embed_num_probes=embed_num_probes,
        )
        return BuiltUHDVectorObjective(objective=objective, source="jax-env")

    if is_hyperscalees_pretrain_env(env_tag):
        objective = HyperscaleESLLMVectorObjective(cfg)
        if int(embed_num_probes) > 0:
            objective.configure_embedding(int(embed_num_probes))
        return BuiltUHDVectorObjective(objective=objective, source="hyperscalees-pretrain")

    raise ValueError(f"Unsupported UHD vector objective env_tag: {env_tag!r}.")
