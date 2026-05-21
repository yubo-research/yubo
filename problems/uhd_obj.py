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
    from problems.isaaclab_env_adapters import is_isaaclab_env_tag
    from problems.jax_env_core import supports_jax_objective_tag
    from problems.pre_obj import is_hyperscalees_pretrain_env, is_nanoegg_pretrain_env
    from problems.text_obj import is_text_env

    return (
        str(env_tag).startswith("rwkv:distill:")
        or supports_jax_objective_tag(env_tag)
        or is_isaaclab_env_tag(env_tag)
        or is_hyperscalees_pretrain_env(env_tag)
        or is_nanoegg_pretrain_env(env_tag)
        or is_text_env(env_tag)
    )


def build_uhd_vector_objective(cfg: UHDConfig, *, embed_num_probes: int = 0) -> BuiltUHDVectorObjective:
    from problems.pre_obj import (
        HyperscaleESLLMVectorObjective,
        NanoEggPretrainVectorObjective,
    )
    from problems.rwkv_distill_objective import RWKVDistillObjective
    from problems.text_obj import TextObjective

    env_tag = str(cfg.env_tag)
    builders = (
        ("rwkv-distill", lambda: _build_rwkv_distill_objective(cfg, embed_num_probes, RWKVDistillObjective)),
        ("text", lambda: _build_text_objective(cfg, embed_num_probes, TextObjective)),
        ("nanoegg-pretrain", lambda: _build_nanoegg_objective(cfg, embed_num_probes, NanoEggPretrainVectorObjective)),
        ("jax-env", lambda: _build_jax_objective(cfg, embed_num_probes)),
        ("isaaclab", lambda: _build_isaaclab_objective(cfg, embed_num_probes)),
        ("hyperscalees-pretrain", lambda: _build_hyperscalees_objective(cfg, embed_num_probes, HyperscaleESLLMVectorObjective)),
    )
    for _name, builder in builders:
        result = builder()
        if result is not None:
            return result

    raise ValueError(f"Unsupported UHD vector objective env_tag: {env_tag!r}.")


def _build_rwkv_distill_objective(cfg: UHDConfig, embed_num_probes: int, cls):
    env_tag = str(cfg.env_tag)
    if not env_tag.startswith("rwkv:distill:"):
        return None
    objective = cls(cfg)
    if int(embed_num_probes) > 0:
        objective.configure_embedding(int(embed_num_probes))
    return BuiltUHDVectorObjective(objective=objective, source="rwkv-distill")


def _build_text_objective(cfg: UHDConfig, embed_num_probes: int, cls):
    from problems.text_obj import is_text_env

    env_tag = str(cfg.env_tag)
    if not is_text_env(env_tag):
        return None
    objective = cls(cfg)
    if int(embed_num_probes) > 0:
        objective.configure_embedding(int(embed_num_probes))
    return BuiltUHDVectorObjective(objective=objective, source="text")


def _build_nanoegg_objective(cfg: UHDConfig, embed_num_probes: int, cls):
    from problems.pre_obj import is_nanoegg_pretrain_env

    env_tag = str(cfg.env_tag)
    if not is_nanoegg_pretrain_env(env_tag):
        return None
    objective = cls(cfg)
    if int(embed_num_probes) > 0:
        objective.configure_embedding(int(embed_num_probes))
    return BuiltUHDVectorObjective(objective=objective, source="nanoegg-pretrain")


def _build_jax_objective(cfg: UHDConfig, embed_num_probes: int):
    from problems.jax_env_core import supports_jax_objective_tag

    env_tag = str(cfg.env_tag)
    if not supports_jax_objective_tag(env_tag):
        return None
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


def _build_isaaclab_objective(cfg: UHDConfig, embed_num_probes: int):
    env_tag = str(cfg.env_tag)
    from problems.isaaclab_env_adapters import is_isaaclab_env_tag

    if not is_isaaclab_env_tag(env_tag):
        return None
    from problems.isaaclab_score import build_isaaclab_evaluator

    objective = build_isaaclab_evaluator(cfg, embed_num_probes=embed_num_probes)
    return BuiltUHDVectorObjective(objective=objective, source="isaaclab")


def _build_hyperscalees_objective(cfg: UHDConfig, embed_num_probes: int, cls):
    env_tag = str(cfg.env_tag)
    from problems.pre_obj import is_hyperscalees_pretrain_env

    if not is_hyperscalees_pretrain_env(env_tag):
        return None
    objective = cls(cfg)
    if int(embed_num_probes) > 0:
        objective.configure_embedding(int(embed_num_probes))
    return BuiltUHDVectorObjective(objective=objective, source="hyperscalees-pretrain")
