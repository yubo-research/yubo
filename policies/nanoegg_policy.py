from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NanoEggPretrainPolicyConfig:
    env_tag: str
    policy_tag: str
    problem_seed: int | None
    default_search_dim: int = 4096


class NanoEggPretrainPolicy:
    """NanoEgg model/objective metadata for the EggRoll experiment runner."""

    is_nanoegg_pretrain_policy = True

    def __init__(self, env_runtime: Any, policy_tag: str, *, default_search_dim: int = 4096) -> None:
        env_tag = str(getattr(env_runtime, "env_name", getattr(env_runtime, "env_tag", "")))
        self.config = NanoEggPretrainPolicyConfig(
            env_tag=env_tag,
            policy_tag=str(policy_tag),
            problem_seed=getattr(env_runtime, "problem_seed", None),
            default_search_dim=int(default_search_dim),
        )
        from problems.pre_obj import resolve_nanoegg_pretrain_spec

        self.spec = resolve_nanoegg_pretrain_spec(env_tag, str(policy_tag))
        self.x = np.zeros((int(default_search_dim),), dtype=np.float64)
        self.last_params = None

    @property
    def env_name(self) -> str:
        return self.config.env_tag

    @property
    def policy_tag(self) -> str:
        return self.config.policy_tag

    @property
    def problem_seed(self) -> int | None:
        return self.config.problem_seed

    def num_params(self) -> int:
        return int(self.x.size)

    def clone(self):
        return copy.deepcopy(self)

    def with_snapshot(self, snapshot):
        policy = self.clone()
        policy.x = np.asarray(snapshot.x, dtype=np.float64).copy()
        policy.last_params = snapshot.params
        return policy

    def make_objective(
        self,
        *,
        search_dim: int | None = None,
        delta_scale: float = 10000.0,
        generation_length: int | None = None,
        num_envs: int = 1,
        lora_only: bool = True,
        basis_max_leaves: int | None = 32,
        sub_dataset_size: int | None = None,
        hf_home: str | None = None,
    ):
        from problems.nanoegg_obj import build_nanoegg_uhd_objective

        dim = int(search_dim or self.config.default_search_dim)
        cfg = SimpleNamespace(
            pretrain_generation_length=None if generation_length is None else int(generation_length),
            max_tokens=1024,
            num_envs=int(num_envs),
            problem_seed=self.config.problem_seed,
            pretrain_search_dim=dim,
            pretrain_delta_scale=float(delta_scale),
            pretrain_lora_only=bool(lora_only),
            pretrain_basis_max_leaves=basis_max_leaves,
            steps_per_episode=1,
            hf_home=hf_home,
            sub_dataset_size=sub_dataset_size,
        )
        objective = build_nanoegg_uhd_objective(cfg=cfg, spec=self.spec)
        self.x = np.asarray(objective.x0, dtype=np.float64).copy()
        return objective

    def get_params(self):
        raise NotImplementedError("NanoEgg params are managed by the EggRoll experiment path.")

    def set_params(self, _flat_params) -> None:
        raise NotImplementedError("NanoEgg params are managed by the EggRoll experiment path.")

    def __call__(self, _state):
        raise NotImplementedError("NanoEgg policies must be evaluated with optimizer.name='eggroll'.")


class NanoEggPretrainPolicyFactory:
    def __init__(self, policy_tag: str) -> None:
        self._policy_tag = str(policy_tag)

    def __call__(self, env_runtime: Any) -> NanoEggPretrainPolicy:
        return NanoEggPretrainPolicy(env_runtime, self._policy_tag)
