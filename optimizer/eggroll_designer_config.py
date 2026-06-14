from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.eggroll_options import eggroll_bool as _as_bool
from optimizer.eggroll_options import unit_decay as _as_unit_decay
from optimizer.step_size_adapter import StepSizeAdapter


@dataclass(frozen=True)
class _EggRollDesignerConfig:
    noiser: str = "eggroll"
    sigma: float = 0.05
    adapt_sigma: bool = False
    lr: float = 0.02
    lr_decay: float = 1.0
    sigma_decay: float = 1.0
    rank: int = 8
    rank_transform: bool = False
    deterministic_policy: bool = False
    steps: int = 200
    num_envs: int = 8
    optax: str = "adamw"
    b1: float = 0.9
    b2: float = 0.999
    weight_decay: float = 0.0
    group_size: int = 0
    freeze_nonlora: bool = False
    noise_reuse: int = 0
    use_batched_update: bool = True
    suppress_noiser_stdout: bool = True
    seed_offset: int = 0
    batch_size: int = 16
    search_dim: int = 4096
    delta_scale: float = 10000.0
    generation_length: int | None = None
    lora_only: bool = True
    basis_max_leaves: int | None = 32
    sub_dataset_size: int | None = None
    hf_home: str | None = None
    jax_sim: bool = False


def configure_adapt_sigma(designer: Any, cfg: _EggRollDesignerConfig, *, dim: int) -> None:
    adapt_sigma = _as_bool(cfg.adapt_sigma, name="adapt_sigma")
    sigma_decay = _as_unit_decay(cfg.sigma_decay, name="sigma_decay")
    if adapt_sigma and sigma_decay != 1.0:
        raise NoSuchDesignerError("EggRoll options 'adapt_sigma=true' and 'sigma_decay' != 1.0 are mutually exclusive.")
    designer._sigma = float(cfg.sigma)
    designer._sigma_decay = sigma_decay
    designer._adapt_sigma = adapt_sigma
    designer._adapter = StepSizeAdapter(sigma_0=designer._sigma, dim=int(dim)) if adapt_sigma else None


__all__ = ["_EggRollDesignerConfig", "configure_adapt_sigma"]
