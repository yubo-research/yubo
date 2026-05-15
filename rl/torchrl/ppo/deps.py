from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rl.torchrl._lazy_exports import resolve_export

"""PPO dependency facade.

This module intentionally uses lazy imports to keep dependency depth low and
avoid cycles. Callers may access attributes like ``deps.tr_modules``; they are
resolved on demand via ``__getattr__``.
"""

if TYPE_CHECKING:
    import torchrl.envs.transforms as tr_transforms
    import torchrl.modules as tr_modules
    import torchrl.modules.distributions as tr_dists
    import torchrl.objectives as tr_objectives

    import common.experiment_seeds as experiment_seeds
    import common.seed_all as seed_all
    import rl.backbone as backbone
    import rl.checkpointing as rl_checkpointing
    import rl.core.env_contract as torchrl_env_contract
    import rl.core.runtime as torchrl_common
    import rl.core.torchrl_runtime as torchrl_runtime
    import rl.eval_noise as eval_noise
    import rl.torchrl.patches as torchrl_patches
    import rl.torchrl.ppo.actor_eval as torchrl_actor_eval
    from analysis.data_io import write_config
    from problems.problem import build_problem
    from rl.torchrl.ppo.checkpoint_io import save_final_checkpoint, save_periodic_checkpoint

    build_eval_plan = eval_noise.build_eval_plan
    normalize_eval_noise_mode = eval_noise.normalize_eval_noise_mode


_EXPORTS: dict[str, tuple[str, str | None]] = {
    "tr_transforms": ("torchrl.envs.transforms", None),
    "tr_modules": ("torchrl.modules", None),
    "tr_dists": ("torchrl.modules.distributions", None),
    "tr_objectives": ("torchrl.objectives", None),
    "backbone": ("rl.backbone", None),
    "rl_checkpointing": ("rl.checkpointing", None),
    "experiment_seeds": ("common.experiment_seeds", None),
    "write_config": ("analysis.data_io", "write_config"),
    "seed_all": ("common.seed_all", "seed_all"),
    "build_problem": ("problems.problem", "build_problem"),
    "torchrl_env_contract": ("rl.core.env_contract", None),
    "torchrl_common": ("rl.core.runtime", None),
    "torchrl_runtime": ("rl.core.torchrl_runtime", None),
    "torchrl_patches": ("rl.torchrl.patches", None),
    "build_eval_plan": ("rl.eval_noise", "build_eval_plan"),
    "normalize_eval_noise_mode": ("rl.eval_noise", "normalize_eval_noise_mode"),
    "torchrl_actor_eval": ("rl.torchrl.ppo.actor_eval", None),
    "save_final_checkpoint": ("rl.torchrl.ppo.checkpoint_io", "save_final_checkpoint"),
    "save_periodic_checkpoint": ("rl.torchrl.ppo.checkpoint_io", "save_periodic_checkpoint"),
}


__all__ = [
    "tr_transforms",
    "tr_modules",
    "tr_dists",
    "tr_objectives",
    "rl_checkpointing",
    "experiment_seeds",
    "backbone",
    "torchrl_actor_eval",
    "torchrl_common",
    "torchrl_env_contract",
    "torchrl_runtime",
    "torchrl_patches",
    "build_eval_plan",
    "normalize_eval_noise_mode",
    "save_final_checkpoint",
    "save_periodic_checkpoint",
    "write_config",
    "seed_all",
    "build_problem",
]


def __getattr__(name: str) -> Any:
    return resolve_export(_EXPORTS, name)
