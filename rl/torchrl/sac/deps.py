from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rl.torchrl._lazy_exports import resolve_export

"""SAC dependency facade.

Keep dependency depth low by lazily resolving heavy imports.
"""

if TYPE_CHECKING:
    import tensordict.nn as td_nn
    import torchrl.data as tr_data
    import torchrl.envs.transforms as tr_transforms
    import torchrl.modules as tr_modules
    import torchrl.modules.distributions as tr_dists
    import torchrl.objectives as tr_objectives

    import common.experiment_seeds as experiment_seeds
    import rl.backbone as backbone
    import rl.checkpointing as checkpointing
    import rl.core.episode_rollout as episode_rollout
    import rl.core.runtime as torchrl_common
    import rl.core.torchrl_runtime as torchrl_runtime
    import rl.eval_noise as eval_noise
    import rl.registry as registry
    import rl.torchrl.patches as torchrl_patches
    import rl.torchrl.sac.actor_eval as torchrl_sac_actor_eval
    import rl.torchrl.sac.loop as torchrl_sac_loop
    from analysis.data_io import write_config
    from common.rl_helpers import seed_all, video
    from problems.problem import build_problem

    BackboneSpec = backbone.BackboneSpec
    HeadSpec = backbone.HeadSpec
    build_backbone = backbone.build_backbone
    build_mlp_head = backbone.build_mlp_head
    CheckpointManager = checkpointing.CheckpointManager
    load_checkpoint = checkpointing.load_checkpoint


_EXPORTS: dict[str, tuple[str, str | None]] = {
    "td_nn": ("tensordict.nn", None),
    "tr_data": ("torchrl.data", None),
    "tr_transforms": ("torchrl.envs.transforms", None),
    "tr_modules": ("torchrl.modules", None),
    "tr_dists": ("torchrl.modules.distributions", None),
    "tr_objectives": ("torchrl.objectives", None),
    "experiment_seeds": ("common.experiment_seeds", None),
    "episode_rollout": ("rl.core.episode_rollout", None),
    "eval_noise": ("rl.eval_noise", None),
    "registry": ("rl.registry", None),
    "write_config": ("analysis.data_io", "write_config"),
    "seed_all": ("common.rl_helpers", "seed_all"),
    "video": ("common.rl_helpers", "video"),
    "build_problem": ("problems.problem", "build_problem"),
    "BackboneSpec": ("rl.backbone", "BackboneSpec"),
    "HeadSpec": ("rl.backbone", "HeadSpec"),
    "build_backbone": ("rl.backbone", "build_backbone"),
    "build_mlp_head": ("rl.backbone", "build_mlp_head"),
    "CheckpointManager": ("rl.checkpointing", "CheckpointManager"),
    "load_checkpoint": ("rl.checkpointing", "load_checkpoint"),
    "torchrl_common": ("rl.core.runtime", None),
    "torchrl_runtime": ("rl.core.torchrl_runtime", None),
    "torchrl_patches": ("rl.torchrl.patches", None),
    "torchrl_sac_actor_eval": ("rl.torchrl.sac.actor_eval", None),
    "torchrl_sac_loop": ("rl.torchrl.sac.loop", None),
}


__all__ = [
    "td_nn",
    "tr_data",
    "tr_transforms",
    "tr_modules",
    "tr_dists",
    "tr_objectives",
    "torchrl_common",
    "torchrl_runtime",
    "torchrl_patches",
    "torchrl_sac_actor_eval",
    "torchrl_sac_loop",
    "write_config",
    "seed_all",
    "build_problem",
    "CheckpointManager",
    "load_checkpoint",
    "BackboneSpec",
    "HeadSpec",
    "build_backbone",
    "build_mlp_head",
    "experiment_seeds",
    "video",
    "episode_rollout",
    "eval_noise",
    "registry",
]


def __getattr__(name: str) -> Any:
    return resolve_export(_EXPORTS, name)
