from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.core.sac_update import (
    SACUpdateBatch,
    SACUpdateHyperParams,
    SACUpdateModules,
    SACUpdateOptimizers,
    sac_update_step,
)
from rl.offpolicy import model_utils

from .config import SACConfig

ActorNet = model_utils.ActorNet
QNet = model_utils.QNet
QNetPixel = model_utils.QNetPixel
capture_actor_state = model_utils.capture_actor_state
restore_actor_state = model_utils.restore_actor_state
use_actor_state = model_utils.use_actor_state


@dataclasses.dataclass
class SACModules(model_utils.OffPolicyModules):
    log_alpha: nn.Parameter


@dataclasses.dataclass
class SACOptimizers(model_utils.OffPolicyOptimizers):
    alpha_optimizer: optim.Optimizer


def build_modules(config: SACConfig, env: Any, obs_spec: Any, *, device: torch.device) -> SACModules:
    shared = model_utils.build_modules(config, env, obs_spec, device=device)
    log_alpha = nn.Parameter(
        torch.tensor(
            np.log(float(max(config.alpha_init, 1e-8))),
            dtype=torch.float32,
            device=device,
        )
    )
    return SACModules(**shared.__dict__, log_alpha=log_alpha)


def build_optimizers(config: SACConfig, modules: SACModules) -> SACOptimizers:
    shared = model_utils.build_optimizers(config, modules)
    return SACOptimizers(
        **shared.__dict__,
        alpha_optimizer=optim.AdamW([modules.log_alpha], lr=float(config.learning_rate_alpha), weight_decay=0.0),
    )


def alpha(modules: SACModules) -> torch.Tensor:
    return modules.log_alpha.exp()


def sac_update(
    config: SACConfig,
    modules: SACModules,
    optimizers: SACOptimizers,
    replay,
    *,
    device: torch.device,
) -> tuple[float, float, float]:
    obs, act, rew, nxt, done = replay.sample(int(config.batch_size), device=device)
    target_entropy = float(config.target_entropy) if config.target_entropy is not None else -float(act.shape[-1])
    use_gac = str(getattr(config, "actor_type", "gaussian")).strip().lower() == "gac"
    return sac_update_step(
        modules=SACUpdateModules(
            actor=modules.actor,
            q1=modules.q1,
            q2=modules.q2,
            q1_target=modules.q1_target,
            q2_target=modules.q2_target,
            log_alpha=modules.log_alpha,
        ),
        optimizers=SACUpdateOptimizers(
            actor=optimizers.actor_optimizer,
            critic=optimizers.critic_optimizer,
            alpha=optimizers.alpha_optimizer,
        ),
        batch=SACUpdateBatch(obs=obs, act=act, rew=rew, nxt=nxt, done=done),
        hyper=SACUpdateHyperParams(
            gamma=float(config.gamma),
            tau=float(config.tau),
            target_entropy=target_entropy,
            use_gac=use_gac,
        ),
    )


__all__ = [
    "ActorNet",
    "QNet",
    "QNetPixel",
    "SACModules",
    "SACOptimizers",
    "alpha",
    "build_modules",
    "build_optimizers",
    "capture_actor_state",
    "restore_actor_state",
    "sac_update",
    "use_actor_state",
]
