from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import tensordict.nn as td_nn
import torch
import torch.nn as nn
import torchrl.data as tr_data
import torchrl.modules as tr_modules
import torchrl.modules.distributions as tr_dists

from analysis.data_io import write_config
from problems.env_conf import get_env_conf
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head
from rl.checkpointing import CheckpointManager
from rl.core import runtime as torchrl_common
from rl.core.env_setup import build_continuous_gym_env_setup

from .config import SACConfig
from .sac_setup_models import _ActorNet, _QNet, _QNetPixel
from .sac_setup_types import _EnvSetup, _Modules, _TrainingSetup


def build_env_setup(config: SACConfig) -> _EnvSetup:
    shared = build_continuous_gym_env_setup(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=bool(getattr(config, "from_pixels", False)),
        pixels_only=bool(getattr(config, "pixels_only", True)),
        get_env_conf_fn=get_env_conf,
        obs_scale_from_env_fn=torchrl_common.obs_scale_from_env,
    )
    env_conf = shared.env_conf
    from_pixels = getattr(env_conf, "from_pixels", False)
    if from_pixels:
        obs_dim = 64
    else:
        obs_space = getattr(env_conf, "state_space", None)
        if obs_space is None:
            gym_conf = getattr(env_conf, "gym_conf", None)
            obs_space = getattr(gym_conf, "state_space", None)
        if obs_space is None:
            raise ValueError("Observation space is missing on env_conf. Call env_conf.ensure_spaces() before SAC setup.")
        obs_dim = int(obs_space.shape[0])
    return _EnvSetup(
        env_conf=env_conf,
        problem_seed=int(shared.problem_seed),
        noise_seed_0=int(shared.noise_seed_0),
        obs_dim=obs_dim,
        act_dim=int(shared.act_dim),
        action_low=shared.action_low,
        action_high=shared.action_high,
        obs_lb=shared.obs_lb,
        obs_width=shared.obs_width,
    )


def _build_specs(config: SACConfig, *, from_pixels: bool) -> tuple[BackboneSpec, HeadSpec, HeadSpec]:
    backbone_name = "nature_cnn" if from_pixels else config.backbone_name
    actor_backbone_spec = BackboneSpec(
        name=backbone_name,
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=config.backbone_activation,
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = HeadSpec(
        hidden_sizes=tuple(config.actor_head_hidden_sizes),
        activation=config.head_activation,
    )
    critic_head_spec = HeadSpec(
        hidden_sizes=tuple(config.critic_head_hidden_sizes),
        activation=config.head_activation,
    )
    return (actor_backbone_spec, actor_head_spec, critic_head_spec)


def _build_actor(
    env: _EnvSetup,
    obs_scaler: nn.Module,
    actor_backbone_spec: BackboneSpec,
    actor_head_spec: HeadSpec,
) -> tuple[nn.Module, nn.Module, _ActorNet, tr_modules.ProbabilisticActor]:
    actor_backbone, actor_feat_dim = build_backbone(actor_backbone_spec, env.obs_dim)
    actor_head = build_mlp_head(actor_head_spec, actor_feat_dim, 2 * env.act_dim)
    actor_net = _ActorNet(actor_backbone, actor_head, obs_scaler, env.act_dim)
    actor = tr_modules.ProbabilisticActor(
        td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        distribution_class=tr_dists.TanhNormal,
        return_log_prob=True,
    )
    return (actor_backbone, actor_head, actor_net, actor)


def _build_q_pair(
    env: _EnvSetup,
    obs_scaler: nn.Module,
    actor_backbone_spec: BackboneSpec,
    critic_head_spec: HeadSpec,
    *,
    from_pixels: bool,
) -> tuple[nn.Module, nn.Module]:
    if from_pixels:
        q1_obs_encoder, q1_obs_dim = build_backbone(actor_backbone_spec, env.obs_dim)
        q2_obs_encoder, q2_obs_dim = build_backbone(actor_backbone_spec, env.obs_dim)
        q1_head = build_mlp_head(critic_head_spec, q1_obs_dim + env.act_dim, 1)
        q2_head = build_mlp_head(critic_head_spec, q2_obs_dim + env.act_dim, 1)
        q1 = _QNetPixel(q1_obs_encoder, q1_head, obs_scaler)
        q2 = _QNetPixel(q2_obs_encoder, q2_head, obs_scaler)
        return (q1, q2)
    critic_input_dim = env.obs_dim + env.act_dim
    q1_backbone, q1_feat_dim = build_backbone(actor_backbone_spec, critic_input_dim)
    q2_backbone, q2_feat_dim = build_backbone(actor_backbone_spec, critic_input_dim)
    q1_head = build_mlp_head(critic_head_spec, q1_feat_dim, 1)
    q2_head = build_mlp_head(critic_head_spec, q2_feat_dim, 1)
    q1 = _QNet(q1_backbone, q1_head, obs_scaler)
    q2 = _QNet(q2_backbone, q2_head, obs_scaler)
    return (q1, q2)


def build_modules(config: SACConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    obs_scaler = torchrl_common.ObsScaler(env.obs_lb, env.obs_width)
    from_pixels = bool(getattr(env.env_conf, "from_pixels", False))
    actor_backbone_spec, actor_head_spec, critic_head_spec = _build_specs(config, from_pixels=from_pixels)
    actor_backbone, actor_head, actor_net, actor = _build_actor(env, obs_scaler, actor_backbone_spec, actor_head_spec)
    q1, q2 = _build_q_pair(env, obs_scaler, actor_backbone_spec, critic_head_spec, from_pixels=from_pixels)
    actor.to(device)
    actor_net.to(device)
    q1.to(device)
    q2.to(device)
    obs_scaler.to(device)
    q1_target = copy.deepcopy(q1).to(device).eval()
    q2_target = copy.deepcopy(q2).to(device).eval()
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)
    log_alpha = nn.Parameter(
        torch.tensor(
            np.log(float(max(config.alpha_init, 1e-08))),
            dtype=torch.float32,
            device=device,
        )
    )
    actor_param_count = sum((p.numel() for p in actor_backbone.parameters())) + sum((p.numel() for p in actor_head.parameters()))
    if config.theta_dim is not None:
        assert actor_param_count == int(config.theta_dim), (
            actor_param_count,
            config.theta_dim,
        )
    return _Modules(
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        obs_scaler=obs_scaler,
        actor=actor,
        actor_model=actor_net,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        log_alpha=log_alpha,
    )


def build_training(config: SACConfig, modules: _Modules) -> _TrainingSetup:
    replay = tr_data.TensorDictReplayBuffer(
        storage=tr_data.LazyTensorStorage(int(config.replay_size)),
        batch_size=int(config.batch_size),
    )
    actor_params = list(modules.actor_backbone.parameters()) + list(modules.actor_head.parameters())
    critic_params = list(modules.q1.parameters()) + list(modules.q2.parameters())
    alpha_params = [modules.log_alpha]
    actor_optimizer = torch.optim.AdamW(actor_params, lr=float(config.learning_rate_actor), weight_decay=0.0)
    critic_optimizer = torch.optim.AdamW(critic_params, lr=float(config.learning_rate_critic), weight_decay=0.0)
    alpha_optimizer = torch.optim.AdamW(alpha_params, lr=float(config.learning_rate_alpha), weight_decay=0.0)
    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), config.to_dict())
    metrics_path = exp_dir / "metrics.jsonl"
    return _TrainingSetup(
        replay=replay,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        exp_dir=exp_dir,
        metrics_path=metrics_path,
        checkpoint_manager=CheckpointManager(exp_dir=exp_dir),
    )


def sac_update_shared(
    config: SACConfig,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    obs: torch.Tensor,
    act: torch.Tensor,
    rew: torch.Tensor,
    nxt: torch.Tensor,
    done: torch.Tensor,
) -> tuple[float, float, float]:
    from rl.core.sac_update import (
        SACUpdateBatch,
        SACUpdateHyperParams,
        SACUpdateModules,
        SACUpdateOptimizers,
        sac_update_step,
    )

    target_entropy = float(config.target_entropy) if config.target_entropy is not None else -float(act.shape[-1])
    return sac_update_step(
        modules=SACUpdateModules(
            actor=modules.actor_model,
            q1=modules.q1,
            q2=modules.q2,
            q1_target=modules.q1_target,
            q2_target=modules.q2_target,
            log_alpha=modules.log_alpha,
        ),
        optimizers=SACUpdateOptimizers(
            actor=training.actor_optimizer,
            critic=training.critic_optimizer,
            alpha=training.alpha_optimizer,
        ),
        batch=SACUpdateBatch(obs=obs, act=act, rew=rew, nxt=nxt, done=done),
        hyper=SACUpdateHyperParams(
            gamma=float(config.gamma),
            tau=float(config.tau),
            target_entropy=target_entropy,
        ),
    )


__all__ = [
    "build_env_setup",
    "build_modules",
    "build_training",
    "sac_update_shared",
]
