from __future__ import annotations

from pathlib import Path

import tensordict.nn as td_nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchrl.collectors as tr_collectors
import torchrl.envs as tr_envs
import torchrl.modules as tr_modules

import rl.backbone as backbone
import rl.checkpointing as rl_checkpointing
import rl.core.env_contract as torchrl_env_contract
import rl.core.runtime as torchrl_common
from analysis.data_io import write_config
from rl.core import torchrl_runtime as torchrl_runtime

from . import models as op_models
from .config import PPOConfig
from .core_collect_env import _make_collect_env, _make_collect_env_factory
from .core_types import _EnvSetup, _Modules, _TanhNormal, _TrainingSetup
from .core_utils import _count_unique_params, _unique_param_list


def build_modules(config: PPOConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    obs_contract = env.io_contract.observation
    backbone_name = torchrl_env_contract.resolve_backbone_name(config.backbone_name, obs_contract)
    backbone_spec = backbone.BackboneSpec(
        name=backbone_name,
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=config.backbone_activation,
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = backbone.HeadSpec(
        hidden_sizes=tuple(config.actor_head_hidden_sizes),
        activation=config.head_activation,
    )
    critic_head_spec = backbone.HeadSpec(
        hidden_sizes=tuple(config.critic_head_hidden_sizes),
        activation=config.head_activation,
    )
    if config.share_backbone:
        shared_backbone, feat_dim = backbone.build_backbone(backbone_spec, env.obs_dim)
        actor_backbone = shared_backbone
        critic_backbone = shared_backbone
        actor_feat_dim = feat_dim
        critic_feat_dim = feat_dim
    else:
        actor_backbone, actor_feat_dim = backbone.build_backbone(backbone_spec, env.obs_dim)
        critic_backbone, critic_feat_dim = backbone.build_backbone(backbone_spec, env.obs_dim)
    actor_head = backbone.build_mlp_head(actor_head_spec, actor_feat_dim, env.act_dim)
    critic_head = backbone.build_mlp_head(critic_head_spec, critic_feat_dim, 1)
    obs_scaler = torchrl_common.ObsScaler(env.obs_lb, env.obs_width)
    critic_net = op_models.CriticNet(critic_backbone, critic_head, obs_scaler, obs_contract=obs_contract)
    critic = td_nn.TensorDictModule(critic_net, in_keys=["observation"], out_keys=["state_value"])
    if env.is_discrete:
        log_std = None
        actor_net = op_models.DiscreteActorNet(actor_backbone, actor_head, obs_scaler, obs_contract=obs_contract)
        actor_module = td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
        actor = tr_modules.ProbabilisticActor(
            actor_module,
            in_keys=["logits"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )
        actor_param_count = _count_unique_params(actor_backbone, actor_head)
    else:
        log_std = nn.Parameter(torch.full((env.act_dim,), float(config.log_std_init)))
        actor_net = op_models.ActorNet(actor_backbone, actor_head, log_std, obs_scaler, obs_contract=obs_contract)
        actor_module = td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
        actor = tr_modules.ProbabilisticActor(
            actor_module,
            in_keys=["loc", "scale"],
            distribution_class=_TanhNormal,
            distribution_kwargs={"low": env.action_low, "high": env.action_high},
            return_log_prob=True,
        )
        actor_param_count = _count_unique_params(actor_backbone, actor_head, extra_params=[log_std])
    actor.to(device)
    critic.to(device)
    obs_scaler.to(device)
    if config.theta_dim is not None:
        assert actor_param_count == int(config.theta_dim), (
            actor_param_count,
            config.theta_dim,
        )
    return _Modules(
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        critic_backbone=critic_backbone,
        critic_head=critic_head,
        log_std=log_std,
        obs_scaler=obs_scaler,
        actor=actor,
        critic=critic,
    )


def build_training(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    *,
    runtime: torchrl_runtime.TorchRLRuntime,
) -> _TrainingSetup:
    import torchrl.objectives as tr_objectives

    frames_per_batch = int(config.num_envs * config.num_steps)
    num_iterations = int(config.total_timesteps // frames_per_batch)
    if num_iterations <= 0:
        raise ValueError("total_timesteps too small for num_envs * num_steps.")
    env_conf = env.env_conf
    vec_env = None
    if runtime.collector_backend == "single":
        env_factory = _make_collect_env_factory(env_conf, int(config.num_envs))
        if runtime.single_env_backend == "parallel":
            vec_env = tr_envs.ParallelEnv(int(config.num_envs), env_factory, serial_for_single=True)
        else:
            vec_env = tr_envs.SerialEnv(int(config.num_envs), env_factory, serial_for_single=True)
    loss_module = tr_objectives.ClipPPOLoss(
        modules.actor,
        modules.critic,
        clip_epsilon=config.clip_coef,
        entropy_coeff=config.ent_coef,
        critic_coeff=config.vf_coef,
        normalize_advantage=config.norm_adv,
        clip_value=config.clip_vloss,
        functional=False,
    )
    gae = tr_objectives.value.GAE(gamma=config.gamma, lmbda=config.gae_lambda, value_network=modules.critic)
    train_params = _unique_param_list(
        modules.actor,
        modules.critic,
        extra_params=[modules.log_std] if modules.log_std is not None else None,
    )
    optimizer = optim.AdamW(train_params, lr=config.learning_rate, eps=1e-05, weight_decay=0.0)
    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), config.to_dict())
    metrics_path = exp_dir / "metrics.jsonl"
    return _TrainingSetup(
        frames_per_batch=frames_per_batch,
        num_iterations=num_iterations,
        env=vec_env,
        loss_module=loss_module,
        gae=gae,
        train_params=train_params,
        optimizer=optimizer,
        exp_dir=exp_dir,
        metrics_path=metrics_path,
        checkpoint_manager=rl_checkpointing.CheckpointManager(exp_dir=exp_dir),
    )


def _build_collector(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    runtime: torchrl_runtime.TorchRLRuntime,
    remaining_iterations: int,
):
    total_frames = int(remaining_iterations * training.frames_per_batch)
    if runtime.collector_backend == "single":
        if training.env is None:
            raise RuntimeError("single collector backend requires training.env")
        return tr_collectors.Collector(
            training.env,
            modules.actor,
            frames_per_batch=training.frames_per_batch,
            total_frames=total_frames,
            **torchrl_common.collector_device_kwargs(runtime.device),
        )
    if runtime.collector_workers is None:
        raise RuntimeError("multi collector backend requires collector_workers")
    num_workers = int(runtime.collector_workers)
    create_env_fns = [lambda i=i: _make_collect_env(env.env_conf, env_index=i) for i in range(num_workers)]
    frames_per_batch = [int(config.num_steps)] * num_workers
    collector_cls = tr_collectors.MultiAsyncCollector if runtime.collector_backend == "multi_async" else tr_collectors.MultiSyncCollector
    return collector_cls(
        create_env_fns,
        modules.actor,
        num_workers=num_workers,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        env_device=torch.device("cpu"),
        policy_device=runtime.device,
        storing_device=torch.device("cpu"),
    )
