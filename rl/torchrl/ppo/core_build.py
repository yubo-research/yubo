from __future__ import annotations

from pathlib import Path

import tensordict.nn as td_nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchrl.envs as tr_envs
import torchrl.modules as tr_modules

from analysis.data_io import write_config
from rl import backbone, checkpointing
from rl.config_model_defaults import resolve_ppo_model_settings
from rl.core import env_contract, torchrl_collectors, torchrl_runtime
from rl.core.runtime import ObsScaler, collector_device_kwargs

from . import models
from .config import PPOConfig
from .core_collect_env import _make_collect_env, _make_collect_env_factory
from .core_types import _EnvSetup, _Modules, _TanhNormal, _TrainingSetup
from .core_utils import _unique_param_list


def build_modules(config: PPOConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    obs_contract = env.io_contract.observation
    model = resolve_ppo_model_settings(config)
    backbone_name = env_contract.resolve_backbone_name(model.backbone_name, obs_contract)
    backbone_spec = backbone.BackboneSpec(
        name=backbone_name,
        hidden_sizes=tuple(model.backbone_hidden_sizes),
        activation=model.backbone_activation,
        layer_norm=bool(model.backbone_layer_norm),
    )
    actor_head_spec = backbone.HeadSpec(
        hidden_sizes=tuple(model.actor_head_hidden_sizes),
        activation=model.head_activation,
    )
    critic_head_spec = backbone.HeadSpec(
        hidden_sizes=tuple(model.critic_head_hidden_sizes),
        activation=model.head_activation,
    )
    if model.share_backbone:
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
    obs_scaler = ObsScaler(env.obs_lb, env.obs_width)
    critic_net = models.CriticNet(critic_backbone, critic_head, obs_scaler, obs_contract=obs_contract)
    critic = td_nn.TensorDictModule(critic_net, in_keys=["observation"], out_keys=["state_value"])
    if env.is_discrete:
        log_std = None
        actor_net = models.DiscreteActorNet(actor_backbone, actor_head, obs_scaler, obs_contract=obs_contract)
        actor_module = td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
        actor = tr_modules.ProbabilisticActor(
            actor_module,
            in_keys=["logits"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )
    else:
        log_std = nn.Parameter(torch.full((env.act_dim,), float(model.log_std_init)))
        actor_net = models.ActorNet(actor_backbone, actor_head, log_std, obs_scaler, obs_contract=obs_contract)
        actor_module = td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
        actor = tr_modules.ProbabilisticActor(
            actor_module,
            in_keys=["loc", "scale"],
            distribution_class=_TanhNormal,
            distribution_kwargs={"low": env.action_low, "high": env.action_high},
            return_log_prob=True,
        )
    actor.to(device)
    critic.to(device)
    obs_scaler.to(device)
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

    frames_per_batch = int(config.collector.frames_per_batch)
    num_iterations = int(config.collector.total_frames // frames_per_batch)
    if num_iterations <= 0:
        raise ValueError("collector.total_frames too small for collector.frames_per_batch.")
    env_conf = env.env_conf
    vec_env = None
    if runtime.collector_backend == "single":
        env_factory = _make_collect_env_factory(env_conf, int(config.collector.num_envs))
        if runtime.single_env_backend == "parallel":
            vec_env = tr_envs.ParallelEnv(int(config.collector.num_envs), env_factory, serial_for_single=True)
        else:
            vec_env = tr_envs.SerialEnv(int(config.collector.num_envs), env_factory, serial_for_single=True)
    loss_module = tr_objectives.ClipPPOLoss(
        modules.actor,
        modules.critic,
        clip_epsilon=config.loss.clip_epsilon,
        entropy_coeff=config.loss.entropy_coeff,
        critic_coeff=config.loss.critic_coeff,
        normalize_advantage=config.loss.normalize_advantage,
        clip_value=config.loss.clip_value_loss,
        functional=False,
    )
    gae = tr_objectives.value.GAE(gamma=config.loss.gamma, lmbda=config.loss.gae_lambda, value_network=modules.critic)
    train_params = _unique_param_list(
        modules.actor,
        modules.critic,
        extra_params=[modules.log_std] if modules.log_std is not None else None,
    )
    optimizer = optim.AdamW(train_params, lr=config.optim.lr, eps=1e-05, weight_decay=0.0)
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
        checkpoint_manager=checkpointing.CheckpointManager(exp_dir=exp_dir),
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
        return torchrl_collectors.collector_class("Collector")(
            training.env,
            modules.actor,
            frames_per_batch=training.frames_per_batch,
            total_frames=total_frames,
            **collector_device_kwargs(runtime.device),
        )
    if runtime.collector_workers is None:
        raise RuntimeError("multi collector backend requires collector_workers")
    num_workers = int(runtime.collector_workers)
    create_env_fns = [lambda i=i: _make_collect_env(env.env_conf, env_index=i) for i in range(num_workers)]
    if int(config.collector.frames_per_batch) % num_workers != 0:
        raise ValueError("collector.frames_per_batch must be divisible by collector.workers for multi collectors.")
    frames_per_batch = [int(config.collector.frames_per_batch) // num_workers] * num_workers
    collector_cls = torchrl_collectors.collector_class("MultiAsyncCollector" if runtime.collector_backend == "multi_async" else "MultiSyncCollector")
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
