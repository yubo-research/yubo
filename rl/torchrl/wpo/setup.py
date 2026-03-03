from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchrl.envs as tr_envs

from rl.core.continuous_actions import scale_action_to_env, unscale_action_from_env
from rl.core.env_setup import build_continuous_gym_env_setup
from rl.core.wpo_update import WPOUpdateBatch, WPOUpdateHyperParams, WPOUpdateModules, WPOUpdateOptimizers, wpo_update_step
from rl.torchrl.offpolicy import models as offpolicy_models

from . import deps as wpo_deps
from .config import WPOConfig


def _scale_action_to_env(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return scale_action_to_env(action, action_low, action_high, clip=False)


def _unscale_action_from_env(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return unscale_action_from_env(action, action_low, action_high, clip=True)


@dataclasses.dataclass(frozen=True)
class _EnvSetup:
    env_conf: object
    problem_seed: int
    noise_seed_0: int
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_lb: np.ndarray | None
    obs_width: np.ndarray | None


@dataclasses.dataclass
class _Modules:
    actor_backbone: nn.Module
    actor_head: nn.Module
    obs_scaler: wpo_deps.torchrl_common.ObsScaler
    actor: wpo_deps.tr_modules.ProbabilisticActor
    actor_model: nn.Module
    actor_target: nn.Module
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    log_alpha_mean: nn.Parameter
    log_alpha_stddev: nn.Parameter


@dataclasses.dataclass
class _TrainingSetup:
    replay: wpo_deps.tr_data.TensorDictReplayBuffer
    actor_optimizer: torch.optim.AdamW
    critic_optimizer: torch.optim.AdamW
    dual_optimizer: torch.optim.AdamW
    exp_dir: Path
    metrics_path: Path
    checkpoint_manager: wpo_deps.CheckpointManager


@dataclasses.dataclass
class _TrainState:
    start_step: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None


_ActorNet = offpolicy_models.ActorNet
_QNet = offpolicy_models.QNet
_QNetPixel = offpolicy_models.QNetPixel


class _ScaleActionToEnv(nn.Module):
    def __init__(self, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.register_buffer("_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        width = (self._high - self._low).clamp(min=1e-8)
        return self._low + width * (1.0 + action) / 2.0


def _is_dm_control_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("dm_control/")


def _make_collect_env_dm_control_wpo(env_conf, env_setup: _EnvSetup, env_index: int = 0):
    from problems.shimmy_dm_control import make as make_dm_env

    seed = int(env_setup.problem_seed) + env_index
    from_pixels = getattr(env_conf, "from_pixels", False)
    pixels_only = getattr(env_conf, "pixels_only", True)
    base = make_dm_env(env_conf.env_name, from_pixels=from_pixels, pixels_only=pixels_only)
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    if from_pixels:
        to_tensor = wpo_deps.tr_transforms.ToTensorImage(in_keys=["pixels"], out_keys=["observation"], from_int=True)
        resize = wpo_deps.tr_transforms.Resize(84, 84, in_keys=["observation"])
        transforms = wpo_deps.tr_transforms.Compose(to_tensor, resize, wpo_deps.tr_transforms.DoubleToFloat())
    else:
        transforms = wpo_deps.tr_transforms.DoubleToFloat()
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), transforms)


def _make_collect_env_wpo(env_conf, env_setup: _EnvSetup, env_index: int = 0):
    if _is_dm_control_env(env_conf):
        return _make_collect_env_dm_control_wpo(env_conf, env_setup, env_index)
    base = env_conf.make()
    seed = int(env_setup.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), wpo_deps.tr_transforms.DoubleToFloat())


def build_env_setup(config: WPOConfig) -> _EnvSetup:
    shared = build_continuous_gym_env_setup(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=bool(getattr(config, "from_pixels", False)),
        pixels_only=bool(getattr(config, "pixels_only", True)),
        get_env_conf_fn=wpo_deps.get_env_conf,
        obs_scale_from_env_fn=wpo_deps.torchrl_common.obs_scale_from_env,
    )
    env_conf = shared.env_conf
    from_pixels = getattr(env_conf, "from_pixels", False)
    obs_dim = 64 if from_pixels else int(env_conf.gym_conf.state_space.shape[0])
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


def _build_specs(config: WPOConfig, *, from_pixels: bool) -> tuple[wpo_deps.BackboneSpec, wpo_deps.HeadSpec, wpo_deps.HeadSpec]:
    backbone_name = "nature_cnn" if from_pixels else config.backbone_name
    actor_backbone_spec = wpo_deps.BackboneSpec(
        name=backbone_name,
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=config.backbone_activation,
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = wpo_deps.HeadSpec(hidden_sizes=tuple(config.actor_head_hidden_sizes), activation=config.head_activation)
    critic_head_spec = wpo_deps.HeadSpec(hidden_sizes=tuple(config.critic_head_hidden_sizes), activation=config.head_activation)
    return (actor_backbone_spec, actor_head_spec, critic_head_spec)


def _build_actor(
    env: _EnvSetup,
    obs_scaler: nn.Module,
    actor_backbone_spec: wpo_deps.BackboneSpec,
    actor_head_spec: wpo_deps.HeadSpec,
) -> tuple[nn.Module, nn.Module, _ActorNet, wpo_deps.tr_modules.ProbabilisticActor]:
    actor_backbone, actor_feat_dim = wpo_deps.build_backbone(actor_backbone_spec, env.obs_dim)
    actor_head = wpo_deps.build_mlp_head(actor_head_spec, actor_feat_dim, 2 * env.act_dim)
    actor_net = _ActorNet(actor_backbone, actor_head, obs_scaler, env.act_dim)
    actor = wpo_deps.tr_modules.ProbabilisticActor(
        wpo_deps.td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        distribution_class=wpo_deps.tr_dists.TanhNormal,
        return_log_prob=True,
    )
    return (actor_backbone, actor_head, actor_net, actor)


def _build_q_pair(
    env: _EnvSetup,
    obs_scaler: nn.Module,
    actor_backbone_spec: wpo_deps.BackboneSpec,
    critic_head_spec: wpo_deps.HeadSpec,
    *,
    from_pixels: bool,
) -> tuple[nn.Module, nn.Module]:
    if from_pixels:
        q1_obs_encoder, q1_obs_dim = wpo_deps.build_backbone(actor_backbone_spec, env.obs_dim)
        q2_obs_encoder, q2_obs_dim = wpo_deps.build_backbone(actor_backbone_spec, env.obs_dim)
        q1_head = wpo_deps.build_mlp_head(critic_head_spec, q1_obs_dim + env.act_dim, 1)
        q2_head = wpo_deps.build_mlp_head(critic_head_spec, q2_obs_dim + env.act_dim, 1)
        q1 = _QNetPixel(q1_obs_encoder, q1_head, obs_scaler)
        q2 = _QNetPixel(q2_obs_encoder, q2_head, obs_scaler)
        return (q1, q2)
    critic_input_dim = env.obs_dim + env.act_dim
    q1_backbone, q1_feat_dim = wpo_deps.build_backbone(actor_backbone_spec, critic_input_dim)
    q2_backbone, q2_feat_dim = wpo_deps.build_backbone(actor_backbone_spec, critic_input_dim)
    q1_head = wpo_deps.build_mlp_head(critic_head_spec, q1_feat_dim, 1)
    q2_head = wpo_deps.build_mlp_head(critic_head_spec, q2_feat_dim, 1)
    q1 = _QNet(q1_backbone, q1_head, obs_scaler)
    q2 = _QNet(q2_backbone, q2_head, obs_scaler)
    return (q1, q2)


def build_modules(config: WPOConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    obs_scaler = wpo_deps.torchrl_common.ObsScaler(env.obs_lb, env.obs_width)
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
    actor_target = copy.deepcopy(actor_net).to(device).eval()
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)
    for p in actor_target.parameters():
        p.requires_grad_(False)
    dual_shape = (int(env.act_dim),) if bool(config.per_dim_constraining) else (1,)
    log_alpha_mean = nn.Parameter(torch.full(dual_shape, float(config.init_log_alpha_mean), dtype=torch.float32, device=device))
    log_alpha_stddev = nn.Parameter(torch.full(dual_shape, float(config.init_log_alpha_stddev), dtype=torch.float32, device=device))
    actor_param_count = sum((p.numel() for p in actor_backbone.parameters())) + sum((p.numel() for p in actor_head.parameters()))
    if config.theta_dim is not None:
        assert actor_param_count == int(config.theta_dim), (actor_param_count, config.theta_dim)
    return _Modules(
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        obs_scaler=obs_scaler,
        actor=actor,
        actor_model=actor_net,
        actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        log_alpha_mean=log_alpha_mean,
        log_alpha_stddev=log_alpha_stddev,
    )


def build_training(config: WPOConfig, modules: _Modules) -> _TrainingSetup:
    replay = wpo_deps.tr_data.TensorDictReplayBuffer(storage=wpo_deps.tr_data.LazyTensorStorage(int(config.replay_size)), batch_size=int(config.batch_size))
    actor_params = list(modules.actor_backbone.parameters()) + list(modules.actor_head.parameters())
    critic_params = list(modules.q1.parameters()) + list(modules.q2.parameters())
    dual_params = [modules.log_alpha_mean, modules.log_alpha_stddev]
    actor_optimizer = torch.optim.AdamW(actor_params, lr=float(config.learning_rate_actor), weight_decay=0.0)
    critic_optimizer = torch.optim.AdamW(critic_params, lr=float(config.learning_rate_critic), weight_decay=0.0)
    dual_optimizer = torch.optim.AdamW(dual_params, lr=float(config.learning_rate_dual), weight_decay=0.0)
    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    wpo_deps.write_config(str(exp_dir), config.to_dict())
    metrics_path = exp_dir / "metrics.jsonl"
    return _TrainingSetup(
        replay=replay,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        dual_optimizer=dual_optimizer,
        exp_dir=exp_dir,
        metrics_path=metrics_path,
        checkpoint_manager=wpo_deps.CheckpointManager(exp_dir=exp_dir),
    )


def wpo_update_shared(
    config: WPOConfig,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    obs: torch.Tensor,
    act: torch.Tensor,
    rew: torch.Tensor,
    nxt: torch.Tensor,
    done: torch.Tensor,
) -> tuple[float, float, float, float, float]:
    return wpo_update_step(
        modules=WPOUpdateModules(
            actor=modules.actor_model,
            actor_target=modules.actor_target,
            q1=modules.q1,
            q2=modules.q2,
            q1_target=modules.q1_target,
            q2_target=modules.q2_target,
            log_alpha_mean=modules.log_alpha_mean,
            log_alpha_stddev=modules.log_alpha_stddev,
        ),
        optimizers=WPOUpdateOptimizers(actor=training.actor_optimizer, critic=training.critic_optimizer, dual=training.dual_optimizer),
        batch=WPOUpdateBatch(obs=obs, act=act, rew=rew, nxt=nxt, done=done),
        hyper=WPOUpdateHyperParams(
            gamma=float(config.gamma),
            tau=float(config.tau),
            num_samples=int(config.num_samples),
            epsilon_mean=float(config.epsilon_mean),
            epsilon_stddev=float(config.epsilon_stddev),
            policy_loss_scale=float(config.policy_loss_scale),
            kl_loss_scale=float(config.kl_loss_scale),
            dual_loss_scale=float(config.dual_loss_scale),
            per_dim_constraining=bool(config.per_dim_constraining),
            squashing_type=str(config.squashing_type),
        ),
    )
