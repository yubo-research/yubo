from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import tensordict.nn as td_nn
import torch
import torch.nn as nn
import torchrl.data as tr_data
import torchrl.envs as tr_envs
import torchrl.envs.transforms as tr_transforms
import torchrl.modules as tr_modules
import torchrl.modules.distributions as tr_dists

from analysis.data_io import write_config
from rl.backbone import (
    ActorCriticSpec,
    BackboneSpec,
    HeadSpec,
    build_backbone,
    build_mlp_head,
)
from rl.checkpointing import CheckpointManager
from rl.core.continuous_actions import scale_action_to_env, unscale_action_from_env
from rl.core.envs import build_continuous_env_setup
from rl.core.runtime import ObsScaler, obs_scale_from_env
from rl.core.sac_update import (
    SACUpdateBatch,
    SACUpdateHyperParams,
    SACUpdateModules,
    SACUpdateOptimizers,
    sac_update_step,
)
from rl.env_provider import get_env_conf_fn
from rl.torchrl.dm_control_collect import make_dm_control_collect_env
from rl.torchrl.offpolicy import models

from .config import SACConfig


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
    obs_scaler: ObsScaler
    actor: tr_modules.ProbabilisticActor
    actor_model: nn.Module
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    log_alpha: nn.Parameter


@dataclasses.dataclass
class _TrainingSetup:
    replay: tr_data.TensorDictReplayBuffer
    actor_optimizer: torch.optim.AdamW
    critic_optimizer: torch.optim.AdamW
    alpha_optimizer: torch.optim.AdamW
    exp_dir: Path
    metrics_path: Path
    checkpoint_manager: CheckpointManager


@dataclasses.dataclass
class _TrainState:
    start_step: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None


_ActorNet = models.ActorNet
_QNet = models.QNet
_QNetPixel = models.QNetPixel


class _ScaleActionToEnv(nn.Module):
    def __init__(self, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.register_buffer("_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        width = (self._high - self._low).clamp(min=1e-08)
        return self._low + width * (1.0 + action) / 2.0


def _is_dm_control_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("dm_control/")


def _make_collect_env_dm_control_sac(env_conf, env_setup: _EnvSetup, env_index: int = 0):
    seed = int(env_setup.problem_seed) + env_index
    return make_dm_control_collect_env(
        env_name=str(env_conf.env_name),
        seed=int(seed),
        obs_mode=str(getattr(env_conf, "obs_mode", "vector")),
        tr_envs_module=tr_envs,
        tr_transforms_module=tr_transforms,
        pixels_transform_builder=lambda m: m.Compose(
            m.ToTensorImage(in_keys=["pixels"], out_keys=["observation"], from_int=True),
            m.Resize(84, 84, in_keys=["observation"]),
            m.DoubleToFloat(),
        ),
    )


def _make_collect_env_sac(env_conf, env_setup: _EnvSetup, env_index: int = 0):
    if _is_dm_control_env(env_conf):
        return _make_collect_env_dm_control_sac(env_conf, env_setup, env_index)
    base = env_conf.make()
    seed = int(env_setup.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), tr_transforms.DoubleToFloat())


def build_env_setup(config: SACConfig) -> _EnvSetup:
    shared = build_continuous_env_setup(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        obs_mode=str(getattr(config, "obs_mode", "vector")),
        get_env_conf_fn=get_env_conf_fn(),
        obs_scale_from_env_fn=obs_scale_from_env,
    )
    env_conf = shared.env_conf
    from_pixels = str(getattr(env_conf, "obs_mode", "vector")) in {
        "image",
        "mixed",
        "pixels",
        "pixels+state",
    }
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


def _build_arch(config: SACConfig, *, from_pixels: bool) -> ActorCriticSpec:
    actor_backbone_name = "nature_cnn" if from_pixels else str(config.backbone_name)
    critic_backbone_name = actor_backbone_name if getattr(config, "critic_backbone_name", None) is None else str(config.critic_backbone_name)
    return ActorCriticSpec.from_config(
        config,
        actor_backbone_name=actor_backbone_name,
        critic_backbone_name=critic_backbone_name,
    )


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
    critic_backbone_spec: BackboneSpec,
    critic_head_spec: HeadSpec,
    *,
    from_pixels: bool,
) -> tuple[nn.Module, nn.Module]:
    if from_pixels:
        q1_obs_encoder, q1_obs_dim = build_backbone(critic_backbone_spec, env.obs_dim)
        q2_obs_encoder, q2_obs_dim = build_backbone(critic_backbone_spec, env.obs_dim)
        q1_head = build_mlp_head(critic_head_spec, q1_obs_dim + env.act_dim, 1)
        q2_head = build_mlp_head(critic_head_spec, q2_obs_dim + env.act_dim, 1)
        q1 = _QNetPixel(q1_obs_encoder, q1_head, obs_scaler)
        q2 = _QNetPixel(q2_obs_encoder, q2_head, obs_scaler)
        return (q1, q2)
    else:
        critic_input_dim = env.obs_dim + env.act_dim
        q1_backbone, q1_feat_dim = build_backbone(critic_backbone_spec, critic_input_dim)
        q2_backbone, q2_feat_dim = build_backbone(critic_backbone_spec, critic_input_dim)
        q1_head = build_mlp_head(critic_head_spec, q1_feat_dim, 1)
        q2_head = build_mlp_head(critic_head_spec, q2_feat_dim, 1)
        q1 = _QNet(q1_backbone, q1_head, obs_scaler)
        q2 = _QNet(q2_backbone, q2_head, obs_scaler)
        return (q1, q2)


def build_modules(config: SACConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    obs_scaler = ObsScaler(env.obs_lb, env.obs_width)
    from_pixels = str(getattr(env.env_conf, "obs_mode", "vector")) in {
        "image",
        "mixed",
        "pixels",
        "pixels+state",
    }
    arch = _build_arch(config, from_pixels=from_pixels)
    actor_backbone, actor_head, actor_net, actor = _build_actor(env, obs_scaler, arch.actor.backbone, arch.actor.head)
    q1, q2 = _build_q_pair(env, obs_scaler, arch.critic.backbone, arch.critic.head, from_pixels=from_pixels)
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
