"""SAC setup: env creation, modules, training, and dataclasses."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchrl.envs as tr_envs

from . import deps as sac_deps
from .config import SACConfig


def _scale_action_to_env(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return action_low + (action_high - action_low) * (1.0 + action) / 2.0


def _unscale_action_from_env(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    width = np.maximum(action_high - action_low, 1e-8)
    scaled = 2.0 * (action - action_low) / width - 1.0
    return np.clip(scaled, -1.0, 1.0)


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
    obs_scaler: sac_deps.torchrl_common.ObsScaler
    actor: sac_deps.tr_modules.ProbabilisticActor


@dataclasses.dataclass
class _TrainingSetup:
    replay: sac_deps.tr_data.TensorDictReplayBuffer
    loss_module: sac_deps.tr_objectives.SACLoss
    target_updater: sac_deps.tr_objectives.SoftUpdate
    actor_optimizer: torch.optim.AdamW
    critic_optimizer: torch.optim.AdamW
    alpha_optimizer: torch.optim.AdamW
    exp_dir: Path
    metrics_path: Path
    checkpoint_manager: sac_deps.CheckpointManager


@dataclasses.dataclass
class _TrainState:
    start_step: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None


class _ActorNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        obs_scaler: sac_deps.torchrl_common.ObsScaler,
        act_dim: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler
        self.act_dim = int(act_dim)

    def forward(self, observation: torch.Tensor):
        obs = self.obs_scaler(observation)
        feats = self.backbone(obs)
        out = self.head(feats)
        loc, log_scale = out[..., : self.act_dim], out[..., self.act_dim :]
        scale = log_scale.clamp(-5.0, 2.0).exp()
        return loc, scale


class _QNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        obs_scaler: sac_deps.torchrl_common.ObsScaler,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        obs = self.obs_scaler(observation)
        x = torch.cat([obs, action], dim=-1)
        feats = self.backbone(x)
        return self.head(feats)


class _QNetPixel(nn.Module):
    """Q-net for pixel observations: CNN(obs) -> concat(latent, action) -> MLP -> Q."""

    def __init__(
        self,
        obs_encoder: nn.Module,
        head: nn.Module,
        obs_scaler: sac_deps.torchrl_common.ObsScaler,
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.head = head
        self.obs_scaler = obs_scaler

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        obs = self.obs_scaler(observation)
        latent = self.obs_encoder(obs)
        x = torch.cat([latent, action], dim=-1)
        return self.head(x)


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


def _make_collect_env_dm_control_sac(env_conf, env_setup: _EnvSetup, env_index: int = 0):
    """Create a seeded dm_control collect env using Shimmy + GymWrapper."""
    from problems.shimmy_dm_control import make as make_dm_env

    seed = int(env_setup.problem_seed) + env_index
    from_pixels = getattr(env_conf, "from_pixels", False)
    pixels_only = getattr(env_conf, "pixels_only", True)

    base = make_dm_env(
        env_conf.env_name,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
    )
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)

    if from_pixels:
        to_tensor = sac_deps.tr_transforms.ToTensorImage(in_keys=["pixels"], out_keys=["observation"], from_int=True)
        resize = sac_deps.tr_transforms.Resize(84, 84, in_keys=["observation"])
        transforms = sac_deps.tr_transforms.Compose(to_tensor, resize, sac_deps.tr_transforms.DoubleToFloat())
    else:
        transforms = sac_deps.tr_transforms.DoubleToFloat()
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), transforms)


def _make_collect_env_sac(env_conf, env_setup: _EnvSetup, env_index: int = 0):
    if _is_dm_control_env(env_conf):
        return _make_collect_env_dm_control_sac(env_conf, env_setup, env_index)
    base = env_conf.make()
    seed = int(env_setup.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), sac_deps.tr_transforms.DoubleToFloat())


def build_env_setup(config: SACConfig) -> _EnvSetup:
    from rl.seed_util import resolve_noise_seed_0, resolve_problem_seed

    problem_seed = resolve_problem_seed(seed=config.seed, problem_seed=config.problem_seed)
    noise_seed_0 = resolve_noise_seed_0(problem_seed=problem_seed, noise_seed_0=config.noise_seed_0)
    env_conf = sac_deps.get_env_conf(
        config.env_tag,
        problem_seed=problem_seed,
        noise_seed_0=noise_seed_0,
        from_pixels=getattr(config, "from_pixels", False),
        pixels_only=getattr(config, "pixels_only", True),
    )
    if env_conf.gym_conf is None:
        raise ValueError(f"SAC expects a gym env_tag, got {config.env_tag}")
    env_conf.ensure_spaces()
    from_pixels = getattr(env_conf, "from_pixels", False)
    obs_dim = 64 if from_pixels else int(env_conf.gym_conf.state_space.shape[0])
    act_dim = int(env_conf.action_space.shape[0])
    action_low = np.asarray(env_conf.action_space.low, dtype=np.float32)
    action_high = np.asarray(env_conf.action_space.high, dtype=np.float32)
    lb, width = sac_deps.torchrl_common.obs_scale_from_env(env_conf)
    return _EnvSetup(
        env_conf=env_conf,
        problem_seed=int(problem_seed),
        noise_seed_0=int(noise_seed_0),
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        obs_lb=lb,
        obs_width=width,
    )


def build_modules(config: SACConfig, env: _EnvSetup, *, device: torch.device) -> tuple[_Modules, sac_deps.tr_objectives.SACLoss]:
    obs_scaler = sac_deps.torchrl_common.ObsScaler(env.obs_lb, env.obs_width)
    from_pixels = getattr(env.env_conf, "from_pixels", False)
    backbone_name = "nature_cnn" if from_pixels else config.backbone_name
    actor_backbone_spec = sac_deps.BackboneSpec(
        name=backbone_name,
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=config.backbone_activation,
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = sac_deps.HeadSpec(
        hidden_sizes=tuple(config.actor_head_hidden_sizes),
        activation=config.head_activation,
    )
    critic_head_spec = sac_deps.HeadSpec(
        hidden_sizes=tuple(config.critic_head_hidden_sizes),
        activation=config.head_activation,
    )

    actor_backbone, actor_feat_dim = sac_deps.build_backbone(actor_backbone_spec, env.obs_dim)
    actor_head = sac_deps.build_mlp_head(actor_head_spec, actor_feat_dim, 2 * env.act_dim)
    actor_net = _ActorNet(actor_backbone, actor_head, obs_scaler, env.act_dim)
    actor = sac_deps.tr_modules.ProbabilisticActor(
        sac_deps.td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        distribution_class=sac_deps.tr_dists.TanhNormal,
        return_log_prob=True,
    )

    if from_pixels:
        q_obs_encoder, q_obs_dim = sac_deps.build_backbone(actor_backbone_spec, env.obs_dim)
        critic_input_dim = q_obs_dim + env.act_dim
        q_head = sac_deps.build_mlp_head(critic_head_spec, critic_input_dim, 1)
        q_net = _QNetPixel(q_obs_encoder, q_head, obs_scaler)
    else:
        critic_input_dim = env.obs_dim + env.act_dim
        q_backbone, q_feat_dim = sac_deps.build_backbone(actor_backbone_spec, critic_input_dim)
        q_head = sac_deps.build_mlp_head(critic_head_spec, q_feat_dim, 1)
        q_net = _QNet(q_backbone, q_head, obs_scaler)
    qvalue = sac_deps.td_nn.TensorDictModule(q_net, in_keys=["observation", "action"], out_keys=["state_action_value"])

    actor.to(device)
    qvalue.to(device)
    obs_scaler.to(device)

    actor_param_count = sum(p.numel() for p in actor_backbone.parameters()) + sum(p.numel() for p in actor_head.parameters())
    if config.theta_dim is not None:
        assert actor_param_count == int(config.theta_dim), (
            actor_param_count,
            config.theta_dim,
        )

    # SACLoss clones and manages target nets internally.
    loss_module = sac_deps.tr_objectives.SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        alpha_init=float(config.alpha_init),
        target_entropy=float(config.target_entropy) if config.target_entropy is not None else -float(env.act_dim),
    )
    loss_module.make_value_estimator(gamma=float(config.gamma))
    return _Modules(
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        obs_scaler=obs_scaler,
        actor=actor,
    ), loss_module


def build_training(config: SACConfig, loss_module: sac_deps.tr_objectives.SACLoss) -> _TrainingSetup:
    replay = sac_deps.tr_data.TensorDictReplayBuffer(
        storage=sac_deps.tr_data.LazyTensorStorage(int(config.replay_size)),
        batch_size=int(config.batch_size),
    )

    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    alpha_params = [loss_module.log_alpha]

    actor_optimizer = torch.optim.AdamW(actor_params, lr=float(config.learning_rate_actor), weight_decay=0.0)
    critic_optimizer = torch.optim.AdamW(critic_params, lr=float(config.learning_rate_critic), weight_decay=0.0)
    alpha_optimizer = torch.optim.AdamW(alpha_params, lr=float(config.learning_rate_alpha), weight_decay=0.0)

    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    sac_deps.write_config(str(exp_dir), config.to_dict())
    metrics_path = exp_dir / "metrics.jsonl"
    return _TrainingSetup(
        replay=replay,
        loss_module=loss_module,
        target_updater=sac_deps.tr_objectives.SoftUpdate(loss_module, tau=float(config.tau)),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        exp_dir=exp_dir,
        metrics_path=metrics_path,
        checkpoint_manager=sac_deps.CheckpointManager(exp_dir=exp_dir),
    )
