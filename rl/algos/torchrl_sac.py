from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SACLoss, SoftUpdate

from analysis.data_io import write_config
from common.seed_all import seed_all
from problems.env_conf import get_env_conf
from rl.algos.checkpointing import CheckpointManager, load_checkpoint
from rl.algos.torchrl_common import ObsScaler, obs_scale_from_env, select_device, temporary_distribution_validate_args
from rl.algos.torchrl_sac_actor_eval import (
    SacActorEvalPolicy,
    capture_sac_actor_snapshot,
    restore_sac_actor_snapshot,
)
from rl.algos.torchrl_sac_loop import (
    advance_env_and_store,
    checkpoint_if_due,
    evaluate_heldout_if_enabled,
    evaluate_if_due,
    log_if_due,
    run_updates_if_due,
    save_final_checkpoint_if_enabled,
    select_training_action,
)
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head


@dataclasses.dataclass
class SACConfig:
    exp_dir: str = "_tmp/sac"
    env_tag: str = "pend"
    seed: int = 1
    problem_seed: int | None = None
    noise_seed_0: int | None = None

    total_timesteps: int = 1000000
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    learning_rate_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    replay_size: int = 1000000
    learning_starts: int = 5000
    update_every: int = 1
    updates_per_step: int = 1
    alpha_init: float = 0.2
    target_entropy: float | None = None

    eval_interval_steps: int = 10000
    num_denoise_eval: int | None = None
    num_denoise_passive_eval: int | None = None
    eval_seed_base: int | None = None

    backbone_name: str = "mlp"
    backbone_hidden_sizes: tuple[int, ...] = (256, 256)
    backbone_activation: str = "silu"
    backbone_layer_norm: bool = False
    actor_head_hidden_sizes: tuple[int, ...] = ()
    critic_head_hidden_sizes: tuple[int, ...] = ()
    head_activation: str = "silu"
    theta_dim: int | None = None

    device: str = "auto"
    log_interval_steps: int = 1000
    checkpoint_interval_steps: int | None = None
    resume_from: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "SACConfig":
        data = dict(raw)
        for key in ["backbone_hidden_sizes", "actor_head_hidden_sizes", "critic_head_hidden_sizes"]:
            if key in data and data[key] is not None:
                data[key] = tuple(int(x) for x in data[key])
        return cls(**data)


@dataclasses.dataclass
class TrainResult:
    best_return: float
    last_eval_return: float
    last_heldout_return: float | None
    num_steps: int


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
    actor: ProbabilisticActor


@dataclasses.dataclass
class _TrainingSetup:
    replay: ReplayBuffer
    loss_module: SACLoss
    target_updater: SoftUpdate
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


class _ActorNet(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, obs_scaler: ObsScaler, act_dim: int):
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
    def __init__(self, backbone: nn.Module, head: nn.Module, obs_scaler: ObsScaler):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        obs = self.obs_scaler(observation)
        x = torch.cat([obs, action], dim=-1)
        feats = self.backbone(x)
        return self.head(feats)


def _build_env_setup(config: SACConfig) -> _EnvSetup:
    problem_seed = config.problem_seed if config.problem_seed is not None else config.seed
    noise_seed_0 = config.noise_seed_0 if config.noise_seed_0 is not None else 10 * problem_seed
    env_conf = get_env_conf(config.env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    if env_conf.gym_conf is None:
        raise ValueError(f"SAC expects a gym env_tag, got {config.env_tag}")
    env_conf.ensure_spaces()
    obs_dim = int(env_conf.gym_conf.state_space.shape[0])
    act_dim = int(env_conf.action_space.shape[0])
    action_low = np.asarray(env_conf.action_space.low, dtype=np.float32)
    action_high = np.asarray(env_conf.action_space.high, dtype=np.float32)
    lb, width = obs_scale_from_env(env_conf)
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


def _scale_action_to_env(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    return action_low + (action_high - action_low) * (1.0 + action) / 2.0


def _unscale_action_from_env(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    width = np.maximum(action_high - action_low, 1e-8)
    scaled = 2.0 * (action - action_low) / width - 1.0
    return np.clip(scaled, -1.0, 1.0)


def _build_modules(config: SACConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    obs_scaler = ObsScaler(env.obs_lb, env.obs_width)
    actor_backbone_spec = BackboneSpec(
        name=config.backbone_name,
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=config.backbone_activation,
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = HeadSpec(hidden_sizes=tuple(config.actor_head_hidden_sizes), activation=config.head_activation)
    critic_head_spec = HeadSpec(hidden_sizes=tuple(config.critic_head_hidden_sizes), activation=config.head_activation)

    actor_backbone, actor_feat_dim = build_backbone(actor_backbone_spec, env.obs_dim)
    actor_head = build_mlp_head(actor_head_spec, actor_feat_dim, 2 * env.act_dim)
    actor_net = _ActorNet(actor_backbone, actor_head, obs_scaler, env.act_dim)
    actor = ProbabilisticActor(
        TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )

    critic_input_dim = env.obs_dim + env.act_dim
    q_backbone, q_feat_dim = build_backbone(actor_backbone_spec, critic_input_dim)
    q_head = build_mlp_head(critic_head_spec, q_feat_dim, 1)
    q_net = _QNet(q_backbone, q_head, obs_scaler)
    qvalue = TensorDictModule(q_net, in_keys=["observation", "action"], out_keys=["state_action_value"])

    actor.to(device)
    qvalue.to(device)
    obs_scaler.to(device)

    actor_param_count = sum(p.numel() for p in actor_backbone.parameters()) + sum(p.numel() for p in actor_head.parameters())
    if config.theta_dim is not None:
        assert actor_param_count == int(config.theta_dim), (actor_param_count, config.theta_dim)

    # SACLoss clones and manages target nets internally.
    loss_module = SACLoss(
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


def _build_training(config: SACConfig, loss_module: SACLoss) -> _TrainingSetup:
    replay = ReplayBuffer(storage=LazyTensorStorage(int(config.replay_size)))

    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    alpha_params = [loss_module.log_alpha]

    actor_optimizer = torch.optim.AdamW(actor_params, lr=float(config.learning_rate_actor), weight_decay=0.0)
    critic_optimizer = torch.optim.AdamW(critic_params, lr=float(config.learning_rate_critic), weight_decay=0.0)
    alpha_optimizer = torch.optim.AdamW(alpha_params, lr=float(config.learning_rate_alpha), weight_decay=0.0)

    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), config.to_dict())
    metrics_path = exp_dir / "metrics.jsonl"
    return _TrainingSetup(
        replay=replay,
        loss_module=loss_module,
        target_updater=SoftUpdate(loss_module, tau=float(config.tau)),
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        exp_dir=exp_dir,
        metrics_path=metrics_path,
        checkpoint_manager=CheckpointManager(exp_dir=exp_dir),
    )


def _evaluate_actor(
    config: SACConfig,
    env: _EnvSetup,
    modules: _Modules,
    *,
    device: torch.device,
    eval_seed: int,
) -> float:
    from optimizer.opt_trajectories import collect_denoised_trajectory

    eval_env = get_env_conf(config.env_tag, problem_seed=env.problem_seed, noise_seed_0=env.noise_seed_0)
    eval_policy = SacActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        act_dim=env.act_dim,
        device=device,
    )
    traj, _ = collect_denoised_trajectory(
        eval_env,
        eval_policy,
        num_denoise=config.num_denoise_eval,
        i_noise=int(eval_seed),
    )
    return float(traj.rreturn)


def _checkpoint_payload(
    modules: _Modules,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    step: int,
) -> dict:
    return {
        "step": int(step),
        "actor_backbone": modules.actor_backbone.state_dict(),
        "actor_head": modules.actor_head.state_dict(),
        "obs_scaler": modules.obs_scaler.state_dict(),
        "replay_state": training.replay.state_dict(),
        "loss_module": training.loss_module.state_dict(),
        "actor_optimizer": training.actor_optimizer.state_dict(),
        "critic_optimizer": training.critic_optimizer.state_dict(),
        "alpha_optimizer": training.alpha_optimizer.state_dict(),
        "best_return": float(state.best_return),
        "best_actor_state": state.best_actor_state,
        "last_eval_return": float(state.last_eval_return),
        "last_heldout_return": state.last_heldout_return,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _resume_if_requested(
    config: SACConfig,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    device: torch.device,
) -> _TrainState:
    state = _TrainState()
    if not config.resume_from:
        return state
    loaded = load_checkpoint(Path(config.resume_from), device=device)
    modules.actor_backbone.load_state_dict(loaded["actor_backbone"])
    modules.actor_head.load_state_dict(loaded["actor_head"])
    if loaded.get("obs_scaler") is not None:
        modules.obs_scaler.load_state_dict(loaded["obs_scaler"])
    if loaded.get("replay_state") is not None:
        training.replay.load_state_dict(loaded["replay_state"])
    if loaded.get("loss_module") is not None:
        training.loss_module.load_state_dict(loaded["loss_module"])
    training.actor_optimizer.load_state_dict(loaded["actor_optimizer"])
    training.critic_optimizer.load_state_dict(loaded["critic_optimizer"])
    training.alpha_optimizer.load_state_dict(loaded["alpha_optimizer"])
    state.start_step = int(loaded.get("step", 0))
    state.best_return = float(loaded.get("best_return", state.best_return))
    state.best_actor_state = loaded.get("best_actor_state")
    state.last_eval_return = float(loaded.get("last_eval_return", state.last_eval_return))
    state.last_heldout_return = loaded.get("last_heldout_return", state.last_heldout_return)
    if "rng_torch" in loaded:
        torch.set_rng_state(loaded["rng_torch"])
    if "rng_numpy" in loaded:
        np.random.set_state(loaded["rng_numpy"])
    if torch.cuda.is_available() and loaded.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(loaded["rng_cuda"])
    return state


def _make_transition(
    state: np.ndarray,
    action_norm: np.ndarray,
    next_state: np.ndarray,
    reward: float,
    terminated: bool,
    done: bool,
) -> TensorDict:
    return TensorDict(
        {
            "observation": torch.as_tensor(state, dtype=torch.float32),
            "action": torch.as_tensor(action_norm, dtype=torch.float32),
            "next": TensorDict(
                {
                    "observation": torch.as_tensor(next_state, dtype=torch.float32),
                    "reward": torch.as_tensor([reward], dtype=torch.float32),
                    "done": torch.as_tensor([done], dtype=torch.bool),
                    "terminated": torch.as_tensor([terminated], dtype=torch.bool),
                },
                [],
            ),
        },
        [],
    )


def _update_step(training: _TrainingSetup, *, device: torch.device, batch_size: int) -> dict[str, float]:
    batch = training.replay.sample(batch_size).to(device)
    training.critic_optimizer.zero_grad()
    critic_loss = training.loss_module(batch)["loss_qvalue"]
    critic_loss.backward()
    training.critic_optimizer.step()

    training.actor_optimizer.zero_grad()
    actor_loss = training.loss_module(batch)["loss_actor"]
    actor_loss.backward()
    training.actor_optimizer.step()

    training.alpha_optimizer.zero_grad()
    alpha_out = training.loss_module(batch)
    alpha_loss = alpha_out["loss_alpha"]
    alpha_loss.backward()
    training.alpha_optimizer.step()

    training.target_updater.step()
    return {
        "loss_actor": float(actor_loss.detach().cpu()),
        "loss_critic": float(critic_loss.detach().cpu()),
        "loss_alpha": float(alpha_loss.detach().cpu()),
        "alpha": float(alpha_out["alpha"].detach().cpu()),
        "entropy": float(alpha_out["entropy"].detach().cpu()),
    }


def train_sac(config: SACConfig) -> TrainResult:
    # TorchRL's `ContinuousDistribution.support` currently calls
    # `torch.distributions.constraints.real()` which fails when `validate_args=True`.
    # Keep SAC stable regardless of global validation settings.
    with temporary_distribution_validate_args(False):
        from optimizer.opt_trajectories import evaluate_for_best

        seed_all(config.seed)
        env = _build_env_setup(config)
        device = select_device(config.device)
        modules, loss_module = _build_modules(config, env, device=device)
        training = _build_training(config, loss_module)
        state = _resume_if_requested(config, modules, training, device=device)

        print(
            "[rl/sac/torchrl]"
            f" env_tag={config.env_tag}"
            f" exp_dir={training.exp_dir}"
            f" seed={config.seed}"
            f" device={device.type}"
            f" obs_dim={env.obs_dim}"
            f" act_dim={env.act_dim}"
            f" total_timesteps={config.total_timesteps}"
            f" learning_starts={config.learning_starts}"
            f" batch_size={config.batch_size}"
            f" replay_size={config.replay_size}"
            f" update_every={config.update_every}"
            f" updates_per_step={config.updates_per_step}",
            flush=True,
        )

        train_env = env.env_conf.make()
        observation, _ = train_env.reset(seed=env.problem_seed)
        if hasattr(train_env, "action_space") and hasattr(train_env.action_space, "seed"):
            train_env.action_space.seed(env.problem_seed)
        start_time = time.time()

        latest_losses = {"loss_actor": float("nan"), "loss_critic": float("nan"), "loss_alpha": float("nan")}
        total_updates = 0

        for step in range(state.start_step + 1, int(config.total_timesteps) + 1):
            action_env, action_norm = select_training_action(
                config,
                env,
                modules,
                step=step,
                observation=observation,
                train_env=train_env,
                device=device,
                unscale_action_from_env=_unscale_action_from_env,
                scale_action_to_env=_scale_action_to_env,
            )
            observation = advance_env_and_store(
                training,
                train_env=train_env,
                observation=observation,
                action_env=action_env,
                action_norm=action_norm,
                make_transition=_make_transition,
            )
            latest_losses, total_updates = run_updates_if_due(
                config,
                training,
                step=step,
                device=device,
                latest_losses=latest_losses,
                total_updates=total_updates,
                update_step=_update_step,
            )
            evaluate_if_due(
                config,
                env,
                modules,
                training,
                state,
                step=step,
                device=device,
                start_time=start_time,
                latest_losses=latest_losses,
                total_updates=total_updates,
                evaluate_actor=_evaluate_actor,
                capture_actor_state=capture_sac_actor_snapshot,
                evaluate_heldout=lambda cfg, env_setup, local_modules, local_state, *, device: evaluate_heldout_if_enabled(
                    cfg,
                    env_setup,
                    local_modules,
                    local_state,
                    device=device,
                    capture_actor_state=capture_sac_actor_snapshot,
                    restore_actor_state=restore_sac_actor_snapshot,
                    eval_policy_factory=lambda actor_modules, actor_env_setup, actor_device: SacActorEvalPolicy(
                        actor_modules.actor_backbone,
                        actor_modules.actor_head,
                        actor_modules.obs_scaler,
                        act_dim=actor_env_setup.act_dim,
                        device=actor_device,
                    ),
                    get_env_conf=get_env_conf,
                    evaluate_for_best=evaluate_for_best,
                ),
            )
            log_if_due(
                config,
                state,
                step=step,
                start_time=start_time,
                latest_losses=latest_losses,
                total_updates=total_updates,
            )
            checkpoint_if_due(
                config,
                modules,
                training,
                state,
                step=step,
                build_checkpoint_payload=_checkpoint_payload,
            )

        train_env.close()
        save_final_checkpoint_if_enabled(
            config,
            modules,
            training,
            state,
            build_checkpoint_payload=_checkpoint_payload,
        )

        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_steps=int(config.total_timesteps),
        )


def register():
    from rl.algos.registry import register_algo

    register_algo("sac", SACConfig, train_sac)


__all__ = ["SACConfig", "TrainResult", "register", "train_sac"]
