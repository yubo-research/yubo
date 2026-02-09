from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.envs import GymWrapper, SerialEnv
from torchrl.envs.libs.gym import set_gym_backend
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from analysis.data_io import write_config
from common.seed_all import seed_all
from optimizer.opt_trajectories import collect_denoised_trajectory, evaluate_for_best
from problems.env_conf import get_env_conf
from rl.algos.torchrl_actor_eval import (
    ActorEvalPolicy,
    capture_actor_snapshot,
    restore_actor_snapshot,
    use_actor_snapshot,
)
from rl.algos.checkpointing import CheckpointManager, append_jsonl, load_checkpoint
from rl.algos.torchrl_checkpoint_io import save_final_checkpoint, save_periodic_checkpoint
from rl.algos.torchrl_common import (
    ObsScaler,
    collector_device_kwargs,
    obs_scale_from_env,
    select_device,
)
from rl.algos.torchrl_video import render_best_policy_videos
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head


@dataclass
class PPOConfig:
    exp_dir: str = "_tmp/ppo"
    env_tag: str = "pend"
    seed: int = 1
    problem_seed: Optional[int] = None
    noise_seed_0: Optional[int] = None

    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    eval_interval: int = 1
    num_denoise_eval: Optional[int] = None
    num_denoise_passive_eval: Optional[int] = None
    eval_seed_base: Optional[int] = None

    backbone_name: str = "mlp"
    backbone_hidden_sizes: tuple[int, ...] = (64, 64)
    backbone_activation: str = "silu"
    backbone_layer_norm: bool = True
    actor_head_hidden_sizes: tuple[int, ...] = ()
    critic_head_hidden_sizes: tuple[int, ...] = ()
    head_activation: str = "silu"
    share_backbone: bool = True
    log_std_init: float = 0.0

    theta_dim: Optional[int] = None
    device: str = "auto"
    log_interval: int = 1
    checkpoint_interval: Optional[int] = None
    resume_from: Optional[str] = None
    video_enable: bool = False
    video_dir: Optional[str] = None
    video_prefix: str = "policy"
    video_num_episodes: int = 10
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "PPOConfig":
        d = dict(raw)
        for key in [
            "backbone_hidden_sizes",
            "actor_head_hidden_sizes",
            "critic_head_hidden_sizes",
        ]:
            if key in d and d[key] is not None:
                d[key] = tuple(int(x) for x in d[key])
        return cls(**d)


@dataclass
class TrainResult:
    best_return: float
    last_eval_return: float
    last_heldout_return: Optional[float]
    num_iterations: int


@dataclass(frozen=True)
class _EnvSetup:
    env_conf: Any
    problem_seed: int
    noise_seed_0: int
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_lb: np.ndarray | None
    obs_width: np.ndarray | None


@dataclass
class _Modules:
    actor_backbone: nn.Module
    actor_head: nn.Module
    critic_backbone: nn.Module
    critic_head: nn.Module
    log_std: nn.Parameter
    obs_scaler: ObsScaler
    actor: ProbabilisticActor
    critic: TensorDictModule


@dataclass
class _TrainingSetup:
    frames_per_batch: int
    num_iterations: int
    env: SerialEnv
    loss_module: ClipPPOLoss
    gae: GAE
    train_params: list[torch.nn.Parameter]
    optimizer: optim.Adam
    exp_dir: Path
    metrics_path: Path
    checkpoint_manager: CheckpointManager


@dataclass
class _TrainState:
    start_iteration: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None


class _TanhNormal(TanhNormal):
    @property
    def support(self):
        return torch.distributions.constraints.real


class _ActorNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        log_std: torch.nn.Parameter,
        obs_scaler: ObsScaler,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.log_std = log_std
        self.obs_scaler = obs_scaler

    def forward(self, obs: torch.Tensor):
        obs = self.obs_scaler(obs)
        feats = self.backbone(obs)
        loc = self.head(feats)
        scale = self.log_std.exp().expand_as(loc)
        return loc, scale


class _CriticNet(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, obs_scaler: ObsScaler):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.obs_scaler = obs_scaler

    def forward(self, obs: torch.Tensor):
        obs = self.obs_scaler(obs)
        feats = self.backbone(obs)
        return self.head(feats)


def _count_unique_params(*modules: nn.Module, extra_params: list[torch.nn.Parameter] | None = None) -> int:
    unique = {}
    for module in modules:
        for p in module.parameters():
            unique[id(p)] = p
    if extra_params:
        for p in extra_params:
            unique[id(p)] = p
    return sum(p.numel() for p in unique.values())


def _unique_param_list(*modules: nn.Module, extra_params: list[torch.nn.Parameter] | None = None) -> list[torch.nn.Parameter]:
    unique = {}
    for module in modules:
        for p in module.parameters():
            unique[id(p)] = p
    if extra_params:
        for p in extra_params:
            unique[id(p)] = p
    return list(unique.values())


def _build_env_setup(config: PPOConfig) -> _EnvSetup:
    problem_seed = config.problem_seed if config.problem_seed is not None else config.seed
    noise_seed_0 = config.noise_seed_0 if config.noise_seed_0 is not None else 10 * problem_seed
    env_conf = get_env_conf(config.env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    if env_conf.gym_conf is None:
        raise ValueError(f"PPO expects a gym env_tag, got {config.env_tag}")
    env_conf.ensure_spaces()

    obs_dim = int(env_conf.gym_conf.state_space.shape[0])
    act_dim = int(env_conf.action_space.shape[0])
    lb, width = obs_scale_from_env(env_conf)
    action_low = np.asarray(env_conf.action_space.low, dtype=np.float32)
    action_high = np.asarray(env_conf.action_space.high, dtype=np.float32)
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


def _build_modules(config: PPOConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    backbone_spec = BackboneSpec(
        name=config.backbone_name,
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=config.backbone_activation,
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = HeadSpec(hidden_sizes=tuple(config.actor_head_hidden_sizes), activation=config.head_activation)
    critic_head_spec = HeadSpec(hidden_sizes=tuple(config.critic_head_hidden_sizes), activation=config.head_activation)

    if config.share_backbone:
        shared_backbone, feat_dim = build_backbone(backbone_spec, env.obs_dim)
        actor_backbone = shared_backbone
        critic_backbone = shared_backbone
        actor_feat_dim = feat_dim
        critic_feat_dim = feat_dim
    else:
        actor_backbone, actor_feat_dim = build_backbone(backbone_spec, env.obs_dim)
        critic_backbone, critic_feat_dim = build_backbone(backbone_spec, env.obs_dim)

    actor_head = build_mlp_head(actor_head_spec, actor_feat_dim, env.act_dim)
    critic_head = build_mlp_head(critic_head_spec, critic_feat_dim, 1)
    log_std = nn.Parameter(torch.full((env.act_dim,), float(config.log_std_init)))

    obs_scaler = ObsScaler(env.obs_lb, env.obs_width)
    actor_net = _ActorNet(actor_backbone, actor_head, log_std, obs_scaler)
    critic_net = _CriticNet(critic_backbone, critic_head, obs_scaler)

    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        actor_module,
        in_keys=["loc", "scale"],
        distribution_class=_TanhNormal,
        distribution_kwargs={"low": env.action_low, "high": env.action_high},
        return_log_prob=True,
    )
    critic = TensorDictModule(critic_net, in_keys=["observation"], out_keys=["state_value"])

    actor.to(device)
    critic.to(device)
    obs_scaler.to(device)

    actor_param_count = _count_unique_params(actor_backbone, actor_head, extra_params=[log_std])
    if config.theta_dim is not None:
        assert actor_param_count == int(config.theta_dim), (actor_param_count, config.theta_dim)
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


def _build_training(config: PPOConfig, env: _EnvSetup, modules: _Modules) -> _TrainingSetup:
    frames_per_batch = int(config.num_envs * config.num_steps)
    num_iterations = int(config.total_timesteps // frames_per_batch)
    if num_iterations <= 0:
        raise ValueError("total_timesteps too small for num_envs * num_steps.")

    env_conf = env.env_conf
    vec_env = SerialEnv(
        int(config.num_envs),
        lambda: GymWrapper(env_conf.make()),
        serial_for_single=True,
    )

    loss_module = ClipPPOLoss(
        modules.actor,
        modules.critic,
        clip_epsilon=config.clip_coef,
        entropy_coeff=config.ent_coef,
        critic_coeff=config.vf_coef,
        normalize_advantage=config.norm_adv,
        clip_value=config.clip_vloss,
        functional=False,
    )
    gae = GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=modules.critic,
    )

    train_params = _unique_param_list(modules.actor, modules.critic, extra_params=[modules.log_std])
    optimizer = optim.Adam(train_params, lr=config.learning_rate, eps=1e-5)

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
        checkpoint_manager=CheckpointManager(exp_dir=exp_dir),
    )


def _resume_if_requested(
    config: PPOConfig,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    device: torch.device,
) -> _TrainState:
    state = _TrainState()
    if not config.resume_from:
        return state
    resume_path = Path(config.resume_from)
    loaded = load_checkpoint(resume_path, device=device)
    modules.actor_backbone.load_state_dict(loaded["actor_backbone"])
    modules.actor_head.load_state_dict(loaded["actor_head"])
    modules.critic_backbone.load_state_dict(loaded["critic_backbone"])
    modules.critic_head.load_state_dict(loaded["critic_head"])
    modules.log_std.data.copy_(loaded["log_std"].to(device))
    if loaded.get("obs_scaler") is not None:
        modules.obs_scaler.load_state_dict(loaded["obs_scaler"])
    training.optimizer.load_state_dict(loaded["optimizer"])
    state.start_iteration = int(loaded.get("iteration", 0))
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


def _anneal_lr(config: PPOConfig, optimizer: optim.Optimizer, *, iteration: int, num_iterations: int) -> None:
    if not config.anneal_lr:
        return
    frac = 1.0 - (float(iteration) - 1.0) / float(num_iterations)
    lr_now = float(frac) * float(config.learning_rate)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_now


def _ppo_update(
    config: PPOConfig,
    training: _TrainingSetup,
    *,
    batch,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    batch = batch.to(device)
    training.gae(batch)
    flat = batch.reshape(-1)
    batch_size = int(flat.batch_size[0])
    minibatch_size = int(batch_size // config.num_minibatches)
    if minibatch_size <= 0:
        raise ValueError("num_minibatches too large for batch_size.")
    clipfracs: list[float] = []
    approx_kls: list[float] = []
    for _ in range(int(config.update_epochs)):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            mb = flat[mb_idx]
            loss_td = training.loss_module(mb)
            loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]
            training.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(training.train_params, config.max_grad_norm)
            training.optimizer.step()
            if "clip_fraction" in loss_td.keys():
                clipfracs.append(float(loss_td["clip_fraction"]))
            if "kl_approx" in loss_td.keys():
                approx_kls.append(float(loss_td["kl_approx"]))
        if config.target_kl is not None and approx_kls and float(np.mean(approx_kls)) > float(config.target_kl):
            break
    return approx_kls, clipfracs


def _evaluate_actor(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    *,
    device: torch.device,
    eval_seed: int,
) -> float:
    eval_env = get_env_conf(config.env_tag, problem_seed=env.problem_seed, noise_seed_0=env.noise_seed_0)
    eval_policy = ActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        device=device,
    )
    traj, _ = collect_denoised_trajectory(
        eval_env,
        eval_policy,
        num_denoise=config.num_denoise_eval,
        i_noise=int(eval_seed),
    )
    return float(traj.rreturn)


def _maybe_eval_and_log(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    iteration: int,
    approx_kls: list[float],
    clipfracs: list[float],
    device: torch.device,
    start_time: float,
) -> None:
    do_eval = bool(config.eval_interval) and iteration % int(config.eval_interval) == 0
    do_log = bool(config.log_interval) and iteration % int(config.log_interval) == 0
    if not do_eval:
        if do_log:
            elapsed = time.time() - start_time
            global_step = iteration * training.frames_per_batch
            sps = float(global_step / elapsed) if elapsed > 0 else float("nan")
            print(
                f"[rl/ppo/torchrl] iter={iteration:04d}/{training.num_iterations} step={global_step} time={elapsed:.1f}s sps={sps:.0f}",
                flush=True,
            )
        return

    eval_seed = int(config.eval_seed_base if config.eval_seed_base is not None else config.seed)
    state.last_eval_return = _evaluate_actor(config, env, modules, device=device, eval_seed=eval_seed)
    if state.last_eval_return > state.best_return:
        state.best_return = float(state.last_eval_return)
        state.best_actor_state = capture_actor_snapshot(modules)

    state.last_heldout_return = None
    if config.num_denoise_passive_eval is not None and state.best_actor_state is not None:
        current = capture_actor_snapshot(modules)
        restore_actor_snapshot(modules, state.best_actor_state, device=device)
        best_eval_policy = ActorEvalPolicy(modules.actor_backbone, modules.actor_head, modules.obs_scaler, device=device)
        state.last_heldout_return = float(
            evaluate_for_best(
                get_env_conf(config.env_tag, problem_seed=env.problem_seed, noise_seed_0=env.noise_seed_0),
                best_eval_policy,
                config.num_denoise_passive_eval,
            )
        )
        restore_actor_snapshot(modules, current, device=device)

    elapsed = time.time() - start_time
    global_step = iteration * training.frames_per_batch
    sps = float(global_step / elapsed) if elapsed > 0 else float("nan")
    record = {
        "iteration": iteration,
        "global_step": global_step,
        "eval_return": state.last_eval_return,
        "heldout_return": state.last_heldout_return,
        "best_return": state.best_return,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else None,
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else None,
        "time_seconds": elapsed,
        "steps_per_second": sps,
    }
    append_jsonl(training.metrics_path, record)
    if do_log:
        heldout_str = "nan" if state.last_heldout_return is None else f"{state.last_heldout_return:.2f}"
        print(
            "[rl/ppo/torchrl]"
            f" iter={iteration:04d}/{training.num_iterations}"
            f" step={global_step}"
            f" eval={state.last_eval_return:.2f}"
            f" heldout={heldout_str}"
            f" best={state.best_return:.2f}"
            f" kl={record['approx_kl'] if record['approx_kl'] is not None else float('nan'):.4f}"
            f" time={elapsed:.1f}s"
            f" sps={sps:.0f}",
            flush=True,
        )


def train_ppo(config: PPOConfig) -> TrainResult:
    set_gym_backend("gymnasium")
    seed_all(config.seed)
    env = _build_env_setup(config)
    device = select_device(config.device)
    modules = _build_modules(config, env, device=device)
    training = _build_training(config, env, modules)
    state = _resume_if_requested(config, modules, training, device=device)

    print(
        "[rl/ppo/torchrl]"
        f" env_tag={config.env_tag}"
        f" exp_dir={training.exp_dir}"
        f" seed={config.seed}"
        f" device={device.type}"
        f" obs_dim={env.obs_dim}"
        f" act_dim={env.act_dim}"
        f" total_timesteps={config.total_timesteps}"
        f" num_envs={config.num_envs}"
        f" num_steps={config.num_steps}"
        f" frames_per_batch={training.frames_per_batch}"
        f" num_iterations={training.num_iterations}"
        f" update_epochs={config.update_epochs}"
        f" eval_interval={config.eval_interval}"
        f" backbone={config.backbone_name}"
        f" hidden={list(config.backbone_hidden_sizes)}"
        f" act={config.backbone_activation}"
        f" ln={bool(config.backbone_layer_norm)}"
        f" share_backbone={bool(config.share_backbone)}",
        flush=True,
    )
    print(
        "[rl/ppo/torchrl]"
        f" actor_head={list(config.actor_head_hidden_sizes)}"
        f" critic_head={list(config.critic_head_hidden_sizes)}"
        f" log_std_init={config.log_std_init}"
        f" lr={config.learning_rate}"
        f" gamma={config.gamma}"
        f" gae_lambda={config.gae_lambda}"
        f" clip_coef={config.clip_coef}"
        f" vf_coef={config.vf_coef}"
        f" ent_coef={config.ent_coef}",
        flush=True,
    )

    remaining_iterations = max(0, training.num_iterations - state.start_iteration)
    collector = Collector(
        training.env,
        modules.actor,
        frames_per_batch=training.frames_per_batch,
        total_frames=int(remaining_iterations * training.frames_per_batch),
        **collector_device_kwargs(device),
    )
    if remaining_iterations == 0:
        collector.shutdown()
        render_best_policy_videos(
            config=config,
            env_setup=env,
            modules=modules,
            training_setup=training,
            train_state=state,
            device=device,
            build_eval_env_conf=lambda problem_seed, noise_seed_0: get_env_conf(
                config.env_tag,
                problem_seed=problem_seed,
                noise_seed_0=noise_seed_0,
            ),
            eval_policy_factory=lambda local_modules, local_device: ActorEvalPolicy(
                local_modules.actor_backbone,
                local_modules.actor_head,
                local_modules.obs_scaler,
                device=local_device,
            ),
            capture_actor_state=capture_actor_snapshot,
            temporary_actor_state=use_actor_snapshot,
        )
        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_iterations=training.num_iterations,
        )

    start_time = time.time()
    for iteration, batch in enumerate(collector, start=state.start_iteration + 1):
        if iteration > training.num_iterations:
            break
        _anneal_lr(config, training.optimizer, iteration=iteration, num_iterations=training.num_iterations)
        approx_kls, clipfracs = _ppo_update(config, training, batch=batch, device=device)
        _maybe_eval_and_log(
            config,
            env,
            modules,
            training,
            state,
            iteration=iteration,
            approx_kls=approx_kls,
            clipfracs=clipfracs,
            device=device,
            start_time=start_time,
        )
        save_periodic_checkpoint(
            config=config,
            training_setup=training,
            modules=modules,
            train_state=state,
            iteration=iteration,
        )

    collector.shutdown()
    save_final_checkpoint(
        config=config,
        training_setup=training,
        modules=modules,
        train_state=state,
    )
    render_best_policy_videos(
        config=config,
        env_setup=env,
        modules=modules,
        training_setup=training,
        train_state=state,
        device=device,
        build_eval_env_conf=lambda problem_seed, noise_seed_0: get_env_conf(
            config.env_tag,
            problem_seed=problem_seed,
            noise_seed_0=noise_seed_0,
        ),
        eval_policy_factory=lambda local_modules, local_device: ActorEvalPolicy(
            local_modules.actor_backbone,
            local_modules.actor_head,
            local_modules.obs_scaler,
            device=local_device,
        ),
        capture_actor_state=capture_actor_snapshot,
        temporary_actor_state=use_actor_snapshot,
    )
    return TrainResult(
        best_return=float(state.best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=state.last_heldout_return,
        num_iterations=training.num_iterations,
    )


def register():
    from rl.algos.registry import register_algo

    register_algo("ppo", PPOConfig, train_ppo)
