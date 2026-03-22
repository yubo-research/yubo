from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensordict.nn as td_nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchrl.collectors as tr_collectors
import torchrl.envs as tr_envs
import torchrl.envs.transforms as tr_transforms
import torchrl.modules as tr_modules
import torchrl.modules.distributions as tr_dists
import torchrl.objectives as tr_objectives

import rl.backbone as backbone
import rl.checkpointing as rl_checkpointing
import rl.core.env_conf as seed_util
import rl.registry as registry
from analysis.data_io import write_config
from common.seed_all import seed_all
from problems.problem import build_problem
from rl.core import env_conf as core_env_conf
from rl.core import env_contract as torchrl_env_contract
from rl.core import episode_rollout
from rl.core import runtime as torchrl_common
from rl.core import torchrl_runtime as torchrl_runtime
from rl.core.pixel_transform import AtariObservationTransform, PixelsToObservation
from rl.core.ppo_eval import evaluate_heldout_with_best_actor as _eval_heldout_with_best_actor
from rl.core.ppo_eval import update_best_actor_if_improved as _update_best_actor_if_improved
from rl.core.ppo_metrics import build_eval_record as _build_eval_record
from rl.core.profiler import run_with_profiler
from rl.eval_noise import build_eval_plan, normalize_eval_noise_mode

from . import actor_eval as torchrl_actor_eval
from . import models as op_models
from .actor_eval import capture_actor_snapshot as _capture_actor_state
from .actor_eval import restore_actor_snapshot as _restore_actor_state
from .checkpoint_io import save_final_checkpoint, save_periodic_checkpoint
from .config import _PPO_RUNTIME_CAPABILITIES, PPOConfig, TrainResult

__all__ = ["PPOConfig", "TrainResult", "_TanhNormal", "_capture_actor_state", "_restore_actor_state", "register", "torch", "train_ppo"]


def _build_env_runtime(env_tag, *, problem_seed=None, noise_seed_0=None, from_pixels=False, pixels_only=True):
    """Wrapper that adapts build_problem to return EnvironmentRuntime for callback-based APIs."""
    return build_problem(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0, from_pixels=from_pixels, pixels_only=pixels_only).env


@dataclasses.dataclass(frozen=True)
class _EnvSetup:
    env_conf: object
    io_contract: torchrl_env_contract.EnvIOContract
    problem_seed: int
    noise_seed_0: int
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    obs_lb: np.ndarray | None
    obs_width: np.ndarray | None
    is_discrete: bool = False


@dataclasses.dataclass
class _Modules:
    actor_backbone: nn.Module
    actor_head: nn.Module
    critic_backbone: nn.Module
    critic_head: nn.Module
    log_std: nn.Parameter | None
    obs_scaler: Any
    actor: Any
    critic: td_nn.TensorDictModule


@dataclasses.dataclass
class _TrainingSetup:
    frames_per_batch: int
    num_iterations: int
    env: object | None
    loss_module: Any
    gae: Any
    train_params: list[torch.nn.Parameter]
    optimizer: optim.AdamW
    exp_dir: Path
    metrics_path: Path
    checkpoint_manager: Any


@dataclasses.dataclass
class _TrainState:
    start_iteration: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None


def _tanh_normal_base():
    return tr_dists.TanhNormal


class _TanhNormal(_tanh_normal_base()):
    @property
    def support(self):
        return torch.distributions.constraints.real


_prepare_obs_for_backbone = op_models.prepare_obs_for_backbone
_ActorNet = op_models.ActorNet
_DiscreteActorNet = op_models.DiscreteActorNet
_CriticNet = op_models.CriticNet


def _count_unique_params(*modules: nn.Module, extra_params: list[torch.nn.Parameter] | None = None) -> int:
    unique = {}
    for module in modules:
        for p in module.parameters():
            unique[id(p)] = p
    if extra_params:
        for p in extra_params:
            unique[id(p)] = p
    return sum((p.numel() for p in unique.values()))


def _unique_param_list(*modules: nn.Module, extra_params: list[torch.nn.Parameter] | None = None) -> list[torch.nn.Parameter]:
    unique = {}
    for module in modules:
        for p in module.parameters():
            unique[id(p)] = p
    if extra_params:
        for p in extra_params:
            unique[id(p)] = p
    return list(unique.values())


def _is_due(step: int, interval: int | None) -> bool:
    return interval is not None and int(interval) > 0 and (int(step) % int(interval) == 0)


def _is_dm_control_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("dm_control/")


def _is_atari_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("ALE/")


def _resolve_observation_contract_for_env(config: PPOConfig, env: _EnvSetup | object) -> torchrl_env_contract.ObservationContract:
    io_contract = getattr(env, "io_contract", None)
    observation = getattr(io_contract, "observation", None)
    if observation is not None:
        return observation
    env_conf = getattr(env, "env_conf", None)
    if env_conf is not None:
        return torchrl_env_contract.resolve_observation_contract(env_conf, default_image_size=84)
    if bool(getattr(config, "from_pixels", False)):
        return torchrl_env_contract.ObservationContract(mode="pixels", raw_shape=(), model_channels=3, image_size=84)
    return torchrl_env_contract.ObservationContract(mode="vector", raw_shape=(), vector_dim=1)


def _make_collect_env_atari(env_conf, env_index: int = 0):
    # env_conf.make() calls _get_atari_dm() which loads env_conf_atari_dm; no explicit import
    # to avoid kiss cycle (core -> env_conf_atari_dm -> env_conf -> policy_backbone -> core).
    base = env_conf.make()
    seed = int(env_conf.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    transforms = tr_transforms.Compose(AtariObservationTransform(size=84), tr_transforms.DoubleToFloat())
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), transforms)


def _make_collect_env_dm_control(env_conf, env_index: int = 0):
    from problems.shimmy_dm_control import make as make_dm_env

    seed = int(env_conf.problem_seed) + env_index
    from_pixels = bool(getattr(env_conf, "from_pixels", False))
    pixels_only = bool(getattr(env_conf, "pixels_only", True))
    base = make_dm_env(env_conf.env_name, from_pixels=from_pixels, pixels_only=pixels_only)
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    if from_pixels:
        transforms = tr_transforms.Compose(PixelsToObservation(size=84), tr_transforms.DoubleToFloat())
    else:
        transforms = tr_transforms.DoubleToFloat()
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), transforms)


def _make_collect_env(env_conf, env_index: int = 0):
    if _is_dm_control_env(env_conf):
        return _make_collect_env_dm_control(env_conf, env_index)
    if _is_atari_env(env_conf):
        return _make_collect_env_atari(env_conf, env_index)
    base = env_conf.make()
    seed = int(env_conf.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), tr_transforms.DoubleToFloat())


def _make_collect_env_factory(env_conf, num_envs: int):
    env_index = [0]

    def fn():
        idx = env_index[0]
        env_index[0] = (idx + 1) % num_envs
        return _make_collect_env(env_conf, env_index=idx)

    return fn


def build_env_setup(config: PPOConfig) -> _EnvSetup:
    resolved = core_env_conf.build_seeded_env_conf_from_run(
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        problem_seed=config.problem_seed,
        noise_seed_0=config.noise_seed_0,
        from_pixels=bool(getattr(config, "from_pixels", False)),
        pixels_only=bool(getattr(config, "pixels_only", True)),
        get_env_conf_fn=_build_env_runtime,
    )
    env_conf = resolved.env_conf
    env_conf.ensure_spaces()
    io_contract = torchrl_env_contract.resolve_env_io_contract(env_conf, default_image_size=84)
    obs_dim = 64 if io_contract.observation.mode == "pixels" else int(io_contract.observation.vector_dim or 1)
    act_dim = int(io_contract.action.dim)
    action_low = io_contract.action.low
    action_high = io_contract.action.high
    lb, width = torchrl_common.obs_scale_from_env(env_conf)
    return _EnvSetup(
        env_conf=env_conf,
        io_contract=io_contract,
        problem_seed=int(resolved.problem_seed),
        noise_seed_0=int(resolved.noise_seed_0),
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        obs_lb=lb,
        obs_width=width,
        is_discrete=io_contract.action.kind == "discrete",
    )


def _build_seeded_eval_env_conf(config: PPOConfig, *, problem_seed: int, noise_seed_0: int, from_pixels: bool):
    resolved = core_env_conf.build_seeded_env_conf(
        env_tag=str(config.env_tag),
        problem_seed=int(problem_seed),
        noise_seed_0=int(noise_seed_0),
        from_pixels=bool(from_pixels),
        pixels_only=bool(getattr(config, "pixels_only", True)),
        get_env_conf_fn=_build_env_runtime,
    )
    return resolved.env_conf


def _build_eval_env_conf(config: PPOConfig, env: _EnvSetup, *, from_pixels: bool):
    return _build_seeded_eval_env_conf(
        config,
        problem_seed=int(env.problem_seed),
        noise_seed_0=int(env.noise_seed_0),
        from_pixels=from_pixels,
    )


def _make_video_context(config: PPOConfig, env: _EnvSetup, *, from_pixels: bool):
    video = __import__("common.video", fromlist=["RLVideoContext", "render_policy_videos_rl"])
    ctx = video.RLVideoContext(
        build_eval_env_conf=lambda ps, ns: _build_seeded_eval_env_conf(
            config,
            problem_seed=int(ps),
            noise_seed_0=int(ns),
            from_pixels=bool(from_pixels),
            pixels_only=bool(getattr(config, "pixels_only", True)),
            get_env_conf_fn=_build_env_runtime,
        ).env_conf,
        make_eval_policy=lambda m, d: torchrl_actor_eval.ActorEvalPolicy(
            m.actor_backbone,
            m.actor_head,
            m.obs_scaler,
            device=d,
            obs_contract=env.io_contract.observation,
            is_discrete=bool(getattr(env, "is_discrete", False)),
        ),
        capture_actor_state=torchrl_actor_eval.capture_actor_snapshot,
        with_actor_state=torchrl_actor_eval.use_actor_snapshot,
    )
    return (video, ctx)


def build_modules(config: PPOConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    obs_contract = env.io_contract.observation
    backbone_name = torchrl_env_contract.resolve_backbone_name(config.backbone_name, obs_contract)
    backbone_spec = backbone.BackboneSpec(
        name=backbone_name, hidden_sizes=tuple(config.backbone_hidden_sizes), activation=config.backbone_activation, layer_norm=bool(config.backbone_layer_norm)
    )
    actor_head_spec = backbone.HeadSpec(hidden_sizes=tuple(config.actor_head_hidden_sizes), activation=config.head_activation)
    critic_head_spec = backbone.HeadSpec(hidden_sizes=tuple(config.critic_head_hidden_sizes), activation=config.head_activation)
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
        actor = tr_modules.ProbabilisticActor(actor_module, in_keys=["logits"], distribution_class=torch.distributions.Categorical, return_log_prob=True)
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


def build_training(config: PPOConfig, env: _EnvSetup, modules: _Modules, *, runtime: torchrl_runtime.TorchRLRuntime) -> _TrainingSetup:
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
    train_params = _unique_param_list(modules.actor, modules.critic, extra_params=[modules.log_std] if modules.log_std is not None else None)
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


def _resume_if_requested(config: PPOConfig, modules: _Modules, training: _TrainingSetup, *, device: torch.device) -> _TrainState:
    state = _TrainState()
    if not config.resume_from:
        return state
    resume_path = Path(config.resume_from)
    loaded = rl_checkpointing.load_checkpoint(resume_path, device=device)
    actor_snapshot: dict[str, Any] = {"backbone": loaded["actor_backbone"], "head": loaded["actor_head"]}
    if "log_std" in loaded:
        actor_snapshot["log_std"] = loaded["log_std"]
    _restore_actor_state(modules, actor_snapshot, device=device)
    modules.critic_backbone.load_state_dict(loaded["critic_backbone"])
    modules.critic_head.load_state_dict(loaded["critic_head"])
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


def _ppo_update(config: PPOConfig, training: _TrainingSetup, *, batch, device: torch.device) -> tuple[list[float], list[float]]:
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
        if config.target_kl is not None and approx_kls and (float(np.mean(approx_kls)) > float(config.target_kl)):
            break
    return (approx_kls, clipfracs)


def _evaluate_actor(config: PPOConfig, env: _EnvSetup, modules: _Modules, *, device: torch.device, eval_seed: int) -> float:
    obs_contract = _resolve_observation_contract_for_env(config, env)
    from_pixels = obs_contract.mode == "pixels"
    eval_env = _build_eval_env_conf(config, env, from_pixels=from_pixels)
    eval_policy = torchrl_actor_eval.ActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        device=device,
        obs_contract=obs_contract,
        is_discrete=bool(getattr(env, "is_discrete", False)),
    )
    traj, _ = episode_rollout.collect_denoised_trajectory(eval_env, eval_policy, num_denoise=config.num_denoise, i_noise=int(eval_seed))
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
    do_eval = _is_due(int(iteration), int(config.eval_interval))
    do_log = _is_due(int(iteration), int(config.log_interval))
    if not do_eval:
        if do_log:
            from rl import logger as rl_logger

            elapsed = time.time() - start_time
            rl_logger.log_progress_iteration(iteration, training.num_iterations, training.frames_per_batch, elapsed, algo_name="ppo")
        return
    plan = build_eval_plan(
        current=iteration,
        interval=int(config.eval_interval),
        seed=int(env.problem_seed),
        eval_seed_base=config.eval_seed_base,
        eval_noise_mode=config.eval_noise_mode,
    )
    state.last_eval_return = _evaluate_actor(config, env, modules, device=device, eval_seed=plan.eval_seed)
    state.best_return, state.best_actor_state, _ = _update_best_actor_if_improved(
        eval_return=float(state.last_eval_return),
        best_return=float(state.best_return),
        best_actor_state=state.best_actor_state,
        capture_actor_state=lambda: torchrl_actor_eval.capture_actor_snapshot(modules),
    )
    state.last_heldout_return = None
    if config.num_denoise_passive is not None and state.best_actor_state is not None:
        obs_contract = _resolve_observation_contract_for_env(config, env)
        from_pixels = obs_contract.mode == "pixels"
        best_eval_policy = torchrl_actor_eval.ActorEvalPolicy(
            modules.actor_backbone,
            modules.actor_head,
            modules.obs_scaler,
            device=device,
            obs_contract=obs_contract,
            is_discrete=bool(getattr(env, "is_discrete", False)),
        )
        state.last_heldout_return = _eval_heldout_with_best_actor(
            best_actor_state=state.best_actor_state,
            num_denoise_passive=config.num_denoise_passive,
            heldout_i_noise=plan.heldout_i_noise,
            with_actor_state=lambda snapshot: torchrl_actor_eval.use_actor_snapshot(modules, snapshot, device=device),
            evaluate_for_best=episode_rollout.evaluate_for_best,
            eval_env_conf=_build_eval_env_conf(config, env, from_pixels=from_pixels),
            eval_policy=best_eval_policy,
        )
    elapsed = time.time() - start_time
    global_step = iteration * training.frames_per_batch
    record = _build_eval_record(
        iteration=int(iteration),
        global_step=int(global_step),
        eval_return=float(state.last_eval_return),
        heldout_return=state.last_heldout_return,
        best_return=float(state.best_return),
        approx_kl=float(np.mean(approx_kls)) if approx_kls else None,
        clipfrac=float(np.mean(clipfracs)) if clipfracs else None,
        started_at=float(start_time),
        now=float(start_time + elapsed),
    )
    from rl import logger as rl_logger

    rl_logger.append_metrics(training.metrics_path, record)
    if do_log:
        from rl import logger as rl_logger

        rl_logger.log_eval_iteration(
            iteration,
            training.num_iterations,
            training.frames_per_batch,
            eval_return=state.last_eval_return,
            heldout_return=state.last_heldout_return,
            best_return=state.best_return,
            algo_metrics={"kl": record["approx_kl"], "clipfrac": record["clipfrac"]},
            algo_name="ppo",
            elapsed=elapsed,
        )


def _run_training_loop(
    config: PPOConfig, env: _EnvSetup, modules: _Modules, training: _TrainingSetup, state: _TrainState, *, collector, device: torch.device
) -> None:
    start_time = time.time()

    def run_iteration(iteration: int, batch):
        _anneal_lr(config, training.optimizer, iteration=iteration, num_iterations=training.num_iterations)
        approx_kls, clipfracs = _ppo_update(config, training, batch=batch, device=device)
        _maybe_eval_and_log(
            config, env, modules, training, state, iteration=iteration, approx_kls=approx_kls, clipfracs=clipfracs, device=device, start_time=start_time
        )
        save_periodic_checkpoint(config=config, training_setup=training, modules=modules, train_state=state, iteration=iteration)

    if getattr(config, "profile_enable", False):
        run_with_profiler(config, collector, run_iteration, device=device, num_iterations=training.num_iterations, start_iteration=state.start_iteration)
    else:
        for iteration, batch in enumerate(collector, start=state.start_iteration + 1):
            if iteration > training.num_iterations:
                break
            run_iteration(iteration, batch)


def _log_ppo_config(config, env, training, runtime, from_pixels, backbone_info):
    print(
        f"[rl/ppo/torchrl] env_tag={config.env_tag} exp_dir={training.exp_dir} seed={config.seed} problem_seed={env.problem_seed} device={runtime.device.type} obs_dim={env.obs_dim} act_dim={env.act_dim} total_timesteps={config.total_timesteps} num_envs={config.num_envs} num_steps={config.num_steps} frames_per_batch={training.frames_per_batch} num_iterations={training.num_iterations} update_epochs={config.update_epochs} eval_interval={config.eval_interval} collector={runtime.collector_backend} single_env={runtime.single_env_backend} from_pixels={from_pixels}{backbone_info} share_backbone={bool(config.share_backbone)}",
        flush=True,
    )
    print(
        f"[rl/ppo/torchrl] actor_head={list(config.actor_head_hidden_sizes)} value_head={list(config.critic_head_hidden_sizes)} log_std_init={config.log_std_init} lr={config.learning_rate} gamma={config.gamma} gae_lambda={config.gae_lambda} clip_coef={config.clip_coef} vf_coef={config.vf_coef} ent_coef={config.ent_coef}",
        flush=True,
    )


def train_ppo(config: PPOConfig) -> TrainResult:
    if config.eval_noise_mode is not None:
        normalize_eval_noise_mode(config.eval_noise_mode)
    resolved = core_env_conf.resolve_run_seeds(seed=int(config.seed), problem_seed=config.problem_seed, noise_seed_0=config.noise_seed_0)
    config.problem_seed = int(resolved.problem_seed)
    config.noise_seed_0 = int(resolved.noise_seed_0)
    seed_all(seed_util.global_seed_for_run(int(resolved.problem_seed)))
    env = build_env_setup(config)
    runtime = config.resolve_runtime(capabilities=_PPO_RUNTIME_CAPABILITIES)
    modules = build_modules(config, env, device=runtime.device)
    training = build_training(config, env, modules, runtime=runtime)
    state = _resume_if_requested(config, modules, training, device=runtime.device)
    from_pixels = env.io_contract.observation.mode == "pixels"
    backbone_resolved = torchrl_env_contract.resolve_backbone_name(config.backbone_name, env.io_contract.observation)
    is_cnn = backbone_resolved in {"nature_cnn", "nature_cnn_atari"}
    backbone_info = f" backbone={backbone_resolved}"
    if not is_cnn:
        backbone_info += f" hidden={list(config.backbone_hidden_sizes)} act={config.backbone_activation} ln={bool(config.backbone_layer_norm)}"
    _log_ppo_config(config, env, training, runtime, from_pixels, backbone_info)
    remaining_iterations = max(0, training.num_iterations - state.start_iteration)
    video, ctx = _make_video_context(config, env, from_pixels=from_pixels)
    if remaining_iterations == 0:
        video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)
        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_iterations=training.num_iterations,
        )
    collector = _build_collector(config, env, modules, training, runtime=runtime, remaining_iterations=remaining_iterations)
    from rl import logger as rl_logger

    rl_logger.log_run_header("ppo", config, env, training, runtime)
    train_start = time.time()
    try:
        _run_training_loop(config, env, modules, training, state, collector=collector, device=runtime.device)
    finally:
        collector.shutdown()
    total_time = time.time() - train_start
    rl_logger.log_run_footer(state.best_return, training.num_iterations, total_time, algo_name="ppo")
    save_final_checkpoint(config=config, training_setup=training, modules=modules, train_state=state)
    video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)
    return TrainResult(
        best_return=float(state.best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=state.last_heldout_return,
        num_iterations=training.num_iterations,
    )


def register():
    registry.register_algo("ppo", PPOConfig, train_ppo)
