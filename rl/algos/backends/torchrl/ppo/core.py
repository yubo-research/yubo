"""PPO training coordinator. Setup (env, modules, training) inlined to reduce dependency depth."""

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
from torchrl.envs.libs.gym import set_gym_backend

from ..common.pixel_transform import AtariObservationTransform, PixelsToObservation
from ..common.profiler import run_with_profiler
from . import deps as op_deps
from . import models as op_models
from .actor_eval import (
    capture_actor_snapshot as _capture_actor_state,
)
from .actor_eval import (
    restore_actor_snapshot as _restore_actor_state,
)
from .config import _PPO_RUNTIME_CAPABILITIES, PPOConfig, TrainResult

__all__ = [
    "PPOConfig",
    "TrainResult",
    "_TanhNormal",
    "_capture_actor_state",
    "_restore_actor_state",
    "register",
    "torch",
    "train_ppo",
]


# --- Setup: dataclasses and helpers ---


@dataclasses.dataclass(frozen=True)
class _EnvSetup:
    env_conf: object
    io_contract: op_deps.torchrl_env_contract.EnvIOContract
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
    return op_deps.tr_dists.TanhNormal


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


def _is_dm_control_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("dm_control/")


def _is_atari_env(env_conf) -> bool:
    return getattr(env_conf, "env_name", "").startswith("ALE/")


def _resolve_observation_contract_for_env(
    config: PPOConfig,
    env: _EnvSetup | object,
) -> op_deps.torchrl_env_contract.ObservationContract:
    io_contract = getattr(env, "io_contract", None)
    observation = getattr(io_contract, "observation", None)
    if observation is not None:
        return observation

    env_conf = getattr(env, "env_conf", None)
    if env_conf is not None:
        return op_deps.torchrl_env_contract.resolve_observation_contract(env_conf, default_image_size=84)

    if bool(getattr(config, "from_pixels", False)):
        return op_deps.torchrl_env_contract.ObservationContract(
            mode="pixels",
            raw_shape=(),
            model_channels=3,
            image_size=84,
        )
    return op_deps.torchrl_env_contract.ObservationContract(
        mode="vector",
        raw_shape=(),
        vector_dim=1,
    )


def _make_collect_env_atari(env_conf, env_index: int = 0):
    import problems.env_conf_atari_dm  # noqa: F401 - required in subprocess

    base = env_conf.make()
    seed = int(env_conf.problem_seed) + env_index
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)
    transforms = op_deps.tr_transforms.Compose(
        AtariObservationTransform(size=84),
        op_deps.tr_transforms.DoubleToFloat(),
    )
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), transforms)


def _make_collect_env_dm_control(env_conf, env_index: int = 0):
    from problems.shimmy_dm_control import make as make_dm_env

    seed = int(env_conf.problem_seed) + env_index
    from_pixels = bool(getattr(env_conf, "from_pixels", False))
    pixels_only = bool(getattr(env_conf, "pixels_only", True))

    base = make_dm_env(
        env_conf.env_name,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
    )
    base.reset(seed=seed)
    if hasattr(base, "action_space") and hasattr(base.action_space, "seed"):
        base.action_space.seed(seed)

    if from_pixels:
        transforms = op_deps.tr_transforms.Compose(
            PixelsToObservation(size=84),
            op_deps.tr_transforms.DoubleToFloat(),
        )
    else:
        transforms = op_deps.tr_transforms.DoubleToFloat()
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
    return tr_envs.TransformedEnv(tr_envs.GymWrapper(base), op_deps.tr_transforms.DoubleToFloat())


def _make_collect_env_factory(env_conf, num_envs: int):
    env_index = [0]

    def fn():
        idx = env_index[0]
        env_index[0] = (idx + 1) % num_envs
        return _make_collect_env(env_conf, env_index=idx)

    return fn


def build_env_setup(config: PPOConfig) -> _EnvSetup:
    problem_seed = op_deps.seed_util.resolve_problem_seed(seed=config.seed, problem_seed=config.problem_seed)
    noise_seed_0 = op_deps.seed_util.resolve_noise_seed_0(problem_seed=problem_seed, noise_seed_0=config.noise_seed_0)
    env_conf = op_deps.get_env_conf(
        config.env_tag,
        problem_seed=problem_seed,
        noise_seed_0=noise_seed_0,
        from_pixels=bool(getattr(config, "from_pixels", False)),
        pixels_only=getattr(config, "pixels_only", True),
    )
    if env_conf.gym_conf is None:
        raise ValueError(f"PPO expects a gym env_tag, got {config.env_tag}")
    env_conf.ensure_spaces()

    io_contract = op_deps.torchrl_env_contract.resolve_env_io_contract(env_conf, default_image_size=84)
    obs_dim = 64 if io_contract.observation.mode == "pixels" else int(io_contract.observation.vector_dim or 1)
    act_dim = int(io_contract.action.dim)
    action_low = io_contract.action.low
    action_high = io_contract.action.high
    lb, width = op_deps.torchrl_common.obs_scale_from_env(env_conf)
    return _EnvSetup(
        env_conf=env_conf,
        io_contract=io_contract,
        problem_seed=int(problem_seed),
        noise_seed_0=int(noise_seed_0),
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        obs_lb=lb,
        obs_width=width,
        is_discrete=io_contract.action.kind == "discrete",
    )


def build_modules(config: PPOConfig, env: _EnvSetup, *, device: torch.device) -> _Modules:
    obs_contract = env.io_contract.observation
    backbone_name = op_deps.torchrl_env_contract.resolve_backbone_name(config.backbone_name, obs_contract)
    backbone_spec = op_deps.backbone.BackboneSpec(
        name=backbone_name,
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=config.backbone_activation,
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = op_deps.backbone.HeadSpec(
        hidden_sizes=tuple(config.actor_head_hidden_sizes),
        activation=config.head_activation,
    )
    critic_head_spec = op_deps.backbone.HeadSpec(
        hidden_sizes=tuple(config.critic_head_hidden_sizes),
        activation=config.head_activation,
    )

    if config.share_backbone:
        shared_backbone, feat_dim = op_deps.backbone.build_backbone(backbone_spec, env.obs_dim)
        actor_backbone = shared_backbone
        critic_backbone = shared_backbone
        actor_feat_dim = feat_dim
        critic_feat_dim = feat_dim
    else:
        actor_backbone, actor_feat_dim = op_deps.backbone.build_backbone(backbone_spec, env.obs_dim)
        critic_backbone, critic_feat_dim = op_deps.backbone.build_backbone(backbone_spec, env.obs_dim)

    actor_head = op_deps.backbone.build_mlp_head(actor_head_spec, actor_feat_dim, env.act_dim)
    critic_head = op_deps.backbone.build_mlp_head(critic_head_spec, critic_feat_dim, 1)
    obs_scaler = op_deps.torchrl_common.ObsScaler(env.obs_lb, env.obs_width)
    critic_net = op_models.CriticNet(
        critic_backbone,
        critic_head,
        obs_scaler,
        obs_contract=obs_contract,
    )
    critic = td_nn.TensorDictModule(critic_net, in_keys=["observation"], out_keys=["state_value"])

    if env.is_discrete:
        log_std = None
        actor_net = op_models.DiscreteActorNet(
            actor_backbone,
            actor_head,
            obs_scaler,
            obs_contract=obs_contract,
        )
        actor_module = td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
        actor = op_deps.tr_modules.ProbabilisticActor(
            actor_module,
            in_keys=["logits"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
        )
        actor_param_count = _count_unique_params(actor_backbone, actor_head)
    else:
        log_std = nn.Parameter(torch.full((env.act_dim,), float(config.log_std_init)))
        actor_net = op_models.ActorNet(
            actor_backbone,
            actor_head,
            log_std,
            obs_scaler,
            obs_contract=obs_contract,
        )
        actor_module = td_nn.TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
        actor = op_deps.tr_modules.ProbabilisticActor(
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
    runtime: op_deps.torchrl_runtime.TorchRLRuntime,
) -> _TrainingSetup:
    frames_per_batch = int(config.num_envs * config.num_steps)
    num_iterations = int(config.total_timesteps // frames_per_batch)
    if num_iterations <= 0:
        raise ValueError("total_timesteps too small for num_envs * num_steps.")

    env_conf = env.env_conf
    vec_env = None
    if runtime.collector_backend == "single":
        env_factory = _make_collect_env_factory(env_conf, int(config.num_envs))
        if runtime.single_env_backend == "parallel":
            vec_env = tr_envs.ParallelEnv(
                int(config.num_envs),
                env_factory,
                serial_for_single=True,
            )
        else:
            vec_env = tr_envs.SerialEnv(
                int(config.num_envs),
                env_factory,
                serial_for_single=True,
            )

    loss_module = op_deps.tr_objectives.ClipPPOLoss(
        modules.actor,
        modules.critic,
        clip_epsilon=config.clip_coef,
        entropy_coeff=config.ent_coef,
        critic_coeff=config.vf_coef,
        normalize_advantage=config.norm_adv,
        clip_value=config.clip_vloss,
        functional=False,
    )
    gae = op_deps.tr_objectives.value.GAE(
        gamma=config.gamma,
        lmbda=config.gae_lambda,
        value_network=modules.critic,
    )

    train_params = _unique_param_list(
        modules.actor,
        modules.critic,
        extra_params=[modules.log_std] if modules.log_std is not None else None,
    )
    optimizer = optim.AdamW(train_params, lr=config.learning_rate, eps=1e-5, weight_decay=0.0)

    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    op_deps.write_config(str(exp_dir), config.to_dict())
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
        checkpoint_manager=op_deps.rl_checkpointing.CheckpointManager(exp_dir=exp_dir),
    )


# --- Core: training loop ---


def _build_collector(
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    training: _TrainingSetup,
    *,
    runtime: op_deps.torchrl_runtime.TorchRLRuntime,
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
            **op_deps.torchrl_common.collector_device_kwargs(runtime.device),
        )

    if runtime.collector_workers is None:
        raise RuntimeError("multi collector backend requires collector_workers")
    num_workers = int(runtime.collector_workers)

    create_env_fns = [(lambda i=i: _make_collect_env(env.env_conf, env_index=i)) for i in range(num_workers)]
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
    loaded = op_deps.rl_checkpointing.load_checkpoint(resume_path, device=device)
    modules.actor_backbone.load_state_dict(loaded["actor_backbone"])
    modules.actor_head.load_state_dict(loaded["actor_head"])
    modules.critic_backbone.load_state_dict(loaded["critic_backbone"])
    modules.critic_head.load_state_dict(loaded["critic_head"])
    if modules.log_std is not None and "log_std" in loaded:
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


def _anneal_lr(
    config: PPOConfig,
    optimizer: optim.Optimizer,
    *,
    iteration: int,
    num_iterations: int,
) -> None:
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
    obs_contract = _resolve_observation_contract_for_env(config, env)
    from_pixels = obs_contract.mode == "pixels"
    eval_env = op_deps.get_env_conf(
        config.env_tag,
        problem_seed=env.problem_seed,
        noise_seed_0=env.noise_seed_0,
        from_pixels=from_pixels,
        pixels_only=getattr(config, "pixels_only", True),
    )
    eval_policy = op_deps.torchrl_actor_eval.ActorEvalPolicy(
        modules.actor_backbone,
        modules.actor_head,
        modules.obs_scaler,
        device=device,
        obs_contract=obs_contract,
        is_discrete=bool(getattr(env, "is_discrete", False)),
    )
    opt_traj = __import__("optimizer.opt_trajectories", fromlist=["collect_denoised_trajectory"])
    traj, _ = opt_traj.collect_denoised_trajectory(
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
            from rl.algos import logger as rl_logger

            elapsed = time.time() - start_time
            rl_logger.log_progress_iteration(
                iteration,
                training.num_iterations,
                training.frames_per_batch,
                elapsed,
                algo_name="ppo",
            )
        return

    plan = op_deps.build_eval_plan(
        current=iteration,
        interval=int(config.eval_interval),
        seed=int(config.seed),
        eval_seed_base=config.eval_seed_base,
        eval_noise_mode=config.eval_noise_mode,
    )
    state.last_eval_return = _evaluate_actor(config, env, modules, device=device, eval_seed=plan.eval_seed)
    if state.last_eval_return > state.best_return:
        state.best_return = float(state.last_eval_return)
        state.best_actor_state = op_deps.torchrl_actor_eval.capture_actor_snapshot(modules)

    state.last_heldout_return = None
    if config.num_denoise_passive_eval is not None and state.best_actor_state is not None:
        current = op_deps.torchrl_actor_eval.capture_actor_snapshot(modules)
        op_deps.torchrl_actor_eval.restore_actor_snapshot(
            modules,
            state.best_actor_state,
            device=device,
        )
        obs_contract = _resolve_observation_contract_for_env(config, env)
        from_pixels = obs_contract.mode == "pixels"
        best_eval_policy = op_deps.torchrl_actor_eval.ActorEvalPolicy(
            modules.actor_backbone,
            modules.actor_head,
            modules.obs_scaler,
            device=device,
            obs_contract=obs_contract,
            is_discrete=bool(getattr(env, "is_discrete", False)),
        )
        opt_traj = __import__("optimizer.opt_trajectories", fromlist=["evaluate_for_best"])
        state.last_heldout_return = float(
            opt_traj.evaluate_for_best(
                op_deps.get_env_conf(
                    config.env_tag,
                    problem_seed=env.problem_seed,
                    noise_seed_0=env.noise_seed_0,
                    from_pixels=from_pixels,
                    pixels_only=getattr(config, "pixels_only", True),
                ),
                best_eval_policy,
                config.num_denoise_passive_eval,
                i_noise=plan.heldout_i_noise,
            )
        )
        op_deps.torchrl_actor_eval.restore_actor_snapshot(modules, current, device=device)

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
    from rl.algos import logger as rl_logger

    rl_logger.append_metrics(training.metrics_path, record)
    if do_log:
        from rl.algos import logger as rl_logger

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
    config: PPOConfig,
    env: _EnvSetup,
    modules: _Modules,
    training: _TrainingSetup,
    state: _TrainState,
    *,
    collector,
    device: torch.device,
) -> None:
    start_time = time.time()

    def run_iteration(iteration: int, batch):
        _anneal_lr(
            config,
            training.optimizer,
            iteration=iteration,
            num_iterations=training.num_iterations,
        )
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
        op_deps.save_periodic_checkpoint(
            config=config,
            training_setup=training,
            modules=modules,
            train_state=state,
            iteration=iteration,
        )

    if getattr(config, "profile_enable", False):
        run_with_profiler(
            config,
            collector,
            run_iteration,
            device=device,
            num_iterations=training.num_iterations,
            start_iteration=state.start_iteration,
        )
    else:
        for iteration, batch in enumerate(collector, start=state.start_iteration + 1):
            if iteration > training.num_iterations:
                break
            run_iteration(iteration, batch)


def _log_ppo_config(config, env, training, runtime, from_pixels, backbone_info):
    print(
        "[rl/ppo/torchrl]"
        f" env_tag={config.env_tag}"
        f" exp_dir={training.exp_dir}"
        f" seed={config.seed} problem_seed={env.problem_seed}"
        f" device={runtime.device.type}"
        f" obs_dim={env.obs_dim}"
        f" act_dim={env.act_dim}"
        f" total_timesteps={config.total_timesteps}"
        f" num_envs={config.num_envs}"
        f" num_steps={config.num_steps}"
        f" frames_per_batch={training.frames_per_batch}"
        f" num_iterations={training.num_iterations}"
        f" update_epochs={config.update_epochs}"
        f" eval_interval={config.eval_interval}"
        f" collector={runtime.collector_backend}"
        f" single_env={runtime.single_env_backend}"
        f" from_pixels={from_pixels}"
        f"{backbone_info}"
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


def train_ppo(config: PPOConfig) -> TrainResult:
    set_gym_backend("gymnasium")
    if config.eval_noise_mode is not None:
        op_deps.normalize_eval_noise_mode(config.eval_noise_mode)
    problem_seed = op_deps.seed_util.resolve_problem_seed(seed=config.seed, problem_seed=config.problem_seed)
    op_deps.seed_all(op_deps.seed_util.global_seed_for_run(problem_seed))
    env = build_env_setup(config)
    runtime = config.resolve_runtime(capabilities=_PPO_RUNTIME_CAPABILITIES)
    modules = build_modules(config, env, device=runtime.device)
    training = build_training(
        config,
        env,
        modules,
        runtime=runtime,
    )
    state = _resume_if_requested(config, modules, training, device=runtime.device)

    from_pixels = env.io_contract.observation.mode == "pixels"
    backbone_resolved = op_deps.torchrl_env_contract.resolve_backbone_name(config.backbone_name, env.io_contract.observation)
    is_cnn = backbone_resolved in {"nature_cnn", "nature_cnn_atari"}
    backbone_info = f" backbone={backbone_resolved}"
    if not is_cnn:
        backbone_info += f" hidden={list(config.backbone_hidden_sizes)} act={config.backbone_activation} ln={bool(config.backbone_layer_norm)}"
    _log_ppo_config(config, env, training, runtime, from_pixels, backbone_info)

    remaining_iterations = max(0, training.num_iterations - state.start_iteration)
    if remaining_iterations == 0:
        video = __import__("common.video", fromlist=["RLVideoContext", "render_policy_videos_rl"])
        ctx = video.RLVideoContext(
            build_eval_env_conf=lambda ps, ns: op_deps.get_env_conf(
                config.env_tag,
                problem_seed=ps,
                noise_seed_0=ns,
                from_pixels=from_pixels,
                pixels_only=getattr(config, "pixels_only", True),
            ),
            make_eval_policy=lambda m, d: op_deps.torchrl_actor_eval.ActorEvalPolicy(
                m.actor_backbone,
                m.actor_head,
                m.obs_scaler,
                device=d,
                obs_contract=env.io_contract.observation,
                is_discrete=bool(getattr(env, "is_discrete", False)),
            ),
            capture_actor_state=op_deps.torchrl_actor_eval.capture_actor_snapshot,
            with_actor_state=op_deps.torchrl_actor_eval.use_actor_snapshot,
        )
        video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)
        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_iterations=training.num_iterations,
        )

    collector = _build_collector(
        config,
        env,
        modules,
        training,
        runtime=runtime,
        remaining_iterations=remaining_iterations,
    )
    from rl.algos import logger as rl_logger

    rl_logger.log_run_header("ppo", config, env, training, runtime)
    train_start = time.time()
    try:
        _run_training_loop(
            config,
            env,
            modules,
            training,
            state,
            collector=collector,
            device=runtime.device,
        )
    finally:
        collector.shutdown()
    total_time = time.time() - train_start
    rl_logger.log_run_footer(state.best_return, training.num_iterations, total_time, algo_name="ppo")
    op_deps.save_final_checkpoint(
        config=config,
        training_setup=training,
        modules=modules,
        train_state=state,
    )
    video = __import__("common.video", fromlist=["RLVideoContext", "render_policy_videos_rl"])
    ctx = video.RLVideoContext(
        build_eval_env_conf=lambda ps, ns: op_deps.get_env_conf(
            config.env_tag,
            problem_seed=ps,
            noise_seed_0=ns,
            from_pixels=from_pixels,
            pixels_only=getattr(config, "pixels_only", True),
        ),
        make_eval_policy=lambda m, d: op_deps.torchrl_actor_eval.ActorEvalPolicy(
            m.actor_backbone,
            m.actor_head,
            m.obs_scaler,
            device=d,
            obs_contract=env.io_contract.observation,
            is_discrete=bool(getattr(env, "is_discrete", False)),
        ),
        capture_actor_state=op_deps.torchrl_actor_eval.capture_actor_snapshot,
        with_actor_state=op_deps.torchrl_actor_eval.use_actor_snapshot,
    )
    video.render_policy_videos_rl(config, env, modules, training, state, ctx, device=runtime.device)
    return TrainResult(
        best_return=float(state.best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=state.last_heldout_return,
        num_iterations=training.num_iterations,
    )


def register():
    op_deps.registry.register_algo("ppo", PPOConfig, train_ppo)
