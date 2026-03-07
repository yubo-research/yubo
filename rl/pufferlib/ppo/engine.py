from __future__ import annotations

import dataclasses
import importlib
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from .config import PufferPPOConfig, TrainResult

__all__ = ["PufferPPOConfig", "TrainResult", "register", "train_ppo_puffer", "train_ppo_puffer_impl"]


def _ensure_pixel_obs_format(obs_t: torch.Tensor, *, channels: int, size: int, scale_float_255: bool) -> torch.Tensor:
    ensure_pixel_obs_format = importlib.import_module("rl.core.pixel_transform").ensure_pixel_obs_format
    return ensure_pixel_obs_format(obs_t, channels=channels, size=size, scale_float_255=scale_float_255)


def _resolve_device(device_raw: str) -> torch.device:
    select_device = importlib.import_module("rl.core.runtime").select_device
    return select_device(str(device_raw))


def _infer_channels(shape: tuple[int, ...], *, fallback: int) -> int:
    env_utils = importlib.import_module("rl.pufferlib.sac.env_utils")
    return int(env_utils._infer_channels(shape, fallback=int(fallback)))


def _infer_image_size(shape: tuple[int, ...], *, default_size: int) -> int:
    env_utils = importlib.import_module("rl.pufferlib.sac.env_utils")
    return int(env_utils._infer_image_size(shape, default_size=int(default_size)))


def _infer_observation_spec(config: PufferPPOConfig, obs_np: np.ndarray):
    specs = importlib.import_module("rl.pufferlib.ppo.specs")
    ppo_envs = importlib.import_module("rl.core.ppo_envs")
    obs_arr = np.asarray(obs_np)
    if obs_arr.ndim == 0:
        raise ValueError("Observation must include at least one dimension.")
    raw_shape = tuple((int(v) for v in (obs_arr.shape[1:] if obs_arr.ndim >= 2 else obs_arr.shape)))
    if ppo_envs.is_atari_env_tag(str(config.env_tag)):
        return specs._ObservationSpec(mode="pixels", raw_shape=raw_shape, channels=4, image_size=84)
    backbone_name = str(config.backbone_name).strip().lower()
    looks_like_pixels = obs_arr.ndim >= 4 or (obs_arr.ndim >= 3 and "nature_cnn" in backbone_name)
    if looks_like_pixels:
        channels = _infer_channels(raw_shape, fallback=max(1, int(config.framestack)))
        image_size = _infer_image_size(raw_shape, default_size=84)
        return specs._ObservationSpec(mode="pixels", raw_shape=raw_shape, channels=channels, image_size=image_size)
    vector_dim = int(np.prod(raw_shape)) if raw_shape else 1
    return specs._ObservationSpec(mode="vector", raw_shape=raw_shape, vector_dim=vector_dim)


def _prepare_obs(obs_np: np.ndarray, *, obs_spec, device: torch.device) -> torch.Tensor:
    obs_t = torch.as_tensor(obs_np)
    if obs_spec.mode == "pixels":
        channels = int(obs_spec.channels or 3)
        size = int(obs_spec.image_size or 84)
        obs_t = _ensure_pixel_obs_format(obs_t, channels=channels, size=size, scale_float_255=False)
    elif obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    elif obs_t.ndim > 2:
        obs_t = obs_t.reshape(obs_t.shape[0], -1)
    return obs_t.to(device=device, dtype=torch.float32)


def _action_spec_from_space(action_space):
    specs = importlib.import_module("rl.pufferlib.ppo.specs")
    if hasattr(action_space, "nvec"):
        raise ValueError("MultiDiscrete action spaces are not supported by ppo_puffer yet.")
    shape = tuple((int(v) for v in getattr(action_space, "shape", ()) or ()))
    if hasattr(action_space, "n") and len(shape) == 0:
        return specs._ActionSpec(kind="discrete", dim=int(action_space.n))
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise ValueError(f"Unsupported action space for ppo_puffer: {type(action_space)!r}")
    dim = int(np.prod(shape)) if shape else 1
    low, high = specs.normalize_action_bounds(np.asarray(action_space.low), np.asarray(action_space.high), dim=dim)
    return specs._ActionSpec(kind="continuous", dim=dim, low=low, high=high)


def _make_vector_env(config: PufferPPOConfig):
    ppo_envs = importlib.import_module("rl.core.ppo_envs")
    make_vector_env_shared = importlib.import_module("rl.pufferlib.vector_env").make_vector_env
    import_pufferlib_modules = importlib.import_module("rl.pufferlib_compat").import_pufferlib_modules
    return make_vector_env_shared(
        config,
        import_pufferlib_modules_fn=import_pufferlib_modules,
        is_atari_env_tag_fn=ppo_envs.is_atari_env_tag,
        to_puffer_game_name_fn=ppo_envs.to_puffer_game_name,
        resolve_gym_env_name_fn=ppo_envs.resolve_gym_env_name,
    )


def _build_plan(config: PufferPPOConfig):
    specs = importlib.import_module("rl.pufferlib.ppo.specs")
    num_envs = int(config.num_envs)
    num_steps = int(config.num_steps)
    batch_size = num_envs * num_steps
    if batch_size <= 0:
        raise ValueError("num_envs * num_steps must be > 0")
    num_minibatches = int(config.num_minibatches)
    if batch_size % num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    num_iterations = int(config.total_timesteps) // batch_size
    if num_iterations <= 0:
        raise ValueError("total_timesteps is too small for num_envs * num_steps")
    return specs._TrainPlan(
        num_envs=num_envs, num_steps=num_steps, batch_size=batch_size, minibatch_size=batch_size // num_minibatches, num_iterations=num_iterations
    )


def _prepare_outputs(config: PufferPPOConfig) -> Path:
    write_config = importlib.import_module("analysis.data_io").write_config
    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), dataclasses.asdict(config))
    return exp_dir / "metrics.jsonl"


def _resolve_backbone_name(config: PufferPPOConfig, obs_spec) -> str:
    backbone_name = str(config.backbone_name)
    key = backbone_name.strip().lower()
    if obs_spec.mode != "pixels":
        return backbone_name
    channels = int(obs_spec.channels or 3)
    if key == "mlp":
        return "nature_cnn_atari" if channels == 4 else "nature_cnn"
    if key == "nature_cnn" and channels == 4:
        return "nature_cnn_atari"
    if key == "nature_cnn_atari" and channels != 4:
        return "nature_cnn"
    return backbone_name


def _build_model(config: PufferPPOConfig, obs_spec, action_spec):
    backbone = importlib.import_module("rl.backbone")
    specs = importlib.import_module("rl.pufferlib.ppo.specs")
    input_dim = 64 if obs_spec.mode == "pixels" else int(obs_spec.vector_dim or 1)
    backbone_spec = backbone.BackboneSpec(
        name=_resolve_backbone_name(config, obs_spec),
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=str(config.backbone_activation),
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = backbone.HeadSpec(hidden_sizes=tuple(config.actor_head_hidden_sizes), activation=str(config.head_activation))
    critic_head_spec = backbone.HeadSpec(hidden_sizes=tuple(config.critic_head_hidden_sizes), activation=str(config.head_activation))
    if bool(config.share_backbone):
        shared, feat_dim = backbone.build_backbone(backbone_spec, input_dim=input_dim)
        actor_backbone, critic_backbone = (shared, shared)
        actor_feat_dim, critic_feat_dim = (feat_dim, feat_dim)
    else:
        actor_backbone, actor_feat_dim = backbone.build_backbone(backbone_spec, input_dim=input_dim)
        critic_backbone, critic_feat_dim = backbone.build_backbone(backbone_spec, input_dim=input_dim)
    actor_head = backbone.build_mlp_head(actor_head_spec, input_dim=actor_feat_dim, output_dim=int(action_spec.dim))
    critic_head = backbone.build_mlp_head(critic_head_spec, input_dim=critic_feat_dim, output_dim=1)
    specs.init_linear(actor_backbone, gain=0.5)
    specs.init_linear(critic_backbone, gain=0.5)
    specs.init_linear(actor_head, gain=0.01)
    specs.init_linear(critic_head, gain=1.0)
    return specs._ActorCritic(
        actor_backbone=actor_backbone,
        critic_backbone=critic_backbone,
        actor_head=actor_head,
        critic_head=critic_head,
        action_spec=action_spec,
        log_std_init=float(getattr(config, "log_std_init", -0.5)),
    )


def _seed_everything(seed: int) -> None:
    seed_everything = importlib.import_module("rl.core.runtime").seed_everything
    seed_everything(int(seed))


def _init_runtime(config: PufferPPOConfig, plan, device: torch.device, envs):
    specs = importlib.import_module("rl.pufferlib.ppo.specs")
    env_seed = int(config.problem_seed) if config.problem_seed is not None else int(config.seed)
    next_obs_np, _ = envs.reset(seed=env_seed)
    obs_spec = _infer_observation_spec(config, next_obs_np)
    next_obs = _prepare_obs(next_obs_np, obs_spec=obs_spec, device=device)
    effective_num_envs = int(next_obs.shape[0])
    if effective_num_envs != plan.num_envs:
        raise ValueError(
            f"Runtime num_envs mismatch: config.num_envs={plan.num_envs}, env_batch={effective_num_envs}. For multiprocessing backend, ensure vector_batch_size matches num_envs."
        )
    action_spec = _action_spec_from_space(envs.single_action_space)
    model = _build_model(config, obs_spec=obs_spec, action_spec=action_spec).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.learning_rate), eps=1e-05)
    next_done = torch.zeros(plan.num_envs, dtype=torch.float32, device=device)
    obs_shape = tuple(next_obs.shape[1:])
    if action_spec.kind == "discrete":
        action_buf_shape = (plan.num_steps, plan.num_envs)
        action_dtype = torch.long
    else:
        action_buf_shape = (plan.num_steps, plan.num_envs, int(action_spec.dim))
        action_dtype = torch.float32
    buffer = specs._RolloutBuffer(
        obs=torch.zeros((plan.num_steps, plan.num_envs, *obs_shape), device=device),
        actions=torch.zeros(action_buf_shape, dtype=action_dtype, device=device),
        logprobs=torch.zeros((plan.num_steps, plan.num_envs), device=device),
        rewards=torch.zeros((plan.num_steps, plan.num_envs), device=device),
        dones=torch.zeros((plan.num_steps, plan.num_envs), device=device),
        values=torch.zeros((plan.num_steps, plan.num_envs), device=device),
    )
    state = specs._RuntimeState(
        next_obs=next_obs,
        next_done=next_done,
        obs_spec=obs_spec,
        action_spec=action_spec,
        global_step=0,
        start_iteration=0,
        start_time=time.time(),
        best_actor_state=None,
        best_return=-float("inf"),
        last_eval_return=float("nan"),
        last_heldout_return=None,
        last_episode_return=float("nan"),
        eval_env_conf=None,
    )
    return (model, optimizer, obs_shape, buffer, state)


def _build_eval_env_conf(config: PufferPPOConfig, *, obs_spec):
    build_eval_env_conf_impl = importlib.import_module("rl.pufferlib.ppo.eval_config").build_eval_env_conf
    ppo_envs = importlib.import_module("rl.core.ppo_envs")
    return build_eval_env_conf_impl(
        config, obs_mode=obs_spec.mode, is_atari_env_tag_fn=ppo_envs.is_atari_env_tag, resolve_gym_env_name_fn=ppo_envs.resolve_gym_env_name
    )


def make_vector_env(config: PufferPPOConfig):
    return _make_vector_env(config)


def build_eval_env_conf(config: PufferPPOConfig, *, obs_spec):
    return _build_eval_env_conf(config, obs_spec=obs_spec)


def _run_training(config: PufferPPOConfig, plan, device: torch.device, metrics_path: Path, envs, *, build_eval_env_conf_fn) -> TrainResult:
    puffer_train_ops = importlib.import_module("rl.pufferlib.ppo.training_ops")
    puffer_ckpt = importlib.import_module("rl.pufferlib.ppo.checkpoint")
    puffer_metrics = importlib.import_module("rl.pufferlib.ppo.metrics")
    puffer_eval = importlib.import_module("rl.pufferlib.ppo.eval")
    checkpoint_manager_cls = importlib.import_module("rl.checkpointing").CheckpointManager
    rl_logger = importlib.import_module("rl.logger")
    model, optimizer, obs_shape, buffer, state = _init_runtime(config, plan, device, envs)
    checkpoint_manager = checkpoint_manager_cls(exp_dir=metrics_path.parent)
    puffer_ckpt.restore_checkpoint_if_requested(config, plan, model, optimizer, state, device=device)
    rl_logger.log_run_header_basic(
        algo_name="ppo",
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        backbone_name=str(config.backbone_name),
        from_pixels=state.obs_spec.mode == "pixels",
        obs_dim=64 if state.obs_spec.mode == "pixels" else int(state.obs_spec.vector_dim or 1),
        act_dim=int(state.action_spec.dim),
        frames_per_batch=int(plan.batch_size),
        num_iterations=int(plan.num_iterations),
        device_type=str(device.type),
    )
    b_inds = np.arange(plan.batch_size)
    for iteration in range(state.start_iteration + 1, plan.num_iterations + 1):
        puffer_metrics._maybe_anneal_lr(config, plan, optimizer, iteration)
        puffer_train_ops.collect_rollout(plan, model, envs, buffer, state, device, prepare_obs_fn=_prepare_obs)
        advantages, returns = puffer_train_ops.compute_advantages(plan, config, model, state, buffer, device)
        batch = puffer_train_ops.flatten_batch(plan, buffer, advantages, returns, obs_shape)
        update_stats = puffer_train_ops.ppo_update(config, plan, model, optimizer, batch, b_inds)
        puffer_eval.maybe_eval_and_update_state(
            config, model, state, iteration=iteration, device=device, build_eval_env_conf_fn=build_eval_env_conf_fn, prepare_obs_fn=_prepare_obs
        )
        metric = puffer_metrics._metric_payload(iteration, plan, optimizer, state, update_stats, batch)
        puffer_metrics._append_metrics_line(metrics_path, metric)
        puffer_metrics._log_iteration(config, metric)
        puffer_ckpt.maybe_save_periodic_checkpoint(config, checkpoint_manager, model, optimizer, state, iteration=iteration)
    best_return = state.best_return
    if best_return == -float("inf"):
        best_return = float("nan")
    total_time = time.time() - state.start_time
    rl_logger.log_run_footer(float(best_return), int(plan.num_iterations), float(total_time), algo_name="ppo")
    final_iteration = int(max(state.start_iteration, plan.num_iterations))
    puffer_ckpt.save_final_checkpoint(config, checkpoint_manager, model, optimizer, state, iteration=final_iteration)
    puffer_eval.maybe_render_videos(
        config, model, state, exp_dir=metrics_path.parent, device=device, build_eval_env_conf_fn=build_eval_env_conf_fn, prepare_obs_fn=_prepare_obs
    )
    return TrainResult(
        best_return=float(best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=puffer_metrics._as_optional_finite(state.last_heldout_return),
        num_iterations=int(plan.num_iterations),
    )


def train_ppo_puffer_impl(config: PufferPPOConfig) -> TrainResult:
    puffer_eval = importlib.import_module("rl.pufferlib.ppo.eval")
    core_env_conf = importlib.import_module("rl.core.env_conf")
    puffer_eval.validate_eval_config(config)
    resolved = core_env_conf.resolve_run_seeds(seed=int(config.seed), problem_seed=config.problem_seed, noise_seed_0=config.noise_seed_0)
    config.problem_seed = int(resolved.problem_seed)
    config.noise_seed_0 = int(resolved.noise_seed_0)
    device = _resolve_device(config.device)
    _seed_everything(int(core_env_conf.global_seed_for_run(int(config.problem_seed))))
    plan = _build_plan(config)
    metrics_path = _prepare_outputs(config)
    with closing(make_vector_env(config)) as envs:
        return _run_training(config, plan, device, metrics_path, envs, build_eval_env_conf_fn=build_eval_env_conf)


def train_ppo_puffer(config: PufferPPOConfig) -> TrainResult:
    return train_ppo_puffer_impl(config)


def register() -> None:
    register_algo = importlib.import_module("rl.registry").register_algo
    register_algo("ppo", PufferPPOConfig, train_ppo_puffer, backend="pufferlib")
