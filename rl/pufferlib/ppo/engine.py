"""PufferLib-backed PPO trainer."""

from __future__ import annotations

import dataclasses
import functools
import random
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from analysis.data_io import write_config
from problems.atari_env import _parse_atari_tag
from rl.backbone import BackboneSpec, HeadSpec, build_backbone, build_mlp_head
from rl.torchrl.common.pixel_transform import ensure_pixel_obs_format

from . import checkpoint as puffer_ckpt
from . import eval as puffer_eval
from . import eval_config as puffer_eval_config
from . import metrics as puffer_metrics
from .config import PufferPPOConfig, TrainResult
from .specs import (
    _ActionSpec,
    _ActorCritic,
    _FlatBatch,
    _ObservationSpec,
    _RolloutBuffer,
    _RuntimeState,
    _TrainPlan,
    _UpdateStats,
    init_linear,
)

__all__ = [
    "PufferPPOConfig",
    "TrainResult",
    "train_ppo_puffer_impl",
    "_to_puffer_game_name",
    "_resolve_gym_env_name",
    "_make_vector_env",
    "_build_eval_env_conf",
]


def _resolve_device(device_raw: str) -> torch.device:
    key = str(device_raw).strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(key)


def _to_puffer_game_name(env_tag: str) -> str:
    ale_id = _parse_atari_tag(str(env_tag))
    game = ale_id.split("/", 1)[1]
    return game.split("-v", 1)[0].lower()


def _infer_channels(shape: tuple[int, ...], *, fallback: int) -> int:
    if len(shape) >= 3:
        if int(shape[0]) in (1, 3, 4):
            return int(shape[0])
        if int(shape[-1]) in (1, 3, 4):
            return int(shape[-1])
    if len(shape) == 2 and int(fallback) > 1:
        return int(fallback)
    return 3


def _infer_image_size(shape: tuple[int, ...], *, default_size: int) -> int:
    if len(shape) >= 3:
        if int(shape[0]) in (1, 3, 4):
            return int(min(shape[1], shape[2]))
        if int(shape[-1]) in (1, 3, 4):
            return int(min(shape[0], shape[1]))
        return int(min(shape[-2], shape[-1]))
    if len(shape) == 2:
        return int(min(shape[0], shape[1]))
    return int(default_size)


def _infer_observation_spec(config: PufferPPOConfig, obs_np: np.ndarray) -> _ObservationSpec:
    obs_arr = np.asarray(obs_np)
    if obs_arr.ndim == 0:
        raise ValueError("Observation must include at least one dimension.")

    raw_shape = tuple(int(v) for v in (obs_arr.shape[1:] if obs_arr.ndim >= 2 else obs_arr.shape))
    if str(config.env_tag).startswith(("atari:", "ALE/")):
        return _ObservationSpec(mode="pixels", raw_shape=raw_shape, channels=4, image_size=84)

    backbone_name = str(config.backbone_name).strip().lower()
    looks_like_pixels = obs_arr.ndim >= 4 or (obs_arr.ndim >= 3 and "nature_cnn" in backbone_name)
    if looks_like_pixels:
        channels = _infer_channels(raw_shape, fallback=max(1, int(config.framestack)))
        image_size = _infer_image_size(raw_shape, default_size=84)
        return _ObservationSpec(mode="pixels", raw_shape=raw_shape, channels=channels, image_size=image_size)

    vector_dim = int(np.prod(raw_shape)) if raw_shape else 1
    return _ObservationSpec(mode="vector", raw_shape=raw_shape, vector_dim=vector_dim)


def _prepare_obs(obs_np: np.ndarray, *, obs_spec: _ObservationSpec, device: torch.device) -> torch.Tensor:
    obs_t = torch.as_tensor(obs_np)
    if obs_spec.mode == "pixels":
        channels = int(obs_spec.channels or 3)
        size = int(obs_spec.image_size or 84)
        obs_t = ensure_pixel_obs_format(obs_t, channels=channels, size=size, scale_float_255=False)
    else:
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        elif obs_t.ndim > 2:
            obs_t = obs_t.reshape(obs_t.shape[0], -1)
    return obs_t.to(device=device, dtype=torch.float32)


def _normalize_action_bounds(low: np.ndarray, high: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    low_v = np.asarray(low, dtype=np.float32).reshape(-1)
    high_v = np.asarray(high, dtype=np.float32).reshape(-1)
    if low_v.size == 1 and dim > 1:
        low_v = np.full((dim,), float(low_v.item()), dtype=np.float32)
    if high_v.size == 1 and dim > 1:
        high_v = np.full((dim,), float(high_v.item()), dtype=np.float32)
    if low_v.size != dim or high_v.size != dim:
        raise ValueError(f"Action bounds must match action dimension {dim}: low={low_v.shape}, high={high_v.shape}")

    low_v = np.where(np.isfinite(low_v), low_v, -1.0)
    high_v = np.where(np.isfinite(high_v), high_v, 1.0)
    high_v = np.maximum(high_v, low_v + 1e-6)
    return low_v, high_v


def _action_spec_from_space(action_space) -> _ActionSpec:
    if hasattr(action_space, "nvec"):
        raise ValueError("MultiDiscrete action spaces are not supported by ppo_puffer yet.")

    shape = tuple(int(v) for v in getattr(action_space, "shape", ()) or ())
    if hasattr(action_space, "n") and len(shape) == 0:
        return _ActionSpec(kind="discrete", dim=int(action_space.n))

    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise ValueError(f"Unsupported action space for ppo_puffer: {type(action_space)!r}")

    dim = int(np.prod(shape)) if shape else 1
    low, high = _normalize_action_bounds(np.asarray(action_space.low), np.asarray(action_space.high), dim=dim)
    return _ActionSpec(kind="continuous", dim=dim, low=low, high=high)


def _vector_backend_from_name(vector_mod, name: str):
    key = str(name).strip().lower()
    if key == "serial":
        return vector_mod.Serial
    if key == "multiprocessing":
        return vector_mod.Multiprocessing
    raise ValueError("vector_backend must be one of: serial, multiprocessing")


def _build_vector_kwargs(config: PufferPPOConfig, backend_cls, vector_mod) -> dict:
    kwargs = {}
    if backend_cls is vector_mod.Multiprocessing:
        if config.vector_num_workers is not None:
            kwargs["num_workers"] = int(config.vector_num_workers)
        if config.vector_batch_size is not None:
            kwargs["batch_size"] = int(config.vector_batch_size)
        kwargs["overwork"] = bool(config.vector_overwork)
    return kwargs


def _resolve_gym_env_name(env_tag: str) -> tuple[str, dict]:
    from problems.env_conf import get_env_conf

    env_conf = get_env_conf(str(env_tag))
    if getattr(env_conf, "gym_conf", None) is not None:
        return str(env_conf.env_name), dict(getattr(env_conf, "kwargs", {}) or {})

    tag = str(env_tag)
    if ":" not in tag and "/" not in tag and "-v" in tag:
        return tag, {}

    raise ValueError(f"Unsupported non-Atari env tag for ppo_puffer: {env_tag}")


def _make_gymnasium_env(*, env_name: str, env_kwargs: dict, render_mode="rgb_array", buf=None, seed=0):
    import gymnasium as gym
    import pufferlib
    import pufferlib.emulation

    kwargs = dict(env_kwargs)
    try:
        env = gym.make(env_name, render_mode=render_mode, **kwargs)
    except TypeError:
        env = gym.make(env_name, **kwargs)

    if isinstance(env.action_space, gym.spaces.Box):
        env = pufferlib.ClipAction(env)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=seed)


def _build_gymnasium_env_creator(config: PufferPPOConfig, pufferlib, *, resolve_gym_env_name_fn):
    env_name, env_kwargs = resolve_gym_env_name_fn(config.env_tag)
    _ = pufferlib  # already imported by caller; worker process imports directly in _make_gymnasium_env
    return functools.partial(_make_gymnasium_env, env_name=env_name, env_kwargs=env_kwargs)


def _make_vector_env(
    config: PufferPPOConfig,
    *,
    import_pufferlib_modules_fn,
    to_puffer_game_name_fn,
    resolve_gym_env_name_fn,
):
    pufferlib, puffer_vector, puffer_atari = import_pufferlib_modules_fn()
    backend_cls = _vector_backend_from_name(puffer_vector, config.vector_backend)
    kwargs = _build_vector_kwargs(config, backend_cls, puffer_vector)
    if str(config.env_tag).startswith(("atari:", "ALE/")):
        game_name = to_puffer_game_name_fn(config.env_tag)
        env_creator = puffer_atari.env_creator(game_name)
        env_kwargs = {"framestack": int(config.framestack)}
    else:
        env_creator = _build_gymnasium_env_creator(config, pufferlib, resolve_gym_env_name_fn=resolve_gym_env_name_fn)
        env_kwargs = {}

    return puffer_vector.make(
        env_creator,
        env_kwargs=env_kwargs,
        backend=backend_cls,
        num_envs=int(config.num_envs),
        seed=int(config.seed),
        **kwargs,
    )


def _build_plan(config: PufferPPOConfig) -> _TrainPlan:
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

    return _TrainPlan(
        num_envs=num_envs,
        num_steps=num_steps,
        batch_size=batch_size,
        minibatch_size=batch_size // num_minibatches,
        num_iterations=num_iterations,
    )


def _prepare_outputs(config: PufferPPOConfig) -> Path:
    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), dataclasses.asdict(config))
    return exp_dir / "metrics.jsonl"


def _resolve_backbone_name(config: PufferPPOConfig, obs_spec: _ObservationSpec) -> str:
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


def _build_model(config: PufferPPOConfig, obs_spec: _ObservationSpec, action_spec: _ActionSpec) -> _ActorCritic:
    input_dim = 64 if obs_spec.mode == "pixels" else int(obs_spec.vector_dim or 1)
    backbone_spec = BackboneSpec(
        name=_resolve_backbone_name(config, obs_spec),
        hidden_sizes=tuple(config.backbone_hidden_sizes),
        activation=str(config.backbone_activation),
        layer_norm=bool(config.backbone_layer_norm),
    )
    actor_head_spec = HeadSpec(
        hidden_sizes=tuple(config.actor_head_hidden_sizes),
        activation=str(config.head_activation),
    )
    critic_head_spec = HeadSpec(
        hidden_sizes=tuple(config.critic_head_hidden_sizes),
        activation=str(config.head_activation),
    )

    if bool(config.share_backbone):
        shared, feat_dim = build_backbone(backbone_spec, input_dim=input_dim)
        actor_backbone, critic_backbone = shared, shared
        actor_feat_dim, critic_feat_dim = feat_dim, feat_dim
    else:
        actor_backbone, actor_feat_dim = build_backbone(backbone_spec, input_dim=input_dim)
        critic_backbone, critic_feat_dim = build_backbone(backbone_spec, input_dim=input_dim)

    actor_head = build_mlp_head(actor_head_spec, input_dim=actor_feat_dim, output_dim=int(action_spec.dim))
    critic_head = build_mlp_head(critic_head_spec, input_dim=critic_feat_dim, output_dim=1)

    init_linear(actor_backbone, gain=0.5)
    init_linear(critic_backbone, gain=0.5)
    init_linear(actor_head, gain=0.01)
    init_linear(critic_head, gain=1.0)

    return _ActorCritic(
        actor_backbone=actor_backbone,
        critic_backbone=critic_backbone,
        actor_head=actor_head,
        critic_head=critic_head,
        action_spec=action_spec,
        log_std_init=float(getattr(config, "log_std_init", -0.5)),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_runtime(config: PufferPPOConfig, plan: _TrainPlan, device: torch.device, envs):
    next_obs_np, _ = envs.reset(seed=int(config.seed))
    obs_spec = _infer_observation_spec(config, next_obs_np)
    next_obs = _prepare_obs(next_obs_np, obs_spec=obs_spec, device=device)
    effective_num_envs = int(next_obs.shape[0])
    if effective_num_envs != plan.num_envs:
        raise ValueError(
            "Runtime num_envs mismatch: "
            f"config.num_envs={plan.num_envs}, env_batch={effective_num_envs}. "
            "For multiprocessing backend, ensure vector_batch_size matches num_envs."
        )

    action_spec = _action_spec_from_space(envs.single_action_space)
    model = _build_model(config, obs_spec=obs_spec, action_spec=action_spec).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.learning_rate), eps=1e-5)
    next_done = torch.zeros(plan.num_envs, dtype=torch.float32, device=device)

    obs_shape = tuple(next_obs.shape[1:])
    if action_spec.kind == "discrete":
        action_buf_shape = (plan.num_steps, plan.num_envs)
        action_dtype = torch.long
    else:
        action_buf_shape = (plan.num_steps, plan.num_envs, int(action_spec.dim))
        action_dtype = torch.float32

    buffer = _RolloutBuffer(
        obs=torch.zeros((plan.num_steps, plan.num_envs, *obs_shape), device=device),
        actions=torch.zeros(action_buf_shape, dtype=action_dtype, device=device),
        logprobs=torch.zeros((plan.num_steps, plan.num_envs), device=device),
        rewards=torch.zeros((plan.num_steps, plan.num_envs), device=device),
        dones=torch.zeros((plan.num_steps, plan.num_envs), device=device),
        values=torch.zeros((plan.num_steps, plan.num_envs), device=device),
    )

    state = _RuntimeState(
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
    return model, optimizer, obs_shape, buffer, state


def _update_episode_stats(state: _RuntimeState, infos: list | None) -> None:
    for item in infos or []:
        if isinstance(item, dict) and "episode_return" in item:
            state.last_episode_return = float(item["episode_return"])


def _collect_rollout(
    plan: _TrainPlan,
    model: _ActorCritic,
    envs,
    buffer: _RolloutBuffer,
    state: _RuntimeState,
    device: torch.device,
):
    for step in range(plan.num_steps):
        state.global_step += plan.num_envs
        buffer.obs[step] = state.next_obs
        buffer.dones[step] = state.next_done

        with torch.no_grad():
            action, logprob, _, value = model.get_action_and_value(state.next_obs)
        buffer.actions[step] = action
        buffer.logprobs[step] = logprob
        buffer.values[step] = value

        next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(action.detach().cpu().numpy())
        buffer.rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device).view(-1)
        done_np = np.logical_or(term_np, trunc_np)
        state.next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device).view(-1)
        state.next_obs = _prepare_obs(next_obs_np, obs_spec=state.obs_spec, device=device)
        _update_episode_stats(state, infos)


def _compute_advantages(
    plan: _TrainPlan,
    config: PufferPPOConfig,
    model: _ActorCritic,
    state: _RuntimeState,
    buffer: _RolloutBuffer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        next_value = model.get_value(state.next_obs)
        advantages = torch.zeros_like(buffer.rewards)
        lastgaelam = torch.zeros(plan.num_envs, dtype=torch.float32, device=device)
        for t in reversed(range(plan.num_steps)):
            if t == plan.num_steps - 1:
                next_nonterminal = 1.0 - state.next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - buffer.dones[t + 1]
                next_values = buffer.values[t + 1]
            delta = buffer.rewards[t] + float(config.gamma) * next_values * next_nonterminal - buffer.values[t]
            lastgaelam = delta + float(config.gamma) * float(config.gae_lambda) * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + buffer.values
    return advantages, returns


def _flatten_batch(
    plan: _TrainPlan,
    buffer: _RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    obs_shape: tuple[int, ...],
) -> _FlatBatch:
    if buffer.actions.ndim == 2:
        flat_actions = buffer.actions.reshape(-1)
    else:
        flat_actions = buffer.actions.reshape(plan.batch_size, *buffer.actions.shape[2:])

    return _FlatBatch(
        obs=buffer.obs.reshape((plan.batch_size, *obs_shape)),
        actions=flat_actions,
        logprobs=buffer.logprobs.reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        values=buffer.values.reshape(-1),
    )


def _compute_losses(
    config: PufferPPOConfig,
    batch: _FlatBatch,
    mb_inds: np.ndarray,
    ratio: torch.Tensor,
    entropy: torch.Tensor,
    newvalue: torch.Tensor,
):
    mb_advantages = batch.advantages[mb_inds]
    if bool(config.norm_adv):
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - float(config.clip_coef), 1.0 + float(config.clip_coef))
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    newvalue = newvalue.view(-1)
    if bool(config.clip_vloss):
        v_loss = _clipped_value_loss(config, batch, mb_inds, newvalue)
    else:
        v_loss = 0.5 * ((newvalue - batch.returns[mb_inds]) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - float(config.ent_coef) * entropy_loss + float(config.vf_coef) * v_loss
    return loss


def _clipped_value_loss(
    config: PufferPPOConfig,
    batch: _FlatBatch,
    mb_inds: np.ndarray,
    newvalue: torch.Tensor,
) -> torch.Tensor:
    v_loss_unclipped = (newvalue - batch.returns[mb_inds]) ** 2
    v_clipped = batch.values[mb_inds] + torch.clamp(
        newvalue - batch.values[mb_inds],
        -float(config.clip_coef),
        float(config.clip_coef),
    )
    v_loss_clipped = (v_clipped - batch.returns[mb_inds]) ** 2
    return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()


def _ppo_update(
    config: PufferPPOConfig,
    plan: _TrainPlan,
    model: _ActorCritic,
    optimizer: optim.Optimizer,
    batch: _FlatBatch,
    b_inds: np.ndarray,
) -> _UpdateStats:
    approx_kl_value = 0.0
    clipfrac_values: list[float] = []

    for _epoch in range(int(config.update_epochs)):
        np.random.shuffle(b_inds)
        for start in range(0, plan.batch_size, plan.minibatch_size):
            mb_inds = b_inds[start : start + plan.minibatch_size]
            _, newlogprob, entropy, newvalue = model.get_action_and_value(batch.obs[mb_inds], action=batch.actions[mb_inds])
            logratio = newlogprob - batch.logprobs[mb_inds]
            ratio = logratio.exp()
            approx_kl_value = _track_kl_clip(logratio, ratio, config, clipfrac_values)

            loss = _compute_losses(config, batch, mb_inds, ratio, entropy, newvalue)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            optimizer.step()

        if config.target_kl is not None and approx_kl_value > float(config.target_kl):
            break

    clipfrac_mean = float(np.mean(clipfrac_values)) if clipfrac_values else 0.0
    return _UpdateStats(approx_kl=float(approx_kl_value), clipfrac_mean=float(clipfrac_mean))


def _track_kl_clip(
    logratio: torch.Tensor,
    ratio: torch.Tensor,
    config: PufferPPOConfig,
    clipfrac_values: list[float],
) -> float:
    with torch.no_grad():
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > float(config.clip_coef)).float().mean().item()
        clipfrac_values.append(float(clipfrac))
        return float(approx_kl.item())


def _build_eval_env_conf(config: PufferPPOConfig, *, obs_spec: _ObservationSpec):
    return puffer_eval_config.build_eval_env_conf(
        config,
        obs_mode=obs_spec.mode,
        is_atari_env_tag_fn=lambda tag: str(tag).startswith(("atari:", "ALE/")),
        resolve_gym_env_name_fn=_resolve_gym_env_name,
    )


def _run_training(
    config: PufferPPOConfig,
    plan: _TrainPlan,
    device: torch.device,
    metrics_path: Path,
    envs,
    *,
    build_eval_env_conf_fn,
) -> TrainResult:
    model, optimizer, obs_shape, buffer, state = _init_runtime(config, plan, device, envs)
    checkpointing = __import__("rl.checkpointing", fromlist=["CheckpointManager"])
    checkpoint_manager = checkpointing.CheckpointManager(exp_dir=metrics_path.parent)
    puffer_ckpt.restore_checkpoint_if_requested(
        config,
        plan,
        model,
        optimizer,
        state,
        device=device,
    )
    from rl import logger as rl_logger

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
        _collect_rollout(plan, model, envs, buffer, state, device)
        advantages, returns = _compute_advantages(plan, config, model, state, buffer, device)
        batch = _flatten_batch(plan, buffer, advantages, returns, obs_shape)
        update_stats = _ppo_update(config, plan, model, optimizer, batch, b_inds)
        puffer_eval.maybe_eval_and_update_state(
            config,
            model,
            state,
            iteration=iteration,
            device=device,
            build_eval_env_conf_fn=build_eval_env_conf_fn,
            prepare_obs_fn=_prepare_obs,
        )
        metric = puffer_metrics._metric_payload(iteration, plan, optimizer, state, update_stats, batch)
        puffer_metrics._append_metrics_line(metrics_path, metric)
        puffer_metrics._log_iteration(config, metric)
        puffer_ckpt.maybe_save_periodic_checkpoint(
            config,
            checkpoint_manager,
            model,
            optimizer,
            state,
            iteration=iteration,
        )

    best_return = state.best_return
    if best_return == -float("inf"):
        best_return = float("nan")
    total_time = time.time() - state.start_time
    rl_logger.log_run_footer(float(best_return), int(plan.num_iterations), float(total_time), algo_name="ppo")
    final_iteration = int(max(state.start_iteration, plan.num_iterations))
    puffer_ckpt.save_final_checkpoint(
        config,
        checkpoint_manager,
        model,
        optimizer,
        state,
        iteration=final_iteration,
    )
    puffer_eval.maybe_render_videos(
        config,
        model,
        state,
        exp_dir=metrics_path.parent,
        device=device,
        build_eval_env_conf_fn=build_eval_env_conf_fn,
        prepare_obs_fn=_prepare_obs,
    )
    return TrainResult(
        best_return=float(best_return),
        last_eval_return=float(state.last_eval_return),
        last_heldout_return=puffer_metrics._as_optional_finite(state.last_heldout_return),
        num_iterations=int(plan.num_iterations),
    )


def train_ppo_puffer_impl(
    config: PufferPPOConfig,
    *,
    make_vector_env_fn=None,
    build_eval_env_conf_fn=None,
) -> TrainResult:
    puffer_eval.validate_eval_config(config)
    device = _resolve_device(config.device)
    _seed_everything(int(config.seed))
    plan = _build_plan(config)
    metrics_path = _prepare_outputs(config)
    if make_vector_env_fn is None:
        compat = __import__("rl.pufferlib_compat", fromlist=["import_pufferlib_modules"])
        make_vector_env = functools.partial(
            _make_vector_env,
            import_pufferlib_modules_fn=compat.import_pufferlib_modules,
            to_puffer_game_name_fn=_to_puffer_game_name,
            resolve_gym_env_name_fn=_resolve_gym_env_name,
        )
    else:
        make_vector_env = make_vector_env_fn
    build_eval_env_conf = build_eval_env_conf_fn or _build_eval_env_conf

    with closing(make_vector_env(config)) as envs:
        return _run_training(
            config,
            plan,
            device,
            metrics_path,
            envs,
            build_eval_env_conf_fn=build_eval_env_conf,
        )
