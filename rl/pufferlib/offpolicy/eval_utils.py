from __future__ import annotations

import dataclasses
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.core.actor_state import capture_backbone_head_snapshot, restore_backbone_head_snapshot, use_backbone_head_snapshot
from rl.core.episode_rollout import collect_denoised_trajectory, evaluate_for_best
from rl.core.offpolicy_eval import evaluate_heldout_with_best_actor, update_best_actor_if_improved
from rl.core.offpolicy_metrics import build_eval_metric_record, build_log_eval_iteration_kwargs
from rl.core.progress import due_mark
from rl.eval_noise import build_eval_plan

from ... import logger as rl_logger
from .env_utils import prepare_obs_np


def capture_actor_state(modules):
    return capture_backbone_head_snapshot(
        modules.actor_backbone, modules.actor_head, log_std=getattr(modules, "log_std", None), state_to_cpu=True, log_std_to_cpu=True, log_std_format="tensor"
    )


def _restore_actor_state(modules, snapshot, *, device: torch.device) -> None:
    restore_backbone_head_snapshot(modules.actor_backbone, modules.actor_head, snapshot, log_std=getattr(modules, "log_std", None), device=device)


@contextmanager
def use_actor_state(modules, actor_state, *, device: torch.device):
    with use_backbone_head_snapshot(
        modules.actor_backbone,
        modules.actor_head,
        actor_state,
        log_std=getattr(modules, "log_std", None),
        device=device,
        state_to_cpu=True,
        log_std_to_cpu=True,
        log_std_format="tensor",
    ):
        yield


@dataclasses.dataclass
class TrainState:
    global_step: int = 0
    total_updates: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict[str, Any] | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None
    last_loss_actor: float = float("nan")
    last_loss_critic: float = float("nan")
    last_loss_alpha: float = float("nan")
    start_time: float = 0.0
    eval_mark: int = 0
    log_mark: int = 0
    ckpt_mark: int = 0


class OffPolicyEvalPolicy:
    def __init__(self, modules: Any, obs_spec: Any, *, device: torch.device):
        self._modules = modules
        self._obs_spec = obs_spec
        self._device = device

    def __call__(self, state: np.ndarray) -> np.ndarray:
        obs_np = prepare_obs_np(state, obs_spec=self._obs_spec)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            action = self._modules.actor.act(obs_t)
        return np.asarray(action.squeeze(0).detach().cpu().numpy(), dtype=np.float32)


def evaluate_actor(config: Any, env: Any, modules: Any, obs_spec: Any, *, device: torch.device, eval_seed: int) -> float:
    policy = OffPolicyEvalPolicy(modules, obs_spec, device=device)
    traj, _ = collect_denoised_trajectory(env.env_conf, policy, num_denoise=config.num_denoise_eval, i_noise=int(eval_seed))
    return float(traj.rreturn)


def evaluate_heldout_if_enabled(
    config: Any,
    env: Any,
    modules: Any,
    obs_spec: Any,
    *,
    device: torch.device,
    heldout_i_noise: int,
    best_actor_state: dict[str, Any] | None = None,
    with_actor_state_fn=None,
) -> float | None:
    if config.num_denoise_passive_eval is None:
        return None
    if best_actor_state is not None and with_actor_state_fn is not None:
        policy = OffPolicyEvalPolicy(modules, obs_spec, device=device)
        return evaluate_heldout_with_best_actor(
            best_actor_state=best_actor_state,
            num_denoise_passive_eval=config.num_denoise_passive_eval,
            heldout_i_noise=int(heldout_i_noise),
            with_actor_state=with_actor_state_fn,
            evaluate_for_best=evaluate_for_best,
            eval_env_conf=env.env_conf,
            eval_policy=policy,
        )
    policy = OffPolicyEvalPolicy(modules, obs_spec, device=device)
    return float(evaluate_for_best(env.env_conf, policy, config.num_denoise_passive_eval, i_noise=int(heldout_i_noise)))


def append_eval_metric(path, state: TrainState, *, step: int) -> None:
    now = float(time.time())
    record = build_eval_metric_record(
        step=int(step),
        eval_return=float(state.last_eval_return),
        heldout_return=state.last_heldout_return,
        best_return=float(state.best_return),
        loss_actor=float(state.last_loss_actor),
        loss_critic=float(state.last_loss_critic),
        loss_alpha=float(state.last_loss_alpha),
        total_updates=int(state.total_updates),
        started_at=float(state.start_time),
        now=now,
    )
    rl_logger.append_metrics(path, record)


def log_if_due(config: Any, state: TrainState, *, step: int, frames_per_batch: int) -> None:
    mark = due_mark(step, config.log_interval_steps, state.log_mark)
    if mark is None:
        return
    state.log_mark = int(mark)
    now = float(time.time())
    kwargs = build_log_eval_iteration_kwargs(
        step=int(step),
        frames_per_batch=int(frames_per_batch),
        started_at=float(state.start_time),
        now=now,
        eval_return=float(state.last_eval_return),
        heldout_return=state.last_heldout_return,
        best_return=float(state.best_return),
        loss_actor=float(state.last_loss_actor),
        loss_critic=float(state.last_loss_critic),
        loss_alpha=float(state.last_loss_alpha),
    )
    rl_logger.log_eval_iteration(**kwargs)


def maybe_eval(config: Any, env: Any, modules: Any, obs_spec: Any, state: TrainState, *, device: torch.device):
    mark = due_mark(state.global_step, config.eval_interval_steps, state.eval_mark)
    if mark is None:
        return
    state.eval_mark = int(mark)
    due_step = int(mark * int(config.eval_interval_steps))
    plan = build_eval_plan(
        current=int(due_step),
        interval=int(config.eval_interval_steps),
        seed=int(config.seed),
        eval_seed_base=config.eval_seed_base,
        eval_noise_mode=config.eval_noise_mode,
    )
    state.last_eval_return = evaluate_actor(config, env, modules, obs_spec, device=device, eval_seed=int(plan.eval_seed))
    state.best_return, state.best_actor_state, _ = update_best_actor_if_improved(
        eval_return=float(state.last_eval_return),
        best_return=float(state.best_return),
        best_actor_state=state.best_actor_state,
        capture_actor_state=lambda: capture_actor_state(modules),
    )
    state.last_heldout_return = evaluate_heldout_if_enabled(
        config,
        env,
        modules,
        obs_spec,
        device=device,
        heldout_i_noise=int(plan.heldout_i_noise),
        best_actor_state=state.best_actor_state,
        with_actor_state_fn=lambda snapshot: use_actor_state(modules, snapshot, device=device),
    )


def render_videos_if_enabled(config: Any, env: Any, modules: Any, obs_spec: Any, *, device: torch.device) -> None:
    if not bool(config.video_enable):
        return
    from common.video import render_policy_videos

    policy = OffPolicyEvalPolicy(modules, obs_spec, device=device)
    seed_base = int(config.video_seed_base) if config.video_seed_base is not None else int(env.problem_seed + 10000)
    render_policy_videos(
        env.env_conf,
        policy,
        video_dir=Path(config.exp_dir) / "videos",
        video_prefix=str(config.video_prefix),
        num_episodes=int(config.video_num_episodes),
        num_video_episodes=int(config.video_num_video_episodes),
        episode_selection=str(config.video_episode_selection),
        seed_base=seed_base,
    )


SacEvalPolicy = OffPolicyEvalPolicy
