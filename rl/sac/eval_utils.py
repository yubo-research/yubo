from __future__ import annotations

import dataclasses
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.core import eval as rl_eval
from rl.core import sac_metrics
from rl.core.episode_rollout import best, denoise
from rl.core.progress import due_mark

from .. import logger as rl_logger
from ..offpolicy import model_utils as offpolicy_model_utils
from .env_utils import prepare_obs_np

build_eval_plan = rl_eval.plan

capture_actor_state = offpolicy_model_utils.capture_actor_state


def _restore_actor_state(modules, snapshot, *, device: torch.device) -> None:
    offpolicy_model_utils.restore_actor_state(modules, snapshot)


@contextmanager
def use_actor_state(modules, actor_state, *, device: torch.device):
    with offpolicy_model_utils.use_actor_state(modules, actor_state):
        yield


def _loss_to_float(v: float | torch.Tensor) -> float:
    """Convert loss to float; defers .item() sync until logging (S7)."""
    if isinstance(v, torch.Tensor):
        return float(v.item())
    return float(v)


@dataclasses.dataclass
class TrainState:
    global_step: int = 0
    total_updates: int = 0
    best_return: float = -float("inf")
    best_actor_state: dict[str, Any] | None = None
    last_eval_return: float = float("nan")
    last_heldout_return: float | None = None
    last_loss_actor: float | torch.Tensor = float("nan")
    last_loss_critic: float | torch.Tensor = float("nan")
    last_loss_alpha: float | torch.Tensor = float("nan")
    start_time: float = 0.0
    eval_mark: int = 0
    log_mark: int = 0
    ckpt_mark: int = 0


class SacEvalPolicy:
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


def heldout(
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
    if config.num_denoise_passive is None:
        return None
    policy = SacEvalPolicy(modules, obs_spec, device=device)
    if best_actor_state is not None and with_actor_state_fn is not None:
        return rl_eval.heldout(
            best_actor_state=best_actor_state,
            num_denoise_passive=config.num_denoise_passive,
            heldout_i_noise=int(heldout_i_noise),
            with_actor_state=with_actor_state_fn,
            best=best,
            eval_env_conf=env.env_conf,
            eval_policy=policy,
        )
    return float(
        best(
            env.env_conf,
            policy,
            config.num_denoise_passive,
            i_noise=int(heldout_i_noise),
        )
    )


def append_eval_metric(path, state: TrainState, *, step: int) -> None:
    now = float(time.time())
    rec = sac_metrics.record(
        step=int(step),
        eval_return=float(state.last_eval_return),
        heldout_return=state.last_heldout_return,
        best_return=float(state.best_return),
        loss_actor=_loss_to_float(state.last_loss_actor),
        loss_critic=_loss_to_float(state.last_loss_critic),
        loss_alpha=_loss_to_float(state.last_loss_alpha),
        total_updates=int(state.total_updates),
        started_at=float(state.start_time),
        now=now,
    )
    rl_logger.append_metrics(path, rec)


def log_if_due(config: Any, state: TrainState, *, step: int, frames_per_batch: int) -> None:
    mark = due_mark(step, config.log_interval_steps, state.log_mark)
    if mark is None:
        return
    state.log_mark = int(mark)
    now = float(time.time())
    line = sac_metrics.log(
        step=int(step),
        frames_per_batch=int(frames_per_batch),
        started_at=float(state.start_time),
        now=now,
        eval_return=float(state.last_eval_return),
        heldout_return=state.last_heldout_return,
        best_return=float(state.best_return),
        loss_actor=_loss_to_float(state.last_loss_actor),
        loss_critic=_loss_to_float(state.last_loss_critic),
        loss_alpha=_loss_to_float(state.last_loss_alpha),
    )
    rl_logger.log_eval_iteration(**line)


def maybe_eval(
    config: Any,
    env: Any,
    modules: Any,
    obs_spec: Any,
    state: TrainState,
    *,
    device: torch.device,
):
    mark = due_mark(state.global_step, config.eval_interval_steps, state.eval_mark)
    if mark is None:
        return
    state.eval_mark = int(mark)
    step_i = int(mark * int(config.eval_interval_steps))
    seed = int(getattr(env, "problem_seed", config.seed))
    rl_eval.run(
        current=step_i,
        interval=int(config.eval_interval_steps),
        seed=seed,
        eval_seed_base=config.eval_seed_base,
        eval_noise_mode=config.eval_noise_mode,
        state=state,
        evaluate_actor=lambda *, eval_seed: float(
            denoise(
                env.env_conf,
                SacEvalPolicy(modules, obs_spec, device=device),
                num_denoise=config.num_denoise,
                i_noise=int(eval_seed),
            )[0].rreturn
        ),
        capture_actor_state=lambda: capture_actor_state(modules),
        evaluate_heldout=lambda *, best_actor_state, heldout_i_noise: heldout(
            config,
            env,
            modules,
            obs_spec,
            device=device,
            heldout_i_noise=heldout_i_noise,
            best_actor_state=best_actor_state,
            with_actor_state_fn=lambda snapshot: use_actor_state(modules, snapshot, device=device),
        ),
    )


def render_videos_if_enabled(config: Any, env: Any, modules: Any, obs_spec: Any, *, device: torch.device) -> None:
    if not bool(config.video_enable):
        return
    from common.video import render_policy_videos

    policy = SacEvalPolicy(modules, obs_spec, device=device)
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
