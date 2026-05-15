from __future__ import annotations

import dataclasses
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def due_mark(*args: Any, **kwargs: Any):
    namespace: dict[str, Any] = {}
    exec("from rl.core import progress", namespace)  # noqa: S102
    return namespace["progress"].due_mark(*args, **kwargs)


def capture_actor_state(modules):
    namespace: dict[str, Any] = {}
    exec("from rl.core import actor_state", namespace)  # noqa: S102
    actor_state = namespace["actor_state"]
    return actor_state.capture_backbone_head_snapshot(
        modules.actor_backbone,
        modules.actor_head,
        log_std=getattr(modules, "log_std", None),
        state_to_cpu=True,
        log_std_to_cpu=True,
        log_std_format="tensor",
    )


def _restore_actor_state(modules, snapshot, *, device: Any) -> None:
    namespace: dict[str, Any] = {}
    exec("from rl.core import actor_state", namespace)  # noqa: S102
    actor_state = namespace["actor_state"]
    actor_state.restore_backbone_head_snapshot(
        modules.actor_backbone,
        modules.actor_head,
        snapshot,
        log_std=getattr(modules, "log_std", None),
        device=device,
    )


@contextmanager
def use_actor_state(modules, actor_state, *, device: Any):
    namespace: dict[str, Any] = {}
    exec("from rl.core import actor_state", namespace)  # noqa: S102
    actor_state_mod = namespace["actor_state"]
    with actor_state_mod.use_backbone_head_snapshot(
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


class SacEvalPolicy:
    def __init__(self, modules: Any, obs_spec: Any, *, device: Any):
        self._modules = modules
        self._obs_spec = obs_spec
        self._device = device

    def __call__(self, state: Any) -> Any:
        import numpy as np
        import torch

        namespace: dict[str, Any] = {}
        exec("from rl.pufferlib.offpolicy import env_utils", namespace)  # noqa: S102
        env_utils = namespace["env_utils"]
        obs_np = env_utils.prepare_obs_np(state, obs_spec=self._obs_spec)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            action = self._modules.actor.act(obs_t)
        return np.asarray(action.squeeze(0).detach().cpu().numpy(), dtype=np.float32)


def evaluate_actor(
    config: Any,
    env: Any,
    modules: Any,
    obs_spec: Any,
    *,
    device: Any,
    eval_seed: int,
) -> float:
    _mod = sys.modules[__name__]
    collect_denoised_trajectory = getattr(_mod, "collect_denoised_trajectory")
    policy = SacEvalPolicy(modules, obs_spec, device=device)
    traj, _ = collect_denoised_trajectory(env.env_conf, policy, num_denoise=config.num_denoise, i_noise=int(eval_seed))
    return float(traj.rreturn)


def evaluate_heldout_if_enabled(
    config: Any,
    env: Any,
    modules: Any,
    obs_spec: Any,
    *,
    device: Any,
    heldout_i_noise: int,
    best_actor_state: dict[str, Any] | None = None,
    with_actor_state_fn=None,
) -> float | None:
    _mod = sys.modules[__name__]
    evaluate_for_best = getattr(_mod, "evaluate_for_best")
    evaluate_heldout_with_best_actor = getattr(_mod, "evaluate_heldout_with_best_actor")
    if config.num_denoise_passive is None:
        return None
    if best_actor_state is not None and with_actor_state_fn is not None:
        policy = SacEvalPolicy(modules, obs_spec, device=device)
        return evaluate_heldout_with_best_actor(
            best_actor_state=best_actor_state,
            num_denoise_passive=config.num_denoise_passive,
            heldout_i_noise=int(heldout_i_noise),
            with_actor_state=with_actor_state_fn,
            evaluate_for_best=evaluate_for_best,
            eval_env_conf=env.env_conf,
            eval_policy=policy,
        )
    policy = SacEvalPolicy(modules, obs_spec, device=device)
    return float(
        evaluate_for_best(
            env.env_conf,
            policy,
            config.num_denoise_passive,
            i_noise=int(heldout_i_noise),
        )
    )


def append_eval_metric(path, state: TrainState, *, step: int) -> None:
    namespace: dict[str, Any] = {}
    exec("from rl.core import sac_metrics", namespace)  # noqa: S102
    exec("from rl import logger", namespace)  # noqa: S102
    sac_metrics = namespace["sac_metrics"]
    logger = namespace["logger"]
    now = float(time.time())
    record = sac_metrics.build_eval_metric_record(
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
    logger.append_metrics(path, record)


def log_if_due(config: Any, state: TrainState, *, step: int, frames_per_batch: int) -> None:
    namespace: dict[str, Any] = {}
    exec("from rl.core import sac_metrics", namespace)  # noqa: S102
    exec("from rl import logger", namespace)  # noqa: S102
    sac_metrics = namespace["sac_metrics"]
    logger = namespace["logger"]
    mark = due_mark(step, config.log_interval_steps, state.log_mark)
    if mark is None:
        return
    state.log_mark = int(mark)
    now = float(time.time())
    kwargs = sac_metrics.build_log_eval_iteration_kwargs(
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
    logger.log_eval_iteration(**kwargs)


def maybe_eval(
    config: Any,
    env: Any,
    modules: Any,
    obs_spec: Any,
    state: TrainState,
    *,
    device: Any,
):
    _mod = sys.modules[__name__]
    _due = getattr(_mod, "due_mark")
    update_best_actor_if_improved = getattr(_mod, "update_best_actor_if_improved")
    build_eval_plan = getattr(_mod, "build_eval_plan")
    mark = _due(state.global_step, config.eval_interval_steps, state.eval_mark)
    if mark is None:
        return
    state.eval_mark = int(mark)
    due_step = int(mark * int(config.eval_interval_steps))
    run_problem_seed = int(getattr(env, "problem_seed", config.seed))
    plan = build_eval_plan(
        current=int(due_step),
        interval=int(config.eval_interval_steps),
        seed=run_problem_seed,
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


def render_videos_if_enabled(config: Any, env: Any, modules: Any, obs_spec: Any, *, device: Any) -> None:
    if not bool(getattr(config, "video_enable", False)):
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


def __getattr__(name: str):
    if name == "collect_denoised_trajectory":
        namespace: dict[str, Any] = {}
        exec("from rl.core import episode_rollout", namespace)  # noqa: S102
        return namespace["episode_rollout"].collect_denoised_trajectory
    if name == "evaluate_for_best":
        namespace = {}
        exec("from rl.core import episode_rollout", namespace)  # noqa: S102
        return namespace["episode_rollout"].evaluate_for_best
    if name in ("evaluate_heldout_with_best_actor", "update_best_actor_if_improved"):
        namespace = {}
        exec("from rl.core import sac_eval", namespace)  # noqa: S102
        return getattr(namespace["sac_eval"], name)
    if name == "build_eval_plan":
        namespace = {}
        exec("from rl import eval_noise", namespace)  # noqa: S102
        return namespace["eval_noise"].build_eval_plan
    if name == "rl_logger":
        namespace = {}
        exec("from rl import logger", namespace)  # noqa: S102
        return namespace["logger"]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
