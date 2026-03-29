from __future__ import annotations

from typing import Any

from rl.core.episode_rollout import collect_denoised_trajectory, evaluate_for_best
from rl.eval_noise import build_eval_plan
from rl.pufferlib.offpolicy import eval_utils as _impl

TrainState = _impl.TrainState
SacEvalPolicy = _impl.SacEvalPolicy
capture_actor_state = _impl.capture_actor_state
use_actor_state = _impl.use_actor_state
append_eval_metric = _impl.append_eval_metric
due_mark = _impl.due_mark
log_if_due = _impl.log_if_due
rl_logger = _impl.rl_logger


def evaluate_actor(config: Any, env: Any, modules: Any, obs_spec: Any, *, device, eval_seed: int) -> float:
    policy = SacEvalPolicy(modules, obs_spec, device=device)
    traj, _ = collect_denoised_trajectory(env.env_conf, policy, num_denoise=config.num_denoise, i_noise=int(eval_seed))
    return float(traj.rreturn)


def evaluate_heldout_if_enabled(
    config: Any,
    env: Any,
    modules: Any,
    obs_spec: Any,
    *,
    device,
    heldout_i_noise: int,
    best_actor_state: dict[str, Any] | None = None,
    with_actor_state_fn=None,
) -> float | None:
    if config.num_denoise_passive is None:
        return None
    if best_actor_state is not None and with_actor_state_fn is not None:
        policy = SacEvalPolicy(modules, obs_spec, device=device)
        return _impl.evaluate_heldout_with_best_actor(
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


def maybe_eval(config: Any, env: Any, modules: Any, obs_spec: Any, state: TrainState, *, device) -> None:
    mark = due_mark(state.global_step, config.eval_interval_steps, state.eval_mark)
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
    state.best_return, state.best_actor_state, _ = _impl.update_best_actor_if_improved(
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


render_videos_if_enabled = _impl.render_videos_if_enabled

__all__ = [
    "SacEvalPolicy",
    "TrainState",
    "append_eval_metric",
    "build_eval_plan",
    "capture_actor_state",
    "collect_denoised_trajectory",
    "due_mark",
    "evaluate_actor",
    "evaluate_for_best",
    "evaluate_heldout_if_enabled",
    "log_if_due",
    "maybe_eval",
    "render_videos_if_enabled",
    "rl_logger",
    "use_actor_state",
]
