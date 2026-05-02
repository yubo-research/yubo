from __future__ import annotations

import sys
from typing import Any


def _offp():
    _ns: dict[str, Any] = {}
    exec("from rl.pufferlib.offpolicy import eval_utils as _m", _ns)  # noqa: S102
    return _ns["_m"]


def _facade():
    return sys.modules["rl.pufferlib.sac.eval_utils"]


def evaluate_actor(config: Any, env: Any, modules: Any, obs_spec: Any, *, device, eval_seed: int) -> float:
    fac = _facade()
    collect_denoised_trajectory = fac.collect_denoised_trajectory
    impl = _offp()
    policy = impl.SacEvalPolicy(modules, obs_spec, device=device)
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
    fac = _facade()
    evaluate_for_best = fac.evaluate_for_best
    impl = _offp()
    if config.num_denoise_passive is None:
        return None
    if best_actor_state is not None and with_actor_state_fn is not None:
        policy = impl.SacEvalPolicy(modules, obs_spec, device=device)
        return impl.evaluate_heldout_with_best_actor(
            best_actor_state=best_actor_state,
            num_denoise_passive=config.num_denoise_passive,
            heldout_i_noise=int(heldout_i_noise),
            with_actor_state=with_actor_state_fn,
            evaluate_for_best=evaluate_for_best,
            eval_env_conf=env.env_conf,
            eval_policy=policy,
        )
    policy = impl.SacEvalPolicy(modules, obs_spec, device=device)
    return float(
        evaluate_for_best(
            env.env_conf,
            policy,
            config.num_denoise_passive,
            i_noise=int(heldout_i_noise),
        )
    )


def maybe_eval(config: Any, env: Any, modules: Any, obs_spec: Any, state, *, device) -> None:
    fac = _facade()
    impl = _offp()
    due_mark = impl.due_mark
    capture_actor_state = impl.capture_actor_state
    use_actor_state = impl.use_actor_state
    build_eval_plan = fac.build_eval_plan
    evaluate_actor_fn = fac.evaluate_actor
    evaluate_heldout_fn = fac.evaluate_heldout_if_enabled
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
    state.last_eval_return = evaluate_actor_fn(config, env, modules, obs_spec, device=device, eval_seed=int(plan.eval_seed))
    state.best_return, state.best_actor_state, _ = impl.update_best_actor_if_improved(
        eval_return=float(state.last_eval_return),
        best_return=float(state.best_return),
        best_actor_state=state.best_actor_state,
        capture_actor_state=lambda: capture_actor_state(modules),
    )
    state.last_heldout_return = evaluate_heldout_fn(
        config,
        env,
        modules,
        obs_spec,
        device=device,
        heldout_i_noise=int(plan.heldout_i_noise),
        best_actor_state=state.best_actor_state,
        with_actor_state_fn=lambda snapshot: use_actor_state(modules, snapshot, device=device),
    )


def __getattr__(name: str):
    impl = _offp()
    if name == "rl_logger":
        _ns: dict[str, Any] = {}
        exec("import rl.logger as _f", _ns)  # noqa: S102
        return _ns["_f"]
    if name == "build_eval_plan":
        _ns: dict[str, Any] = {}
        exec("from rl.eval_noise import build_eval_plan as _f", _ns)  # noqa: S102
        return _ns["_f"]
    if name in ("collect_denoised_trajectory", "evaluate_for_best"):
        _ns = {}
        exec(f"from rl.core.episode_rollout import {name} as _f", _ns)  # noqa: S102
        return _ns["_f"]
    if hasattr(impl, name):
        return getattr(impl, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
