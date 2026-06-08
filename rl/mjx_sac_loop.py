from __future__ import annotations

from rl.mjx_eval import make_mjx_eval_step
from rl.mjx_ppo import _policy


def sac_eval_action(params, jnp, norm_obs, runtime):
    mean, _std = _policy(params, jnp, norm_obs)
    action = jnp.tanh(mean)
    return runtime.low + 0.5 * (action + 1.0) * (runtime.high - runtime.low)


def sac_eval_args(state):
    return state.actor, state.obs_rms


def make_sac_eval_step(config, runtime):
    return make_mjx_eval_step(config, runtime, sac_eval_action)


def make_sac_result(result_cls):
    def result(best_return: float, last_return: float, iterations: int, frames: int):
        return result_cls(
            best_return=best_return,
            last_rollout_return=last_return,
            num_steps=iterations * frames,
        )

    return result


def sac_iter_record(
    *,
    iteration: int,
    frames_per_iter: int,
    elapsed: float,
    iter_dt: float,
    metrics: dict[str, float],
    ret_best: float,
    ret_eval: float | None = None,
    eval_dt: float | None = None,
) -> dict[str, float | int]:
    from rl.mjx_metrics import build_iter_record

    return build_iter_record(
        algo_name="sac",
        iteration=iteration,
        frames_per_iter=frames_per_iter,
        elapsed=elapsed,
        iter_dt=iter_dt,
        metrics=metrics,
        ret_best=ret_best,
        ret_eval=ret_eval,
        eval_dt=eval_dt,
    )
