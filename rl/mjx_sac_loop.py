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
) -> dict[str, float | int]:
    step = int(iteration) * int(frames_per_iter)
    return {
        "iter": int(iteration),
        "step": step,
        "elapsed": float(elapsed),
        "fps": step / elapsed if elapsed > 0 else float("nan"),
        "ret_rollout": metrics["rollout_return"],
        "ep_ret": metrics["ep_ret"],
        "ep_len": metrics["ep_len"],
        "ret_best": float(ret_best),
        "rew": metrics["rollout_reward"],
        "done_frac": metrics["done_fraction"],
        "actor": metrics["loss_actor"],
        "critic": metrics["loss_critic"],
        "alpha": metrics["alpha_value"],
        "alpha_loss": metrics["loss_alpha"],
        "iter_dt": float(iter_dt),
    }
