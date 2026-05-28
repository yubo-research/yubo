from __future__ import annotations

from typing import Any, NamedTuple

from problems.jax_env_core import _RewardRunningMeanStd, _RunningMeanStd


class _TrainState(NamedTuple):
    iteration: Any
    params: dict[str, Any]
    opt_state: Any
    obs_rms: _RunningMeanStd
    reward_rms: _RewardRunningMeanStd
    discounted_return: Any
    obs: Any
    env_state: Any
    running_return: Any
    running_length: Any
    key: Any


class _AgentState(NamedTuple):
    iteration: Any
    params: dict[str, Any]
    opt_state: Any
    obs_rms: _RunningMeanStd
    reward_rms: _RewardRunningMeanStd


class MJXPPOResult(NamedTuple):
    best_return: float
    last_rollout_return: float
    num_iterations: int


def _checkpoint_fn(state: _TrainState) -> _AgentState:
    return _AgentState(
        iteration=state.iteration,
        params=state.params,
        opt_state=state.opt_state,
        obs_rms=state.obs_rms,
        reward_rms=state.reward_rms,
    )


def _ppo_iter_record(iteration, frames_per_iter, elapsed, iter_dt, metrics, ret_best):
    return {
        "iter": iteration,
        "step": iteration * frames_per_iter,
        "elapsed": elapsed,
        "iter_dt": iter_dt,
        "fps": frames_per_iter / iter_dt,
        "ret_rollout": metrics["rollout_return"],
        "ep_ret": metrics["ep_ret"],
        "ep_len": metrics["ep_len"],
        "ret_best": ret_best,
        "rew": metrics["rollout_reward"],
        "loss": metrics["loss"],
        "loss_pi": metrics["loss_objective"],
        "loss_v": metrics["loss_critic"],
        "entropy": metrics["entropy"],
        "kl": metrics["approx_kl"],
        "clipfrac": metrics["clipfrac"],
        "done_frac": metrics["done_fraction"],
    }


def _eval_args(state):
    return state.params, state.obs_rms


def _result(best_return: float, last_return: float, iterations: int, _frames: int):
    return MJXPPOResult(
        best_return=best_return,
        last_rollout_return=last_return,
        num_iterations=iterations,
    )
