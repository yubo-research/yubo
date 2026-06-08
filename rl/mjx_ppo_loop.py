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


def _ppo_iter_record(
    iteration,
    frames_per_iter,
    elapsed,
    iter_dt,
    metrics,
    ret_best,
    *,
    ret_eval=None,
    eval_dt=None,
):
    from rl.mjx_metrics import build_iter_record

    return build_iter_record(
        algo_name="ppo",
        iteration=iteration,
        frames_per_iter=frames_per_iter,
        elapsed=elapsed,
        iter_dt=iter_dt,
        metrics=metrics,
        ret_best=ret_best,
        ret_eval=ret_eval,
        eval_dt=eval_dt,
    )


def _eval_args(state):
    return state.params, state.obs_rms


def _result(best_return: float, last_return: float, iterations: int, _frames: int):
    return MJXPPOResult(
        best_return=best_return,
        last_rollout_return=last_return,
        num_iterations=iterations,
    )
