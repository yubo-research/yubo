from __future__ import annotations

from typing import Any, NamedTuple

from problems.jax_env_core import _RewardRunningMeanStd, _RunningMeanStd


class _Replay(NamedTuple):
    obs: Any
    action: Any
    reward: Any
    terminated: Any
    truncated: Any
    next_obs: Any
    ptr: Any
    size: Any


class _TrainState(NamedTuple):
    actor: dict[str, Any]
    critic1: dict[str, Any]
    critic2: dict[str, Any]
    target1: dict[str, Any]
    target2: dict[str, Any]
    actor_opt: Any
    critic_opt: Any
    alpha_opt: Any
    log_alpha: Any
    obs_rms: _RunningMeanStd
    reward_rms: _RewardRunningMeanStd
    discounted_return: Any
    obs: Any
    env_state: Any
    running_return: Any
    running_length: Any
    replay: _Replay
    key: Any


class _AgentState(NamedTuple):
    actor: dict[str, Any]
    critic1: dict[str, Any]
    critic2: dict[str, Any]
    target1: dict[str, Any]
    target2: dict[str, Any]
    actor_opt: Any
    critic_opt: Any
    alpha_opt: Any
    log_alpha: Any
    obs_rms: _RunningMeanStd
    reward_rms: _RewardRunningMeanStd


def _checkpoint_fn(state: _TrainState) -> _AgentState:
    return _AgentState(
        actor=state.actor,
        critic1=state.critic1,
        critic2=state.critic2,
        target1=state.target1,
        target2=state.target2,
        actor_opt=state.actor_opt,
        critic_opt=state.critic_opt,
        alpha_opt=state.alpha_opt,
        log_alpha=state.log_alpha,
        obs_rms=state.obs_rms,
        reward_rms=state.reward_rms,
    )
