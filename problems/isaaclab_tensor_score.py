from __future__ import annotations

import math
from typing import Any

import numpy as np

from problems.isaaclab_tensor_io import reset_tensor_batch, step_tensor_batch
from problems.torch_policy_batch import try_functional_policy_actions_tensor


def try_evaluate_many_tensor_vectorized(scorer: Any, xs: np.ndarray, env: Any, *, seed: int):
    try:
        import torch

        return _evaluate_many_tensor(torch, scorer, xs, env, seed=int(seed))
    except Exception:
        return None


def _evaluate_many_tensor(torch, scorer: Any, xs: np.ndarray, env: Any, *, seed: int):
    num_candidates = int(xs.shape[0])
    slots = num_candidates * int(scorer._episodes)
    reset_out = reset_tensor_batch(env, seed=int(seed))
    if reset_out is None:
        return None
    obs, _info = reset_out
    if int(obs.ndim) != 2 or int(obs.shape[0]) != slots:
        return None

    candidate_idx = np.repeat(np.arange(num_candidates, dtype=np.int64), int(scorer._episodes))
    returns = torch.zeros(slots, dtype=torch.float64, device=obs.device)
    steps = torch.zeros(slots, dtype=torch.int64, device=obs.device)
    active = torch.ones(slots, dtype=torch.bool, device=obs.device)
    zero_action = torch.zeros(_action_shape(env), dtype=torch.float32, device=obs.device)

    for _ in range(int(scorer.steps_per_episode)):
        raw_actions = try_functional_policy_actions_tensor(scorer._policy, scorer._codec, xs, candidate_idx, obs, active, zero_action)
        if raw_actions is None:
            return None
        step_out = step_tensor_batch(
            env,
            _scale_action_tensor_to_space(raw_actions, getattr(env, "action_space", None)),
        )
        if step_out is None:
            return None
        obs, reward, terminated, truncated, _info = step_out
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        returns[active] += reward.to(dtype=torch.float64)[active]
        steps[active] += 1
        active &= ~done
        if not bool(torch.any(active).item()):
            break
    return (
        *_summarize_tensor_returns(returns, int(scorer._episodes), num_candidates),
        int(steps.sum().item()),
    )


def _scale_action_tensor_to_space(actions, action_space: Any):
    if action_space is None or not hasattr(action_space, "low"):
        return actions
    torch = _torch_from(actions)
    low = torch.as_tensor(action_space.low, dtype=actions.dtype, device=actions.device)
    high = torch.as_tensor(action_space.high, dtype=actions.dtype, device=actions.device)
    if not bool(torch.all(torch.isfinite(low)).item()) or not bool(torch.all(torch.isfinite(high)).item()):
        return torch.clamp(actions, -1.0, 1.0)
    return low + (high - low) * (1.0 + actions) / 2.0


def _summarize_tensor_returns(returns, episodes: int, num_candidates: int):
    values = returns.reshape(int(num_candidates), int(episodes))
    means = values.mean(dim=1).detach().cpu().numpy().astype(np.float64)
    if int(episodes) <= 1:
        ses = np.zeros((int(num_candidates),), dtype=np.float64)
    else:
        ses = (values.std(dim=1, unbiased=False) / math.sqrt(float(episodes))).detach().cpu().numpy().astype(np.float64)
    return means, ses


def _action_shape(env: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(getattr(env, "action_space", None), "shape", ()))


def _torch_from(tensor):
    import torch

    _ = tensor
    return torch
