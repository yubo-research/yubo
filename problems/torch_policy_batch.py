from __future__ import annotations

from typing import Any

import numpy as np


def try_functional_policy_actions(
    policy: Any,
    codec: Any,
    xs: np.ndarray,
    candidate_idx: np.ndarray,
    obs: np.ndarray,
    active: np.ndarray,
    zero_action: np.ndarray,
) -> np.ndarray | None:
    actions = try_functional_policy_actions_tensor(policy, codec, xs, candidate_idx, obs, active, zero_action)
    if actions is None:
        return None
    return actions.detach().cpu().numpy().astype(np.float32)


def try_functional_policy_actions_tensor(
    policy: Any,
    codec: Any,
    xs: np.ndarray,
    candidate_idx: np.ndarray,
    obs: Any,
    active: Any,
    zero_action: Any,
):
    if not _supports_functional_policy(policy):
        return None
    try:
        import torch
        from torch.func import functional_call, vmap

        return _functional_policy_actions(
            torch,
            functional_call,
            vmap,
            policy,
            codec,
            xs,
            candidate_idx,
            obs,
            active,
            zero_action,
        )
    except Exception:
        return None


def _supports_functional_policy(policy: Any) -> bool:
    if not all(hasattr(policy, name) for name in ("named_parameters", "parameters", "forward")):
        return False
    if not all(hasattr(policy, name) for name in ("_flat_params_init", "_const_scale")):
        return False
    if getattr(policy, "_rnn_hidden_size", None) is not None:
        return False
    if bool(getattr(policy, "_use_prev_action", False)) or bool(getattr(policy, "_use_phase_features", False)):
        return False
    return True


def _functional_policy_actions(
    torch,
    functional_call,
    vmap,
    policy: Any,
    codec: Any,
    xs: np.ndarray,
    candidate_idx: np.ndarray,
    obs: np.ndarray,
    active: np.ndarray,
    zero_action: np.ndarray,
) -> np.ndarray | None:
    wrapper = _FunctionalPolicy(policy)
    params = _batched_param_dict(torch, wrapper, policy, codec, xs)
    if params is None:
        return None
    slots_by_candidate = _slots_by_candidate(candidate_idx, int(xs.shape[0]))
    if slots_by_candidate is None:
        return None
    obs_batch = _candidate_obs_tensor(torch, obs, slots_by_candidate, next(iter(params.values())))
    buffers = {name: value.detach() for name, value in wrapper.named_buffers()}

    def call_one(single_params, single_obs):
        return functional_call(wrapper, (single_params, buffers), (single_obs,))

    with torch.no_grad():
        actions = vmap(call_one)(params, obs_batch)
    return _scatter_actions(actions, slots_by_candidate, active, zero_action)


def _batched_param_dict(torch, wrapper, policy: Any, codec: Any, xs: np.ndarray):
    named_params = list(wrapper.named_parameters())
    if sum(int(param.numel()) for _name, param in named_params) != int(codec.dim):
        return None
    first = named_params[0][1]
    flat = torch.as_tensor(xs, dtype=first.dtype, device=first.device)
    init = torch.as_tensor(policy._flat_params_init, dtype=first.dtype, device=first.device)
    flat = init.reshape(1, -1) + flat * float(policy._const_scale)
    params = {}
    cursor = 0
    for name, param in named_params:
        size = int(param.numel())
        params[name] = flat[:, cursor : cursor + size].reshape((flat.shape[0], *tuple(param.shape)))
        cursor += size
    return params


def _slots_by_candidate(candidate_idx: np.ndarray, num_candidates: int) -> list[np.ndarray] | None:
    slots = [np.flatnonzero(candidate_idx == idx) for idx in range(int(num_candidates))]
    sizes = {int(slot.size) for slot in slots}
    if len(sizes) != 1 or 0 in sizes:
        return None
    return slots


def _candidate_obs_tensor(torch, obs: np.ndarray, slots_by_candidate: list[np.ndarray], template):
    if hasattr(obs, "detach"):
        return torch.stack([obs[slots].to(dtype=template.dtype, device=template.device) for slots in slots_by_candidate], dim=0)
    obs_batch = np.stack([np.asarray(obs[slots], dtype=np.float32) for slots in slots_by_candidate], axis=0)
    return torch.as_tensor(obs_batch, dtype=template.dtype, device=template.device)


def _scatter_actions(actions, slots_by_candidate: list[np.ndarray], active: np.ndarray, zero_action: np.ndarray) -> np.ndarray:
    torch = _torch_from(actions)
    active_tensor = torch.as_tensor(active, dtype=torch.bool, device=actions.device).reshape(-1)
    zero = torch.as_tensor(zero_action, dtype=actions.dtype, device=actions.device)
    raw = torch.zeros((int(active_tensor.numel()), *tuple(zero.shape)), dtype=actions.dtype, device=actions.device)
    for cand_idx, slots in enumerate(slots_by_candidate):
        raw[torch.as_tensor(slots, dtype=torch.long, device=actions.device)] = actions[int(cand_idx)].reshape((int(slots.size), *tuple(zero.shape)))
    raw[~active_tensor] = zero
    return raw


def _torch_from(tensor):
    import torch

    _ = tensor
    return torch


class _FunctionalPolicy:
    def __new__(cls, policy):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.policy = wrapped

            def forward(self, x):
                if hasattr(self.policy, "forward_tensor"):
                    return self.policy.forward_tensor(x)
                return self.policy.forward(x)

        return Wrapper(policy)
