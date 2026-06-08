from __future__ import annotations

from typing import Any


def reset_tensor_batch(env: Any, *args, **kwargs):
    raw = _raw_env(env)
    if raw is not env and hasattr(raw, "reset"):
        obs, info = raw.reset(*args, **kwargs)
    elif hasattr(env, "reset_batch"):
        obs, info = env.reset_batch(*args, **kwargs)
    else:
        return None
    return _flatten_obs_tensor_batch(obs, num_envs=_num_envs(env), device=_device(env)), info


def step_tensor_batch(env: Any, actions):
    raw = _raw_env(env)
    if raw is not env and hasattr(raw, "step"):
        step_out = raw.step(_format_tensor_actions(env, actions))
    elif hasattr(env, "step_batch"):
        step_out = env.step_batch(actions)
    else:
        return None
    if len(step_out) != 5:
        raise ValueError(f"Unsupported Isaac Lab step return arity: {len(step_out)}")
    obs, reward, terminated, truncated, info = step_out
    num_envs = _num_envs(env)
    device = _device(env)
    return (
        _flatten_obs_tensor_batch(obs, num_envs=num_envs, device=device),
        _flat_tensor(reward, num_envs=num_envs, device=device, dtype="float32"),
        _flat_tensor(terminated, num_envs=num_envs, device=device, dtype="bool"),
        _flat_tensor(truncated, num_envs=num_envs, device=device, dtype="bool"),
        info,
    )


def _flatten_obs_tensor_batch(obs: Any, *, num_envs: int, device: Any):
    if isinstance(obs, dict):
        if "policy" in obs:
            return _flatten_obs_tensor_batch(obs["policy"], num_envs=num_envs, device=device)
        if "observation" in obs:
            return _flatten_obs_tensor_batch(obs["observation"], num_envs=num_envs, device=device)
        parts = [_flatten_obs_tensor_batch(obs[key], num_envs=num_envs, device=device) for key in sorted(obs)]
        return _torch().cat(parts, dim=1) if parts else _torch().zeros((int(num_envs), 0), dtype=_torch().float32, device=device)

    tensor = _as_tensor(obs, device=device, dtype="float32")
    if tensor.ndim == 0:
        return tensor.reshape(1, 1).repeat(int(num_envs), 1)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1) if int(num_envs) == 1 else tensor.reshape(int(num_envs), -1)
    if int(tensor.shape[0]) != int(num_envs):
        return tensor.reshape(int(num_envs), -1)
    return tensor.reshape(int(num_envs), -1)


def _format_tensor_actions(env: Any, actions):
    tensor = _as_tensor(actions, device=_device(env), dtype="float32")
    single_shape = tuple(int(v) for v in getattr(getattr(env, "action_space", None), "shape", ()))
    target_shape = (_num_envs(env), *single_shape)
    return tensor.reshape(target_shape) if single_shape and tuple(tensor.shape) != target_shape else tensor


def _flat_tensor(value: Any, *, num_envs: int, device: Any, dtype: str):
    return _as_tensor(value, device=device, dtype=dtype).reshape(int(num_envs))


def _as_tensor(value: Any, *, device: Any, dtype: str):
    torch = _torch()
    torch_dtype = torch.bool if dtype == "bool" else torch.float32
    if hasattr(value, "detach"):
        return value.detach().to(device=device, dtype=torch_dtype)
    return torch.as_tensor(value, dtype=torch_dtype, device=device)


def _raw_env(env: Any):
    return getattr(env, "_env", env)


def _num_envs(env: Any) -> int:
    return int(getattr(env, "num_envs", getattr(env, "_num_envs", 1)))


def _device(env: Any):
    return getattr(_raw_env(env), "device", None)


def _torch():
    import torch

    return torch
