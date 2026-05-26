from __future__ import annotations

from typing import Any

import torch
import torchrl.envs as tr_envs
import torchrl.envs.transforms as tr_transforms

from problems.isaaclab_env_adapters import is_isaaclab_env_tag, make_raw_isaaclab_env


def _gym_wrapper_without_isaaclab_probe(base):
    import torchrl.envs.libs.gym as torchrl_gym

    if hasattr(torchrl_gym, "_has_isaaclab"):
        torchrl_gym._has_isaaclab = False
    return tr_envs.GymWrapper(base)


def uses_native_isaaclab_collect_env(env_conf: Any) -> bool:
    return is_isaaclab_env_tag(str(getattr(env_conf, "env_name", "")))


def _normalize_raw_isaaclab_kwargs(env_conf: Any, *, num_envs: int, device: torch.device | str | None) -> dict[str, Any]:
    kwargs = dict(getattr(env_conf, "kwargs", {}) or {})
    kwargs.pop("batched", None)
    kwargs.pop("num_envs", None)
    raw_device = kwargs.pop("device", None)
    if raw_device is None and device is not None:
        raw_device = str(device)
    return {
        "num_envs": int(num_envs),
        "device": raw_device,
        **kwargs,
    }


def _isaaclab_wrapper_cls():
    wrapper = getattr(tr_envs, "IsaacLabWrapper", None)
    if wrapper is not None:
        return wrapper
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    return IsaacLabWrapper


def _make_native_isaaclab_collect_env(env_conf: Any, *, env_index: int, num_envs: int, device: torch.device | str | None):
    seed = int(getattr(env_conf, "problem_seed", 0)) + int(env_index)
    raw = make_raw_isaaclab_env(
        str(getattr(env_conf, "env_name")),
        seed=seed,
        **_normalize_raw_isaaclab_kwargs(env_conf, num_envs=int(num_envs), device=device),
    )
    wrapped = _isaaclab_wrapper_cls()(raw, device=torch.device(device) if device is not None else None)
    return tr_envs.TransformedEnv(
        wrapped,
        tr_transforms.Compose(
            tr_transforms.RenameTransform(["policy"], ["observation"], create_copy=False),
            tr_transforms.DoubleToFloat(),
        ),
    )


def make_collect_env(env_conf: Any, *, env_index: int = 0, num_envs: int = 1, device: torch.device | str | None = None):
    """Unified creation of a TorchRL-compatible collection environment."""
    if uses_native_isaaclab_collect_env(env_conf):
        return _make_native_isaaclab_collect_env(env_conf, env_index=int(env_index), num_envs=int(num_envs), device=device)

    # 1. Use the core unified Gym creator (handles pixels, skip, clip, normalization)
    seed = int(getattr(env_conf, "problem_seed", 0)) + env_index
    base = env_conf.make_gym_env(seed=seed)

    # 2. TorchRL wrapping
    wrapped = _gym_wrapper_without_isaaclab_probe(base)

    # 3. Standard transforms (Always Float32)
    return tr_envs.TransformedEnv(wrapped, tr_transforms.DoubleToFloat())
