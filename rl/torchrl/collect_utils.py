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
    env_name = str(getattr(env_conf, "env_name", ""))
    return is_isaaclab_env_tag(env_name) or env_name.startswith("warp:")


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


def make_collect_env(
    env_conf: Any,
    *,
    env_index: int = 0,
    num_envs: int = 1,
    device: torch.device | str | None = None,
):
    """Unified creation of a TorchRL-compatible collection environment."""
    env_name = str(getattr(env_conf, "env_name", ""))
    if is_isaaclab_env_tag(env_name):
        return _make_native_isaaclab_collect_env(env_conf, env_index=int(env_index), num_envs=int(num_envs), device=device)
    if env_name.startswith("warp:"):
        return _make_native_warp_collect_env(env_conf, env_index=int(env_index), num_envs=int(num_envs), device=device)

    # 1. Use the core unified Gym creator (handles pixels, skip, clip, normalization)
    seed = int(getattr(env_conf, "problem_seed", 0)) + env_index
    base = env_conf.make_gym_env(seed=seed)

    # 2. TorchRL wrapping
    wrapped = _gym_wrapper_without_isaaclab_probe(base)

    # 3. Standard transforms (Always Float32)
    return tr_envs.TransformedEnv(wrapped, tr_transforms.DoubleToFloat())


def _make_native_warp_collect_env(env_conf: Any, *, env_index: int, num_envs: int, device: torch.device | str | None):
    # WarpAdapter handles its own vectorization
    base = env_conf.make(num_envs=num_envs)

    # We need a TorchRL wrapper for our UnifiedMJXWarpAdapter
    from tensordict import TensorDict
    from torchrl.envs.common import EnvBase

    class TorchRLWarpWrapper(EnvBase):
        def __init__(self, adapter, num_envs, device):
            super().__init__(device=device, batch_size=[num_envs])
            self.adapter = adapter
            self._num_envs = num_envs
            # Define specs using TorchRL 0.11.0 naming
            from torchrl.data import Bounded, UnboundedContinuous

            self.observation_spec = UnboundedContinuous(shape=(num_envs, *adapter.observation_space.shape), device=device)
            low = torch.as_tensor(adapter.action_space.low).to(device)
            high = torch.as_tensor(adapter.action_space.high).to(device)
            # Expand bounds to match the batch dimension
            low = low.unsqueeze(0).expand(num_envs, *low.shape)
            high = high.unsqueeze(0).expand(num_envs, *high.shape)

            self.action_spec = Bounded(
                low=low,
                high=high,
                shape=(num_envs, *adapter.action_space.shape),
                device=device,
            )
            self.reward_spec = UnboundedContinuous(shape=(num_envs, 1), device=device)
            self.done_spec = Bounded(low=0, high=1, shape=(num_envs, 1), dtype=torch.bool, device=device)

        def _reset(self, tensordict=None, **kwargs):
            obs_dict, data = self.adapter.reset()
            self._data = data  # Store simulation state
            return TensorDict(
                {"observation": obs_dict["obs"]},
                batch_size=self.batch_size,
                device=self.device,
            )

        def _step(self, tensordict):
            action = tensordict["action"]
            state = self.adapter.step(self._data, action)
            views = state.torch()
            return TensorDict(
                {
                    "observation": views["obs"],
                    "reward": views["reward"],
                    "done": views["done"],
                },
                batch_size=self.batch_size,
                device=self.device,
            )

        def _set_seed(self, seed):
            pass

    wrapped = TorchRLWarpWrapper(base, num_envs, device=torch.device(device) if device else None)
    return tr_envs.TransformedEnv(wrapped, tr_transforms.DoubleToFloat())
