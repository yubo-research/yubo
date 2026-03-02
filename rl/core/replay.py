"""Shared replay buffers for off-policy RL algorithms."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch


class NumpyReplayBuffer:
    def __init__(self, obs_shape: tuple[int, ...], act_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.nxt = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.act = np.zeros((self.capacity, int(act_dim)), dtype=np.float32)
        self.rew = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        nxt: np.ndarray,
        done: np.ndarray,
    ) -> None:
        n = int(obs.shape[0])
        idx = (np.arange(n) + self.ptr) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = np.asarray(rew, dtype=np.float32).reshape(-1)
        self.nxt[idx] = nxt
        self.done[idx] = np.asarray(done, dtype=np.float32).reshape(-1)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        obs = torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device)
        act = torch.as_tensor(self.act[idx], dtype=torch.float32, device=device)
        rew = torch.as_tensor(self.rew[idx], dtype=torch.float32, device=device)
        nxt = torch.as_tensor(self.nxt[idx], dtype=torch.float32, device=device)
        done = torch.as_tensor(self.done[idx], dtype=torch.float32, device=device)
        return obs, act, rew, nxt, done

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "ptr": int(self.ptr),
            "size": int(self.size),
            "obs": np.asarray(self.obs).copy(),
            "nxt": np.asarray(self.nxt).copy(),
            "act": np.asarray(self.act).copy(),
            "rew": np.asarray(self.rew).copy(),
            "done": np.asarray(self.done).copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.ptr = int(state.get("ptr", self.ptr))
        loaded_size = int(state.get("size", self.size))
        self.size = int(max(0, min(loaded_size, self.capacity)))

        obs = np.asarray(state["obs"], dtype=np.float32)
        nxt = np.asarray(state["nxt"], dtype=np.float32)
        act = np.asarray(state["act"], dtype=np.float32)
        rew = np.asarray(state["rew"], dtype=np.float32).reshape(-1)
        done = np.asarray(state["done"], dtype=np.float32).reshape(-1)

        if obs.shape != self.obs.shape:
            raise ValueError(f"Replay obs shape mismatch: expected {self.obs.shape}, got {obs.shape}")
        if nxt.shape != self.nxt.shape:
            raise ValueError(f"Replay nxt shape mismatch: expected {self.nxt.shape}, got {nxt.shape}")
        if act.shape != self.act.shape:
            raise ValueError(f"Replay act shape mismatch: expected {self.act.shape}, got {act.shape}")
        if rew.shape != self.rew.shape:
            raise ValueError(f"Replay rew shape mismatch: expected {self.rew.shape}, got {rew.shape}")
        if done.shape != self.done.shape:
            raise ValueError(f"Replay done shape mismatch: expected {self.done.shape}, got {done.shape}")

        self.obs[...] = obs
        self.nxt[...] = nxt
        self.act[...] = act
        self.rew[...] = rew
        self.done[...] = done


class TorchRLReplayBuffer:
    def __init__(self, obs_shape: tuple[int, ...], act_dim: int, capacity: int):
        _ = obs_shape, act_dim
        tr_data = __import__("torchrl.data", fromlist=["TensorDictReplayBuffer", "LazyTensorStorage"])
        self.capacity = int(capacity)
        self._write_count = 0
        self._replay = tr_data.TensorDictReplayBuffer(
            storage=tr_data.LazyTensorStorage(int(capacity)),
        )

    @property
    def size(self) -> int:
        return int(min(int(self._write_count), self.capacity))

    @property
    def ptr(self) -> int:
        return int(int(self._write_count) % int(self.capacity))

    def add_batch(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        nxt: np.ndarray,
        done: np.ndarray,
    ) -> None:
        tensordict_mod = __import__("tensordict", fromlist=["TensorDict"])
        tensor_dict = tensordict_mod.TensorDict
        n = int(obs.shape[0])
        reward = torch.as_tensor(rew, dtype=torch.float32).reshape(n, 1)
        done_t = torch.as_tensor(done, dtype=torch.float32).reshape(n, 1)
        td = tensor_dict(
            {
                "observation": torch.as_tensor(obs, dtype=torch.float32),
                "action": torch.as_tensor(act, dtype=torch.float32),
                "next": tensor_dict(
                    {
                        "observation": torch.as_tensor(nxt, dtype=torch.float32),
                        "reward": reward,
                        "done": done_t,
                    },
                    batch_size=[n],
                ),
            },
            batch_size=[n],
        )
        self._replay.extend(td)
        self._write_count += int(n)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch = self._replay.sample(int(batch_size)).to(device)
        obs = batch["observation"]
        act = batch["action"]
        rew = batch["next", "reward"].reshape(-1)
        nxt = batch["next", "observation"]
        done = batch["next", "done"].reshape(-1)
        return obs, act, rew, nxt, done

    def state_dict(self) -> dict[str, Any]:
        return {
            "replay_state": self._replay.state_dict(),
            "write_count": int(self._write_count),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        replay_state = state.get("replay_state") if isinstance(state, dict) else None
        if replay_state is None:
            replay_state = state
        self._replay.load_state_dict(replay_state)
        self._write_count = int(state.get("write_count", int(self._replay.write_count)))


def make_replay_buffer(
    *,
    obs_shape: tuple[int, ...],
    act_dim: int,
    capacity: int,
    backend: str = "numpy",
):
    backend_name = str(backend).strip().lower()
    if backend_name == "numpy":
        return NumpyReplayBuffer(obs_shape=obs_shape, act_dim=act_dim, capacity=capacity)
    if backend_name == "torchrl":
        return TorchRLReplayBuffer(obs_shape=obs_shape, act_dim=act_dim, capacity=capacity)
    raise ValueError(f"Unsupported replay backend: {backend}")


def resolve_replay_backend(
    backend: str,
    *,
    device: torch.device | str,
    platform_name: str | None = None,
) -> str:
    backend_name = str(backend).strip().lower()
    if backend_name in {"numpy", "torchrl"}:
        return backend_name
    if backend_name != "auto":
        raise ValueError(f"Unsupported replay backend: {backend}")
    platform_key = str(platform_name or sys.platform).strip().lower()
    if platform_key == "darwin":
        return "numpy"
    if isinstance(device, torch.device):
        device_type = str(device.type).strip().lower()
    else:
        device_type = str(device).strip().lower()
    if device_type == "cuda":
        return "torchrl"
    return "numpy"


__all__ = [
    "NumpyReplayBuffer",
    "TorchRLReplayBuffer",
    "make_replay_buffer",
    "resolve_replay_backend",
]
