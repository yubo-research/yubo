from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch


def _as_torch_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _move_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _to_cpu_replay_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, torch.device):
        return torch.device("cpu")
    if isinstance(value, dict):
        return type(value)((k, _to_cpu_replay_state(v)) for k, v in value.items())
    if isinstance(value, list):
        return [_to_cpu_replay_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple((_to_cpu_replay_state(v) for v in value))
    return value


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

    def add_batch(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, nxt: np.ndarray, done: np.ndarray) -> None:
        n = int(obs.shape[0])
        idx = (np.arange(n) + self.ptr) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = np.asarray(rew, dtype=np.float32).reshape(-1)
        self.nxt[idx] = nxt
        self.done[idx] = np.asarray(done, dtype=np.float32).reshape(-1)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def sample(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, ...]:
        device_t = _as_torch_device(device)
        idx = np.random.randint(0, self.size, size=int(batch_size))
        if os.environ.get("SAC_FUSED_REPLAY_SAMPLE", "1") in ("1", "true", "yes"):
            return self._sample_fused(idx, device_t)
        obs = _move_tensor_to_device(torch.from_numpy(self.obs[idx]), device_t)
        act = _move_tensor_to_device(torch.from_numpy(self.act[idx]), device_t)
        rew = _move_tensor_to_device(torch.from_numpy(self.rew[idx]), device_t)
        nxt = _move_tensor_to_device(torch.from_numpy(self.nxt[idx]), device_t)
        done = _move_tensor_to_device(torch.from_numpy(self.done[idx]), device_t)
        return (obs, act, rew, nxt, done)

    def _sample_fused(self, idx: np.ndarray, device_t: torch.device) -> tuple[torch.Tensor, ...]:
        """Single bulk transfer instead of five; S4 hypothesis."""
        obs_dim = self.obs.shape[1:]
        act_dim = self.act.shape[1]
        obs_flat = int(np.prod(obs_dim))
        batch = np.concatenate(
            [
                self.obs[idx].reshape(idx.shape[0], -1),
                self.nxt[idx].reshape(idx.shape[0], -1),
                self.act[idx],
                self.rew[idx].reshape(-1, 1),
                self.done[idx].reshape(-1, 1),
            ],
            axis=1,
            dtype=np.float32,
        )
        t = _move_tensor_to_device(torch.from_numpy(batch), device_t)
        n = idx.shape[0]
        obs = t[:, :obs_flat].reshape(n, *obs_dim)
        nxt = t[:, obs_flat : 2 * obs_flat].reshape(n, *obs_dim)
        act = t[:, 2 * obs_flat : 2 * obs_flat + act_dim]
        rew = t[:, -2].squeeze(-1)
        done = t[:, -1].squeeze(-1)
        return (obs, act, rew, nxt, done)

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


class CudaReplayBuffer:
    """GPU-backed replay; sample returns device tensors directly (S2). CUDA only."""

    def __init__(self, obs_shape: tuple[int, ...], act_dim: int, capacity: int, *, device: torch.device):
        if device.type != "cuda":
            raise ValueError("CudaReplayBuffer requires device.type=='cuda'")
        self.device = device
        self.capacity = int(capacity)
        self.obs = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32, device=device)
        self.nxt = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32, device=device)
        self.act = torch.zeros((self.capacity, int(act_dim)), dtype=torch.float32, device=device)
        self.rew = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self.done = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs: np.ndarray, act: np.ndarray, rew: np.ndarray, nxt: np.ndarray, done: np.ndarray) -> None:
        n = int(obs.shape[0])
        idx = (np.arange(n) + self.ptr) % self.capacity
        self.obs[idx] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.act[idx] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        self.rew[idx] = torch.as_tensor(np.asarray(rew, dtype=np.float32).reshape(-1), device=self.device)
        self.nxt[idx] = torch.as_tensor(nxt, dtype=torch.float32, device=self.device)
        self.done[idx] = torch.as_tensor(np.asarray(done, dtype=np.float32).reshape(-1), device=self.device)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def sample(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        obs = self.obs[idx]
        nxt = self.nxt[idx]
        act = self.act[idx]
        rew = self.rew[idx]
        done = self.done[idx]
        device_t = _as_torch_device(device)
        if obs.device != device_t:
            return (
                obs.to(device_t),
                act.to(device_t),
                rew.to(device_t),
                nxt.to(device_t),
                done.to(device_t),
            )
        return (obs, act, rew, nxt, done)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "ptr": int(self.ptr),
            "size": int(self.size),
            "obs": _to_cpu_replay_state(self.obs).numpy(),
            "nxt": _to_cpu_replay_state(self.nxt).numpy(),
            "act": _to_cpu_replay_state(self.act).numpy(),
            "rew": _to_cpu_replay_state(self.rew).numpy(),
            "done": _to_cpu_replay_state(self.done).numpy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.ptr = int(state.get("ptr", self.ptr))
        loaded_size = int(state.get("size", self.size))
        self.size = int(max(0, min(loaded_size, self.capacity)))
        obs_np = np.asarray(state["obs"], dtype=np.float32)
        nxt_np = np.asarray(state["nxt"], dtype=np.float32)
        act_np = np.asarray(state["act"], dtype=np.float32)
        rew_np = np.asarray(state["rew"], dtype=np.float32).reshape(-1)
        done_np = np.asarray(state["done"], dtype=np.float32).reshape(-1)
        if obs_np.shape != self.obs.shape:
            raise ValueError(f"Replay obs shape mismatch: expected {self.obs.shape}, got {obs_np.shape}")
        self.obs.copy_(torch.as_tensor(obs_np, device=self.device))
        self.nxt.copy_(torch.as_tensor(nxt_np, device=self.device))
        self.act.copy_(torch.as_tensor(act_np, device=self.device))
        self.rew.copy_(torch.as_tensor(rew_np, device=self.device))
        self.done.copy_(torch.as_tensor(done_np, device=self.device))


def make_replay_buffer(
    *,
    obs_shape: tuple[int, ...],
    act_dim: int,
    capacity: int,
    backend: str = "numpy",
    device: torch.device | None = None,
):
    backend_name = str(backend).strip().lower()
    if backend_name == "numpy":
        return NumpyReplayBuffer(obs_shape=obs_shape, act_dim=act_dim, capacity=capacity)
    if backend_name == "cuda":
        if device is None:
            raise ValueError("replay_backend='cuda' requires device")
        return CudaReplayBuffer(obs_shape=obs_shape, act_dim=act_dim, capacity=capacity, device=device)
    raise ValueError(f"Unsupported replay backend: {backend}. Use 'numpy' or 'cuda'.")


def resolve_replay_backend(backend: str, *, device: torch.device | str, platform_name: str | None = None) -> str:
    backend_name = str(backend).strip().lower()
    if backend_name == "numpy":
        return backend_name
    if backend_name == "cuda":
        return backend_name
    if backend_name != "auto":
        raise ValueError(f"Unsupported replay backend: {backend}")
    device_t = _as_torch_device(device)
    if device_t.type == "cuda":
        return "cuda"
    return "numpy"


__all__ = ["CudaReplayBuffer", "NumpyReplayBuffer", "make_replay_buffer", "resolve_replay_backend"]
