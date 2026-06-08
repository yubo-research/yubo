from __future__ import annotations

from typing import Any, NamedTuple

import mujoco_warp
import numpy as np
import warp as wp


class WarpState(NamedTuple):
    obs: wp.array
    reward: wp.array
    done: wp.array
    data: mujoco_warp.Data


class GymnasiumWarpAdapter:
    """Warp execution backend for Gymnasium MuJoCo registry envs."""

    def __init__(self, env_name: str, num_envs: int = 1) -> None:
        import gymnasium as gym

        from problems.mjx_env import _action_bounds, parse_gymnasium_env_id

        wp.init()
        self.env_id = parse_gymnasium_env_id(env_name)
        self.num_envs = int(num_envs)

        # Loader: Use gymnasium just for the MjModel
        tmp_env = gym.make(self.env_id)
        self.model = tmp_env.unwrapped.model
        obs_shape = tuple(int(dim) for dim in tmp_env.observation_space.shape)
        tmp_env.close()

        # Warp model resides on the GPU (shared across all worlds)
        self.warp_model = mujoco_warp.put_model(self.model)

        # Spaces
        low, high = _action_bounds(self.model, np)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def reset(self, seed: int | None = None) -> tuple[wp.array, mujoco_warp.Data]:
        # CORRECT USAGE: Pass the MuJoCo model (self.model), not the warp_model
        data = mujoco_warp.make_data(self.model, nworld=self.num_envs)
        mujoco_warp.reset_data(self.warp_model, data)
        # In-place reset (we'd add noise kernels here for full implementation)
        return self._get_obs(data), data

    def step(self, data: mujoco_warp.Data, action: Any) -> WarpState:
        # Action handling with zero-copy if possible
        if not isinstance(action, wp.array):
            # Ensure action is a Warp array on the same device as the state
            action = wp.array(action, dtype=wp.float32, device=data.qpos.device)

        # mujoco-warp expects batched control [nworld, nu]
        if action.ndim == 1:
            action = action.reshape((1, -1))

        data.ctrl = action
        mujoco_warp.step(self.warp_model, data)

        obs = self._get_obs(data)
        reward = self._get_reward(data)
        # Vectorized done flag
        done = wp.zeros((self.num_envs, 1), dtype=wp.bool, device=data.qpos.device)

        return WarpState(obs=obs, reward=reward, done=done, data=data)

    def _get_obs(self, data: mujoco_warp.Data) -> wp.array:
        import torch

        qpos = wp.to_torch(data.qpos)
        qvel = wp.to_torch(data.qvel)

        qpos_feat = qpos[:, 1:] if qpos.shape[1] > 1 else qpos
        parts = [qpos_feat, qvel]

        if self.model.nsensordata > 0:
            sensordata = wp.to_torch(data.sensordata)
            parts.append(sensordata)

        # Cat and ensure contiguous memory for Warp
        obs_t = torch.cat(parts, dim=1).contiguous().float()

        # Save reference to prevent PyTorch GC while Warp uses it
        self._current_obs_t = obs_t
        return wp.from_torch(obs_t)

    def _get_reward(self, data: mujoco_warp.Data) -> wp.array:
        # Simplified reward: just forward velocity (qvel[0]) for testing
        qvel = wp.to_torch(data.qvel)
        reward_t = qvel[:, 0:1].contiguous().float()
        self._current_reward_t = reward_t
        return wp.from_torch(reward_t)

    @staticmethod
    def to_torch(warp_array: wp.array):
        return wp.to_torch(warp_array)

    @staticmethod
    def to_jax(warp_array: wp.array):
        return wp.to_jax(warp_array)
