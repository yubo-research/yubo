"""Torch-env wrapper for MLP policies to enable direct parameter perturbation.

This module provides a wrapper around gym environments with MLP policies
that exposes the underlying torch module for direct perturbation via
GaussianPerturbator, enabling use with BSZO and other UHD optimizers.
"""

from __future__ import annotations

import numpy as np
import torch


class MLPTorchEnv:
    """Torch environment exposing an MLP policy module for direct perturbation.

    This class wraps an MLP policy (torch nn.Module) and provides the interface
    needed by BSZO and other UHD optimizers that require direct module access.
    """

    def __init__(self, module: torch.nn.Module, env, max_steps: int = 1000):
        self._module = module
        self._env = env
        self._max_steps = max_steps
        self._current_step = 0
        self._obs = None

        # Observation/action spaces from underlying env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @property
    def module(self) -> torch.nn.Module:
        """Return the underlying torch module for parameter perturbation."""
        return self._module

    def reset(self, seed: int | None = None) -> tuple:
        """Reset the underlying environment."""
        self._current_step = 0
        self._obs, info = self._env.reset(seed=seed)
        return self._obs, info

    def step(self, action: np.ndarray) -> tuple:
        """Step the underlying environment with the given action."""
        self._current_step += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._obs = obs

        # Add step limit to termination conditions
        if self._current_step >= self._max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the underlying environment."""
        self._env.close()


class MLPTorchEnvWrapper:
    """Wrapper for gym environments with MLP policies that exposes torch module.

    This wrapper creates a gym-like environment that can be used with UHD
    optimizers requiring direct module access (like BSZO).
    """

    def __init__(
        self,
        env,
        policy_module: torch.nn.Module,
        max_steps: int = 1000,
        num_frames_skip: int = 1,
    ):
        self._env = env
        self._policy_module = policy_module
        self._max_steps = max_steps
        self._num_frames_skip = num_frames_skip

        # Expose spaces for compatibility
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._rng = np.random.default_rng()

    def torch_env(self) -> MLPTorchEnv:
        """Return a TorchEnv variant sharing this wrapper's policy module."""
        return MLPTorchEnv(
            module=self._policy_module,
            env=self._env,
            max_steps=self._max_steps,
        )

    def reset(self, seed: int | None = None) -> tuple:
        """Reset the environment."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self._env.reset(seed=seed)
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        """Step the environment with the given action."""
        obs, reward, terminated, truncated, info = self._env.step(action)

        # Handle frame skipping
        for _ in range(self._num_frames_skip - 1):
            if terminated or truncated:
                break
            obs, r, terminated, truncated, info = self._env.step(action)
            reward += r

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the underlying environment."""
        self._env.close()


def wrap_mlp_env(env, policy, max_steps: int = 1000, num_frames_skip: int = 1):
    """Wrap a gym environment with an MLP policy for use with UHD optimizers.

    Args:
        env: The underlying gym environment
        policy: The MLP policy (torch nn.Module)
        max_steps: Maximum steps per episode
        num_frames_skip: Number of frames to skip between actions

    Returns:
        MLPTorchEnvWrapper that exposes the policy module via torch_env()
    """
    return MLPTorchEnvWrapper(
        env=env,
        policy_module=policy,
        max_steps=max_steps,
        num_frames_skip=num_frames_skip,
    )
