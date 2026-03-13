from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from common import obs_mode
from rl.core import episode_rollout


def test_obs_mode_helpers():
    assert obs_mode.normalize_obs_mode(None) == "vector"
    assert obs_mode.obs_mode_uses_pixels("image")
    assert obs_mode.obs_mode_uses_pixels("mixed")
    assert not obs_mode.obs_mode_uses_pixels("vector")
    assert obs_mode.obs_mode_pixels_only("image")
    assert not obs_mode.obs_mode_pixels_only("mixed")
    assert obs_mode.obs_mode_from_flags(from_pixels=False, pixels_only=False) == "vector"
    assert obs_mode.obs_mode_from_flags(from_pixels=True, pixels_only=False) == "mixed"
    with pytest.raises(ValueError, match="obs_mode must be one of"):
        obs_mode.normalize_obs_mode("bad-mode")


class _FakeContinuousEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(
            low=np.asarray([-2.0], dtype=np.float32),
            high=np.asarray([2.0], dtype=np.float32),
        )
        self._step = 0

    def reset(self, seed=None):
        self._step = 0
        return (
            {
                "aux": np.asarray([2.0], dtype=np.float32),
                "state": np.asarray([1.0, 3.0], dtype=np.float32),
            },
            {"seed": seed},
        )

    def step(self, action):
        self._step += 1
        done = self._step >= 2
        reward = np.asarray([1.5], dtype=np.float32)
        obs = {"pixels": np.asarray([7.0], dtype=np.float32)}
        return (obs, reward, done, False, {})

    def close(self):
        return None


def test_episode_rollout_public_api():
    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(max_steps=5),
        noise_seed_0=10,
        frozen_noise=False,
        make=lambda: _FakeContinuousEnv(),
    )
    seen = []

    def policy(obs):
        seen.append(np.asarray(obs).copy())
        return np.asarray([0.0], dtype=np.float32)

    traj = episode_rollout.Trajectory(1.0, 0.2)
    mean_result = episode_rollout.MeanReturnResult(1.0, 0.0, True)
    assert traj.rreturn == 1.0 and traj.rreturn_se == 0.2
    assert mean_result.all_same is True

    ret = episode_rollout.collect_episode_return(env_conf, policy, noise_seed=3)
    assert ret == 3.0
    assert seen[0].shape == (2,)

    noisy_traj, noise_seed = episode_rollout.collect_trajectory_with_noise(env_conf, policy, i_noise=5, denoise_seed=2)
    assert noisy_traj.rreturn == 3.0
    assert noise_seed == 17

    mean_over_runs = episode_rollout.mean_return_over_runs(env_conf, policy, 2, i_noise=1)
    assert mean_over_runs.mean == 3.0
    assert mean_over_runs.all_same is True
    with pytest.raises(ValueError, match="num_denoise must be > 0"):
        episode_rollout.mean_return_over_runs(env_conf, policy, 0)

    denoised_once, denoised_seed = episode_rollout.denoise(env_conf, policy, num_denoise=None, i_noise=1)
    assert denoised_once.rreturn == 3.0
    assert denoised_seed == 11

    denoised_mean, denoised_seed2 = episode_rollout.denoise(env_conf, policy, num_denoise=3, i_noise=1)
    assert denoised_mean.rreturn == 3.0
    assert denoised_mean.rreturn_se == 0.0
    assert denoised_seed2 is None

    best = episode_rollout.best(env_conf, policy, 2, i_noise=1)
    assert best == 3.0
