"""Tests for Atari env support. Skip if ale-py not installed."""

from __future__ import annotations

import pytest

try:
    from ale_py import roms

    HAS_ALE = True
except ImportError:
    HAS_ALE = False


def _has_pong_rom() -> bool:
    if not HAS_ALE:
        return False
    try:
        return roms.get_rom_path("pong") is not None
    except OSError:
        return False


@pytest.mark.skipif(
    not _has_pong_rom(),
    reason="ale-py Pong ROM not installed; pip install gymnasium[accept-rom-license]",
)
def test_atari_env_conf_and_trajectory():
    import numpy as np

    from optimizer.trajectories import collect_trajectory
    from problems.env_conf import default_policy, get_env_conf
    from problems.env_conf_backends import register_with_env_conf

    register_with_env_conf()
    env_conf = get_env_conf("atari:Pong", problem_seed=17)
    env_conf.max_steps = 500  # Short episode for test
    policy = default_policy(env_conf)

    assert policy.num_params() > 1000
    x = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(x)
    traj = collect_trajectory(env_conf, policy, noise_seed=0)
    assert isinstance(traj.rreturn, (int, float))
