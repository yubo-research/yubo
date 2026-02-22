"""Tests for Atari env support. Skip if ale-py not installed."""

from __future__ import annotations

import pytest

try:
    import ale_py  # noqa: F401

    HAS_ALE = True
except ImportError:
    HAS_ALE = False


@pytest.mark.skipif(
    not HAS_ALE,
    reason="ale-py not installed; pip install gymnasium[accept-rom-license]",
)
def test_atari_env_conf_and_trajectory():
    import numpy as np

    import problems.env_conf_atari_dm  # noqa: F401 - register atari/dm handlers
    from optimizer.trajectories import collect_trajectory
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("atari:Pong", problem_seed=17)
    env_conf.gym_conf.max_steps = 500  # Short episode for test
    policy = default_policy(env_conf)

    assert policy.num_params() > 1000
    x = np.random.uniform(-1, 1, size=policy.num_params())
    policy.set_params(x)
    traj = collect_trajectory(env_conf, policy, noise_seed=0)
    assert isinstance(traj.rreturn, (int, float))
