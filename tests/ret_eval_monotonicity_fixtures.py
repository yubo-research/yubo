"""Designers and helpers for ret_eval monotonicity tests."""

from __future__ import annotations

import numpy as np

from problems.env_conf import default_policy, get_env_conf


class MockDesignerWithRreturnEst:
    """Designer that sets rreturn_est (like turbo-enn-fit-ucb)."""

    def __init__(self, num_params):
        self._num_params = num_params
        self._data = []
        self._call_count = 0

    def __call__(self, data, num_arms, telemetry=None):
        self._data = data
        self._call_count += 1

        for _i, d in enumerate(data):
            d.trajectory.rreturn_est = float(d.trajectory.rreturn) + np.random.randn() * 10

        policies = []
        for _ in range(num_arms):
            policy = _make_mock_policy(self._num_params)
            policy.set_params(np.random.uniform(-1, 1, self._num_params))
            policies.append(policy)
        return policies


class MockDesignerWithoutRreturnEst:
    """Designer that does NOT set rreturn_est (like random or sobol)."""

    def __init__(self, num_params):
        self._num_params = num_params

    def __call__(self, data, num_arms, telemetry=None):
        policies = []
        for _ in range(num_arms):
            policy = _make_mock_policy(self._num_params)
            policy.set_params(np.random.uniform(-1, 1, self._num_params))
            policies.append(policy)
        return policies


def _make_mock_policy(num_params):
    env_conf = get_env_conf("f:sphere-2d", problem_seed=0, noise_seed_0=0)
    return default_policy(env_conf)
