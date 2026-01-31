"""Regression tests for ret_eval behavior with different noise types and designer modes.

Tests cover the 2x2 matrix:
- Noise type: {frozen, natural}
- Designer mode: {provides rreturn_est, does not provide rreturn_est}

For frozen noise: ret_eval should be monotonically increasing (best raw rreturn seen so far)
For natural noise: ret_eval can fluctuate (re-evaluation of best policy with passive seeds)
"""

import numpy as np

from common.collector import Collector
from optimizer.optimizer import Optimizer
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

        # Set rreturn_est to a UCB-like estimate (different from raw rreturn)
        for i, d in enumerate(data):
            # UCB estimate: simulate model uncertainty
            d.trajectory.rreturn_est = float(d.trajectory.rreturn) + np.random.randn() * 10

        # Return random policies
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
        # Does NOT set rreturn_est - leaves it as None
        policies = []
        for _ in range(num_arms):
            policy = _make_mock_policy(self._num_params)
            policy.set_params(np.random.uniform(-1, 1, self._num_params))
            policies.append(policy)
        return policies


def _make_mock_policy(num_params):
    """Create a mock policy for testing."""
    env_conf = get_env_conf("f:sphere-2d", problem_seed=0, noise_seed_0=0)
    return default_policy(env_conf)


def _run_optimizer(
    *,
    num_iterations: int,
    num_denoise_passive: int | None,
    designer_provides_rreturn_est: bool,
) -> list[float]:
    """Run optimizer and return list of ret_eval values."""
    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)
    collector = Collector()

    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=3,
        num_denoise_passive=num_denoise_passive,
    )

    if designer_provides_rreturn_est:
        designer = MockDesignerWithRreturnEst(num_params=2)
    else:
        designer = MockDesignerWithoutRreturnEst(num_params=2)

    opt._opt_designers = [designer]
    opt._trace = []
    opt._t_0 = 1.0
    opt._cum_dt_proposing = 0.0

    ret_eval_values = []
    for _ in range(num_iterations):
        trace = opt.iterate()
        ret_eval_values.append(trace[-1].rreturn)

    return ret_eval_values


def _is_monotonically_increasing(values: list[float]) -> bool:
    """Check if values are monotonically non-decreasing."""
    for i in range(1, len(values)):
        if values[i] < values[i - 1] - 1e-9:  # tolerance for floating point
            return False
    return True


class TestRetEvalFrozenNoise:
    """Tests for frozen noise: ret_eval should be monotonically increasing."""

    def test_frozen_noise_with_rreturn_est(self):
        """Frozen noise + designer provides rreturn_est: ret_eval monotonic."""
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=5,
            num_denoise_passive=None,  # frozen noise
            designer_provides_rreturn_est=True,
        )

        assert len(ret_eval_values) == 5
        assert _is_monotonically_increasing(ret_eval_values), f"ret_eval should be monotonically increasing for frozen noise, but got: {ret_eval_values}"

    def test_frozen_noise_without_rreturn_est(self):
        """Frozen noise + designer does NOT provide rreturn_est: ret_eval monotonic."""
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=5,
            num_denoise_passive=None,  # frozen noise
            designer_provides_rreturn_est=False,
        )

        assert len(ret_eval_values) == 5
        assert _is_monotonically_increasing(ret_eval_values), f"ret_eval should be monotonically increasing for frozen noise, but got: {ret_eval_values}"


class TestRetEvalNaturalNoise:
    """Tests for natural noise: ret_eval CAN fluctuate (not required to be monotonic)."""

    def test_natural_noise_with_rreturn_est(self):
        """Natural noise + designer provides rreturn_est: test runs without error."""
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=5,
            num_denoise_passive=1,  # natural noise
            designer_provides_rreturn_est=True,
        )

        assert len(ret_eval_values) == 5
        # For natural noise, we don't assert monotonicity - it's allowed to fluctuate
        # Just verify all values are finite
        assert all(np.isfinite(v) for v in ret_eval_values)

    def test_natural_noise_without_rreturn_est(self):
        """Natural noise + designer does NOT provide rreturn_est: test runs without error."""
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=5,
            num_denoise_passive=1,  # natural noise
            designer_provides_rreturn_est=False,
        )

        assert len(ret_eval_values) == 5
        # For natural noise, we don't assert monotonicity - it's allowed to fluctuate
        # Just verify all values are finite
        assert all(np.isfinite(v) for v in ret_eval_values)


class TestRetEvalEdgeCases:
    """Edge case tests for ret_eval behavior."""

    def test_frozen_noise_first_iteration(self):
        """First iteration should set y_best correctly."""
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=1,
            num_denoise_passive=None,
            designer_provides_rreturn_est=True,
        )

        assert len(ret_eval_values) == 1
        assert np.isfinite(ret_eval_values[0])

    def test_frozen_noise_improvement_tracked(self):
        """When a better rreturn is found, y_best should update."""
        np.random.seed(123)  # Different seed for variety
        ret_eval_values = _run_optimizer(
            num_iterations=10,
            num_denoise_passive=None,
            designer_provides_rreturn_est=True,
        )

        # Should see at least one improvement over 10 iterations
        assert ret_eval_values[-1] >= ret_eval_values[0]
        assert _is_monotonically_increasing(ret_eval_values)
