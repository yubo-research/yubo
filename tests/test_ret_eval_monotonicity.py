"""Regression tests for ret_eval behavior with different noise types and designer modes."""

import numpy as np
from ret_eval_monotonicity_fixtures import MockDesignerWithoutRreturnEst, MockDesignerWithRreturnEst

from common.collector import Collector
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf


def _run_optimizer(
    *,
    num_iterations: int,
    num_denoise_passive: int | None,
    designer_provides_rreturn_est: bool,
) -> list[float]:
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
    for i in range(1, len(values)):
        if values[i] < values[i - 1] - 1e-9:
            return False
    return True


class TestRetEvalFrozenNoise:
    def test_frozen_noise_with_rreturn_est(self):
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=5,
            num_denoise_passive=None,
            designer_provides_rreturn_est=True,
        )

        assert len(ret_eval_values) == 5
        assert _is_monotonically_increasing(ret_eval_values), f"ret_eval should be monotonically increasing for frozen noise, but got: {ret_eval_values}"

    def test_frozen_noise_without_rreturn_est(self):
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=5,
            num_denoise_passive=None,
            designer_provides_rreturn_est=False,
        )

        assert len(ret_eval_values) == 5
        assert _is_monotonically_increasing(ret_eval_values), f"ret_eval should be monotonically increasing for frozen noise, but got: {ret_eval_values}"


class TestRetEvalNaturalNoise:
    def test_natural_noise_with_rreturn_est(self):
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=5,
            num_denoise_passive=1,
            designer_provides_rreturn_est=True,
        )

        assert len(ret_eval_values) == 5
        assert all(np.isfinite(v) for v in ret_eval_values)

    def test_natural_noise_without_rreturn_est(self):
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=5,
            num_denoise_passive=1,
            designer_provides_rreturn_est=False,
        )

        assert len(ret_eval_values) == 5
        assert all(np.isfinite(v) for v in ret_eval_values)


class TestRetEvalEdgeCases:
    def test_frozen_noise_first_iteration(self):
        np.random.seed(42)
        ret_eval_values = _run_optimizer(
            num_iterations=1,
            num_denoise_passive=None,
            designer_provides_rreturn_est=True,
        )

        assert len(ret_eval_values) == 1
        assert np.isfinite(ret_eval_values[0])

    def test_frozen_noise_improvement_tracked(self):
        np.random.seed(123)
        ret_eval_values = _run_optimizer(
            num_iterations=10,
            num_denoise_passive=None,
            designer_provides_rreturn_est=True,
        )

        assert ret_eval_values[-1] >= ret_eval_values[0]
        assert _is_monotonically_increasing(ret_eval_values)
