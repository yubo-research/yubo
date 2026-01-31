import numpy as np

from common.collector import Collector
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf


class DesignerWithNullBestDatum:
    def __init__(self, base_designer):
        self._base = base_designer

    def __call__(self, data, num_arms, *, telemetry=None):
        return self._base(data, num_arms, telemetry=telemetry)

    def best_datum(self):
        return None


def test_update_best_when_designer_best_datum_returns_none():
    env_conf = get_env_conf("f:ackley-3d", problem_seed=0, noise_seed_0=0)
    policy = default_policy(env_conf)

    opt = Optimizer(
        Collector(),
        env_conf=env_conf,
        policy=policy,
        num_arms=1,
        num_denoise_measurement=None,
        num_denoise_passive=None,
    )

    from optimizer.designers import Designers

    designers = Designers(policy, 1)
    base_designer = designers.create("sobol")
    wrapped_designer = DesignerWithNullBestDatum(base_designer)

    opt._opt_designers = [wrapped_designer]
    opt._trace = []
    opt._t_0 = 0

    opt.iterate()

    assert opt.r_best_est > -1e50, f"r_best_est was not updated: {opt.r_best_est}"
    assert np.isfinite(opt.r_best_est), f"r_best_est is not finite: {opt.r_best_est}"


def test_update_best_fallback_to_batch_on_iter_zero():
    env_conf = get_env_conf("f:ackley-3d", problem_seed=42, noise_seed_0=0)
    policy = default_policy(env_conf)

    opt = Optimizer(
        Collector(),
        env_conf=env_conf,
        policy=policy,
        num_arms=2,
        num_denoise_measurement=None,
        num_denoise_passive=None,
    )

    opt.initialize("sobol")
    opt.iterate()

    assert opt.r_best_est > -1e50, f"r_best_est not updated on iter 0: {opt.r_best_est}"
    assert opt.y_best is not None
    assert opt.best_datum is not None
    assert opt.best_policy is not None
