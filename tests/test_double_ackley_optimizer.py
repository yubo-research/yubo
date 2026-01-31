import numpy as np
import pytest


@pytest.mark.parametrize(
    "designer_name,num_arms",
    [
        ("sobol", 4),
        ("morbo-enn-fit/acq_type=ucb", 1),
    ],
)
def test_double_ackley_runs(designer_name, num_arms):
    from common.collector import Collector
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:doubleackley-20d", problem_seed=0, noise_seed_0=17)
    policy = default_policy(env_conf)

    opt = Optimizer(
        Collector(),
        env_conf=env_conf,
        policy=policy,
        num_arms=num_arms,
        num_denoise_measurement=None,
        num_denoise_passive=None,
    )
    trace = opt.collect_trace(designer_name=designer_name, max_iterations=3, max_proposal_seconds=np.inf)
    assert len(trace) == 3
    assert np.isfinite(trace[-1].rreturn)
