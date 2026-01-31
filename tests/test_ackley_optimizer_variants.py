import numpy as np
import pytest


@pytest.mark.parametrize("designer_name", ["vecchia", "turbo-zero"])
def test_ackley_3d_runs_with_optimizer(designer_name):
    from common.collector import Collector
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

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
    trace = opt.collect_trace(designer_name=designer_name, max_iterations=3, max_proposal_seconds=np.inf)
    assert len(trace) == 3
    assert np.isfinite(trace[-1].rreturn)
