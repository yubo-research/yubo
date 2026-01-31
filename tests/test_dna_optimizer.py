import numpy as np


def test_dna_runs_with_random_optimizer():
    from common.collector import Collector
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("dna", problem_seed=0, noise_seed_0=0)
    policy = default_policy(env_conf)

    opt = Optimizer(
        Collector(),
        env_conf=env_conf,
        policy=policy,
        num_arms=1,
        num_denoise_measurement=None,
        num_denoise_passive=None,
    )
    trace = opt.collect_trace(designer_name="random", max_iterations=3, max_proposal_seconds=np.inf)
    assert len(trace) == 3
    assert np.isfinite(trace[-1].rreturn)
