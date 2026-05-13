from __future__ import annotations

import numpy as np

from common.collector import Collector
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf


def assert_short_optimizer_trace_finite(
    env_tag: str,
    *,
    designer_name: str,
    problem_seed: int = 0,
    noise_seed_0: int = 0,
    num_arms: int = 1,
    max_iterations: int = 3,
    policy_tag: str | None = None,
):
    env_conf = get_env_conf(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    policy = default_policy(env_conf)
    kw = dict(
        env_conf=env_conf,
        policy=policy,
        num_arms=num_arms,
        num_denoise_measurement=None,
        num_denoise_passive=None,
    )
    if policy_tag is not None:
        kw["policy_tag"] = policy_tag
    opt = Optimizer(Collector(), **kw)
    trace = opt.collect_trace(
        designer_name=designer_name,
        max_iterations=max_iterations,
        max_proposal_seconds=np.inf,
    )
    assert len(trace) == max_iterations
    assert np.isfinite(trace[-1].rreturn)
