from common.collector import Collector
from optimizer.center_designer import CenterDesigner
from optimizer.iteration_step import (
    IterateResult,
    evaluate_policies,
    iterate_step,
    propose_policies,
)
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf


def test_iteration_step_smoke_sequential():
    env_conf = get_env_conf("f:sphere-2d", problem_seed=0, noise_seed_0=0)
    policy = default_policy(env_conf)
    opt = Optimizer(
        Collector(),
        env_conf=env_conf,
        env_tag="f:sphere-2d",
        policy=policy,
        num_arms=1,
    )
    designer = CenterDesigner(policy)

    policies, dt_prop = propose_policies(opt, designer, num_arms=1)
    assert len(policies) == 1
    assert dt_prop >= 0.0

    data, dt_eval = evaluate_policies(opt, designer=designer, policies=policies)
    assert len(data) == 1
    assert dt_eval >= 0.0

    result = iterate_step(opt, designer, num_arms=1)
    assert isinstance(result, IterateResult)
    assert len(result.data) == 1
    assert result.dt_prop >= 0.0
    assert result.dt_eval >= 0.0
