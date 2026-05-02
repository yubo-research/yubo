import time
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def sphere_2d_setup():
    """Common test setup for sphere-2d environment."""
    from common.collector import Collector
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)
    collector = Collector()
    return env_conf, policy, collector


def test_optimizer_init(sphere_2d_setup):
    from optimizer.optimizer import Optimizer

    env_conf, policy, collector = sphere_2d_setup
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=3,
    )
    assert opt._num_arms == 3
    assert opt.num_params == 2


def test_optimizer_initialize(sphere_2d_setup):
    from optimizer.optimizer import Optimizer

    env_conf, policy, collector = sphere_2d_setup
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=3,
    )
    opt.initialize("random")
    assert opt._opt_designers is not None
    assert isinstance(opt._opt_designers, list)
    assert len(opt._opt_designers) >= 1
    assert opt._trace == []
    assert opt._t_0 > 0


def test_optimizer_iterate(sphere_2d_setup):
    from optimizer.optimizer import Optimizer

    env_conf, policy, collector = sphere_2d_setup
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=3,
    )
    opt.initialize("random")
    trace = opt.iterate()
    assert len(trace) == 1
    assert trace[0].rreturn is not None
    assert trace[0].dt_prop >= 0
    assert trace[0].dt_eval >= 0
    assert opt._i_iter == 1
    assert len(opt._data) == 3
    assert opt.best_policy is not None
    assert opt.best_datum is not None


def test_optimizer_iterate_multiple(sphere_2d_setup):
    from optimizer.optimizer import Optimizer

    env_conf, policy, collector = sphere_2d_setup
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=2,
    )
    opt.initialize("random")
    opt.iterate()
    opt.iterate()
    trace = opt.iterate()
    assert len(trace) == 3
    assert opt._i_iter == 3
    assert len(opt._data) == 6


def test_optimizer_stop(sphere_2d_setup):
    from optimizer.optimizer import Optimizer

    env_conf, policy, collector = sphere_2d_setup
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=2,
    )
    opt.initialize("random")
    opt.iterate()
    opt.stop()


def test_optimizer_stop_with_designer_stop_method():
    from common.collector import Collector
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    collector = Collector()
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=2,
    )
    opt.initialize("random")
    opt.iterate()

    mock_designer = MagicMock()
    mock_designer.stop = MagicMock()
    opt._opt_designers = [mock_designer]
    opt.stop()
    mock_designer.stop.assert_called_once()


def test_optimizer_collect_trace():
    from common.collector import Collector
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    collector = Collector()
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=2,
    )
    trace = opt.collect_trace("random", max_iterations=3)
    assert len(trace) == 3
    assert opt._i_iter == 3
    for entry in trace:
        assert entry.rreturn is not None
        assert entry.dt_prop >= 0
        assert entry.dt_eval >= 0


def test_optimizer_collect_trace_max_proposal_seconds():
    from common.collector import Collector
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    collector = Collector()
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=2,
    )
    trace = opt.collect_trace("random", max_iterations=1000, max_proposal_seconds=0.001)
    assert len(trace) < 1000


def test_optimizer_r_best_est_updates():
    from common.collector import Collector
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    collector = Collector()
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        policy=policy,
        num_arms=3,
    )
    initial_best = opt.r_best_est
    opt.initialize("random")
    opt.iterate()
    assert opt.r_best_est > initial_best


def test_optimizer_iterate_internal():
    from unittest.mock import MagicMock

    from common.collector import Collector
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)
    collector = Collector()
    opt = Optimizer(collector, env_conf=env_conf, policy=policy, num_arms=1)

    mock_designer = MagicMock()
    mock_designer.return_value = [policy]

    data, dt_prop, dt_eval = opt._iterate(mock_designer, num_arms=1)

    assert len(data) == 1
    assert dt_prop >= 0
    assert dt_eval >= 0
    mock_designer.assert_called_once()


def test_optimizer_iterate_multiobjective():
    from unittest.mock import MagicMock

    import numpy as np

    from common.collector import Collector
    from optimizer.datum import Datum
    from optimizer.optimizer import Optimizer
    from optimizer.trajectory import Trajectory
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)
    collector = Collector()
    opt = Optimizer(collector, env_conf=env_conf, policy=policy, num_arms=2)

    # Mock designer to return policies
    mock_designer = MagicMock()
    mock_designer.return_value = [policy, policy]
    opt._opt_designers = [mock_designer]
    opt._t_0 = time.time()
    opt._trace = []  # Initialize trace

    # Mock _iterate to return multi-objective data
    traj1 = Trajectory(rreturn=np.array([1.0, 2.0]), states=np.array([]), actions=np.array([]))
    traj2 = Trajectory(rreturn=np.array([2.0, 1.0]), states=np.array([]), actions=np.array([]))
    data = [
        Datum(mock_designer, policy, None, traj1),
        Datum(mock_designer, policy, None, traj2),
    ]

    opt._iterate = MagicMock(return_value=(data, 0.1, 0.1))

    # Manually set a 2D reference point to match the 2D mock rewards
    opt._ref_point = np.array([0.0, 0.0])

    # We need a reference point for HV calculation.
    # iterate() will create one if it's None, but we pre-set it.
    opt.iterate()

    assert opt.y_best is not None
    assert len(opt.y_best) == 2
    assert opt.r_best_est > -1e99  # HV should be computed


def test_pareto_mask_max():
    from optimizer.optimizer import _pareto_mask_max

    y = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],  # dominates [1, 1]
            [2.0, 1.0],  # dominated by [2, 2]
            [1.0, 2.0],  # dominated by [2, 2]
            [3.0, 0.5],  # non-dominated
        ]
    )
    mask = _pareto_mask_max(y)
    expected = np.array([False, True, False, False, True])
    np.testing.assert_array_equal(mask, expected)


def test_pareto_mask_min():
    from optimizer.optimizer import _pareto_mask_min

    y = np.array(
        [
            [1.0, 1.0],  # dominates [2, 2]
            [2.0, 2.0],  # dominated by [1, 1]
            [0.5, 3.0],  # non-dominated
            [3.0, 0.5],  # non-dominated
        ]
    )
    mask = _pareto_mask_min(y)
    expected = np.array([True, False, True, True])
    np.testing.assert_array_equal(mask, expected)


def test_designers_init():
    from optimizer.designers import Designers
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    d = Designers(policy, num_arms=5)
    assert d._policy == policy
    assert d._num_arms == 5


def test_designers_create_random():
    from optimizer.designers import Designers
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    d = Designers(policy, num_arms=5)
    designer = d.create("random")
    assert designer is not None


def test_designers_create_sobol():
    from optimizer.designers import Designers
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    d = Designers(policy, num_arms=5)
    designer = d.create("sobol")
    assert designer is not None


def test_designers_create_invalid():
    from optimizer.designers import Designers, NoSuchDesignerError
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    d = Designers(policy, num_arms=5)
    with pytest.raises(NoSuchDesignerError):
        d.create("invalid_designer_name")
