import threading
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


class TestTrajectory:
    def test_trajectory_dataclass(self):
        from optimizer.trajectories import Trajectory

        traj = Trajectory(
            rreturn=10.5,
            states=np.array([[1, 2], [3, 4]]),
            actions=np.array([[0.1], [0.2]]),
        )
        assert traj.rreturn == 10.5
        assert traj.states.shape == (2, 2)
        assert traj.actions.shape == (2, 1)
        assert traj.rreturn_se is None

    def test_trajectory_with_se(self):
        from optimizer.trajectories import Trajectory

        traj = Trajectory(rreturn=10.5, states=np.array([]), actions=np.array([]), rreturn_se=0.5)
        assert traj.rreturn_se == 0.5


class TestCollectTrajectory:
    def test_collect_trajectory(self):
        from optimizer.trajectories import Trajectory, collect_trajectory
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        traj = collect_trajectory(env_conf, policy, noise_seed=123)

        assert isinstance(traj, Trajectory)
        assert isinstance(traj.rreturn, float)
        assert traj.rreturn_se is None


class TestDatum:
    def test_datum_dataclass(self):
        from optimizer.datum import Datum
        from optimizer.trajectories import Trajectory

        traj = Trajectory(rreturn=5.0, states=np.array([]), actions=np.array([]))
        datum = Datum(designer="test", policy=MagicMock(), expected_acqf=0.5, trajectory=traj)
        assert datum.designer == "test"
        assert datum.expected_acqf == 0.5
        assert datum.trajectory.rreturn == 5.0


class TestAskTellInverter:
    def test_init(self):
        from optimizer.ask_tell_inverter import AskTellInverter

        ati = AskTellInverter(timeout_seconds=5)
        assert ati._timeout_seconds == 5
        assert ati._running is True

    def test_stop(self):
        from optimizer.ask_tell_inverter import AskTellInverter

        ati = AskTellInverter()
        ati.stop()
        assert ati._running is False

    def test_ask_tell_flow(self):
        from optimizer.ask_tell_inverter import AskTellInverter

        ati = AskTellInverter(timeout_seconds=1)
        x_vals = [1.0, 2.0, 3.0]
        y_vals = [10.0, 20.0, 30.0]

        def caller():
            result = ati(x_vals)
            assert result == y_vals

        thread = threading.Thread(target=caller)
        thread.start()

        time.sleep(0.01)
        asked = ati.ask()
        assert asked == x_vals
        ati.tell(y_vals)
        thread.join(timeout=1)

    def test_timeout_error(self):
        from optimizer.ask_tell_inverter import AskTellInverter, ATITimeoutError

        ati = AskTellInverter(timeout_seconds=0.01)
        with pytest.raises(ATITimeoutError):
            ati.ask()

    def test_stopped_error(self):
        from optimizer.ask_tell_inverter import AskTellInverter, ATIStopped

        ati = AskTellInverter(timeout_seconds=1)

        def stopper():
            time.sleep(0.01)
            ati.stop()

        thread = threading.Thread(target=stopper)
        thread.start()
        with pytest.raises(ATIStopped):
            ati.ask()
        thread.join()


class TestArmBestObs:
    def test_call(self):
        from optimizer.arm_best_obs import ArmBestObs
        from optimizer.datum import Datum
        from optimizer.trajectories import Trajectory

        abo = ArmBestObs()

        mock_policy_1 = MagicMock()
        mock_policy_2 = MagicMock()
        data = [
            Datum(
                designer=None,
                policy=mock_policy_1,
                expected_acqf=None,
                trajectory=Trajectory(rreturn=5.0, states=np.array([]), actions=np.array([])),
            ),
            Datum(
                designer=None,
                policy=mock_policy_2,
                expected_acqf=None,
                trajectory=Trajectory(rreturn=10.0, states=np.array([]), actions=np.array([])),
            ),
        ]

        policy, ret = abo(data)
        assert policy == mock_policy_2
        assert ret == 10.0


class TestOptimizer:
    def test_init(self, sphere_2d_setup):
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

    def test_initialize(self, sphere_2d_setup):
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

    def test_iterate(self, sphere_2d_setup):
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

    def test_iterate_multiple(self, sphere_2d_setup):
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

    def test_stop(self, sphere_2d_setup):
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

    def test_stop_with_designer_stop_method(self):
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

    def test_collect_trace(self):
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

    def test_collect_trace_max_proposal_seconds(self):
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

    def test_r_best_est_updates(self):
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

    def test_iterate_multiobjective(self, monkeypatch):
        import optimizer.optimizer as optimizer_mod
        from common.collector import Collector
        from optimizer.datum import Datum
        from optimizer.optimizer import Optimizer
        from optimizer.trajectories import Trajectory
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)
        collector = Collector()
        opt = Optimizer(collector, env_conf=env_conf, policy=policy, num_arms=2)

        opt._opt_designers = [MagicMock()]
        opt._t_0 = time.time()
        opt._trace = []

        traj1 = Trajectory(rreturn=np.array([1.0, 2.0]), states=np.array([]), actions=np.array([]))
        traj2 = Trajectory(rreturn=np.array([2.0, 1.0]), states=np.array([]), actions=np.array([]))
        data = [
            Datum(None, policy, None, traj1),
            Datum(None, policy, None, traj2),
        ]

        monkeypatch.setattr(optimizer_mod, "iterate_step", lambda *_a, **_k: (data, 0.1, 0.1))
        opt._ref_point = np.array([0.0, 0.0])

        opt.iterate()

        assert opt.y_best is not None
        assert len(opt.y_best) == 2
        assert opt.r_best_est > -1e99


class TestParetoMask:
    def test_pareto_mask_max(self):
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

    def test_pareto_mask_min(self):
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
