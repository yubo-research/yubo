import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestTrajectory:
    def test_trajectory_dataclass(self):
        from optimizer.trajectories import Trajectory

        traj = Trajectory(rreturn=10.5, states=np.array([[1, 2], [3, 4]]), actions=np.array([[0.1], [0.2]]))
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
            Datum(designer=None, policy=mock_policy_1, expected_acqf=None, trajectory=Trajectory(rreturn=5.0, states=np.array([]), actions=np.array([]))),
            Datum(designer=None, policy=mock_policy_2, expected_acqf=None, trajectory=Trajectory(rreturn=10.0, states=np.array([]), actions=np.array([]))),
        ]

        policy, ret = abo(data)
        assert policy == mock_policy_2
        assert ret == 10.0


class TestCenterDesigner:
    def test_init(self):
        from optimizer.center_designer import CenterDesigner

        mock_policy = MagicMock()
        cd = CenterDesigner(mock_policy)
        assert cd._policy == mock_policy

    def test_call(self):
        from optimizer.center_designer import CenterDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
        policy = default_policy(env_conf)

        cd = CenterDesigner(policy)
        policies = cd(None, num_arms=1)

        assert len(policies) == 1
        params = policies[0].get_params()
        assert len(params) == 2
        np.testing.assert_array_almost_equal(params, [0.0, 0.0])

    def test_call_asserts_num_arms(self):
        from optimizer.center_designer import CenterDesigner

        cd = CenterDesigner(MagicMock())
        with pytest.raises(AssertionError):
            cd(None, num_arms=2)


class TestRandomDesigner:
    def test_init(self):
        from optimizer.random_designer import RandomDesigner

        mock_policy = MagicMock()
        rd = RandomDesigner(mock_policy)
        assert rd._policy == mock_policy

    def test_call(self):
        import common.all_bounds as all_bounds
        from optimizer.random_designer import RandomDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
        policy = default_policy(env_conf)

        rd = RandomDesigner(policy)
        policies = rd(None, num_arms=3)

        assert len(policies) == 3
        for p in policies:
            params = p.get_params()
            assert len(params) == 2
            assert all(all_bounds.p_low <= x <= all_bounds.p_high for x in params)

    def test_call_with_telemetry(self):
        from optimizer.random_designer import RandomDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
        policy = default_policy(env_conf)

        mock_telemetry = MagicMock()

        rd = RandomDesigner(policy)
        rd(None, num_arms=1, telemetry=mock_telemetry)

        mock_telemetry.set_dt_fit.assert_called_once_with(0)
        mock_telemetry.set_dt_select.assert_called_once()


class TestSobolDesigner:
    def test_init(self):
        from optimizer.sobol_designer import SobolDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-3d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        sd = SobolDesigner(policy)
        assert sd._policy == policy
        assert sd.seed == 42 + 12345

    def test_call(self):
        from optimizer.sobol_designer import SobolDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        sd = SobolDesigner(policy)
        policies = sd(None, num_arms=5)

        assert len(policies) == 5
        assert len(sd.fig_last_arms) == 5

    def test_estimate(self):
        from optimizer.sobol_designer import SobolDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        sd = SobolDesigner(policy)
        result = sd.estimate([], [[1, 2], [3, 4]])
        assert result == [None, None]


class TestLHDDesigner:
    def test_call(self):
        from optimizer.lhd_designer import LHDDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        lhd = LHDDesigner(policy)
        policies = lhd(None, num_arms=2)

        assert len(policies) == 2


class TestOptunaDesigner:
    def test_init(self):
        from optimizer.optuna_designer import OptunaDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        od = OptunaDesigner(policy)
        assert od._policy == policy

    def test_call(self):
        from optimizer.optuna_designer import OptunaDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        od = OptunaDesigner(policy)
        policies = od([], num_arms=2)

        assert len(policies) == 2


class TestAxDesigner:
    def test_init(self):
        from optimizer.ax_designer import AxDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        ad = AxDesigner(policy)
        assert ad._policy == policy
        assert ad._ax_client is None

    def test_call_asserts_num_arms(self):
        from optimizer.ax_designer import AxDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        ad = AxDesigner(policy)
        with pytest.raises(AssertionError):
            ad([], num_arms=2)


class TestBTDesigner:
    def test_init(self):
        from optimizer.bt_designer import BTDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        bd = BTDesigner(
            policy,
            lambda m: MagicMock(),
            num_restarts=10,
            raw_samples=10,
            start_at_max=False,
        )
        assert bd._policy == policy
        assert bd._num_restarts == 10

    def test_call_sobol_init(self):
        from optimizer.bt_designer import BTDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        bd = BTDesigner(
            policy,
            lambda m: MagicMock(),
            num_restarts=10,
            raw_samples=10,
            start_at_max=False,
            init_sobol=5,
        )

        policies = bd([], num_arms=1)
        assert len(policies) == 1


class TestTurboRefDesigner:
    def test_init(self):
        from optimizer.turbo_ref_designer import TuRBORefDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        trd = TuRBORefDesigner(policy, num_trust_regions=1, num_init=10)
        assert trd._policy == policy
        assert trd._num_init == 10


class TestTurboENNDesigner:
    def test_init(self):
        from optimizer.turbo_enn_designer import TurboENNDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-5d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        ted = TurboENNDesigner(policy, turbo_mode="turbo-enn", k=10)
        assert ted._policy == policy
        assert ted._k == 10


class TestTurboYUBODesigner:
    def test_init(self):
        from optimizer.turbo_yubo_designer import TurboYUBODesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-5d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        tyd = TurboYUBODesigner(policy)
        assert tyd._policy == policy


class TestMTSDesigner:
    def test_init(self):
        from optimizer.mts_designer import MTSDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        md = MTSDesigner(policy, init_style="find")
        assert md._policy == policy
        assert md._init_style == "find"


class TestENNDesigner:
    def test_init(self):
        from acq.acq_enn import ENNConfig
        from optimizer.enn_designer import ENNDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-5d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        config = ENNConfig(k=10)
        ed = ENNDesigner(policy, config)
        assert ed._policy == policy


class TestVHDDesigner:
    def test_init(self):
        from acq.acq_vhd import VHDConfig
        from optimizer.vhd_designer import VHDDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-5d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        config = VHDConfig(k=10)
        vd = VHDDesigner(policy, config)
        assert vd._policy == policy


class TestCMADesigner:
    def test_init(self):
        from optimizer.cma_designer import CMAESDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        cd = CMAESDesigner(policy)
        assert cd._policy == policy

    def test_call_asserts_num_arms(self):
        from optimizer.cma_designer import CMAESDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        cd = CMAESDesigner(policy)
        with pytest.raises(AssertionError):
            cd([], num_arms=1)


class TestMCMCBODesigner:
    def test_init(self):
        from optimizer.mcmc_bo_designer import MCMCBODesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        md = MCMCBODesigner(policy, num_init=10)
        assert md._policy == policy
        assert md._num_init == 10


class TestVecchiaDesigner:
    def test_init(self):
        from optimizer.vecchia_designer import VecchiaDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        vd = VecchiaDesigner(policy, num_candidates_per_arm=100)
        assert vd._policy == policy
        assert vd._num_candidates_per_arm == 100


class TestOptimizer:
    def test_init(self):
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
        assert opt._num_arms == 3
        assert opt.num_params == 2

    def test_initialize(self):
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
        opt.initialize("random")
        assert opt._opt_designers is not None
        assert isinstance(opt._opt_designers, list)
        assert len(opt._opt_designers) >= 1
        assert opt._trace == []
        assert opt._t_0 > 0

    def test_iterate(self):
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

    def test_iterate_multiple(self):
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
        opt.iterate()
        trace = opt.iterate()
        assert len(trace) == 3
        assert opt._i_iter == 3
        assert len(opt._data) == 6

    def test_stop(self):
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


class TestDesigners:
    def test_init(self):
        from optimizer.designers import Designers
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        d = Designers(policy, num_arms=5)
        assert d._policy == policy
        assert d._num_arms == 5

    def test_create_random(self):
        from optimizer.designers import Designers
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        d = Designers(policy, num_arms=5)
        designer = d.create("random")
        assert designer is not None

    def test_create_sobol(self):
        from optimizer.designers import Designers
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        d = Designers(policy, num_arms=5)
        designer = d.create("sobol")
        assert designer is not None

    def test_create_invalid(self):
        from optimizer.designers import Designers, NoSuchDesignerError
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        d = Designers(policy, num_arms=5)
        with pytest.raises(NoSuchDesignerError):
            d.create("invalid_designer_name")
