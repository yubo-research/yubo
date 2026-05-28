import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_trajectory_dataclass():
    from optimizer.trajectory import Trajectory

    traj = Trajectory(
        rreturn=10.5,
        states=np.array([[1, 2], [3, 4]]),
        actions=np.array([[0.1], [0.2]]),
    )
    assert traj.rreturn == 10.5
    assert traj.states.shape == (2, 2)
    assert traj.actions.shape == (2, 1)
    assert traj.rreturn_se is None


def test_trajectory_with_se():
    from optimizer.trajectory import Trajectory

    traj = Trajectory(rreturn=10.5, states=np.array([]), actions=np.array([]), rreturn_se=0.5)
    assert traj.rreturn_se == 0.5


def test_collect_trajectory_collect_trajectory():
    from optimizer.trajectories import collect_trajectory
    from optimizer.trajectory import Trajectory
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    traj = collect_trajectory(env_conf, policy, noise_seed=123)

    assert isinstance(traj, Trajectory)
    assert isinstance(traj.rreturn, float)
    assert traj.rreturn_se is None


def test_datum_dataclass():
    from optimizer.datum import Datum
    from optimizer.trajectory import Trajectory

    traj = Trajectory(rreturn=5.0, states=np.array([]), actions=np.array([]))
    datum = Datum(designer="test", policy=MagicMock(), expected_acqf=0.5, trajectory=traj)
    assert datum.designer == "test"
    assert datum.expected_acqf == 0.5
    assert datum.trajectory.rreturn == 5.0


def test_ask_tell_inverter_init():
    from optimizer.ask_tell_inverter import AskTellInverter

    ati = AskTellInverter(timeout_seconds=5)
    assert ati._timeout_seconds == 5
    assert ati._running is True


def test_ask_tell_inverter_stop():
    from optimizer.ask_tell_inverter import AskTellInverter

    ati = AskTellInverter()
    ati.stop()
    assert ati._running is False


def test_ask_tell_inverter_ask_tell_flow():
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


def test_ask_tell_inverter_timeout_error():
    from optimizer.ask_tell_inverter import AskTellInverter, ATITimeoutError

    ati = AskTellInverter(timeout_seconds=0.01)
    with pytest.raises(ATITimeoutError):
        ati.ask()


def test_ask_tell_inverter_stopped_error():
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


def test_arm_best_obs_call():
    from optimizer.arm_best_obs import ArmBestObs
    from optimizer.datum import Datum
    from optimizer.trajectory import Trajectory

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


def test_center_designer_init():
    from optimizer.center_designer import CenterDesigner

    mock_policy = MagicMock()
    cd = CenterDesigner(mock_policy)
    assert cd._policy == mock_policy


def test_center_designer_call():
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


def test_center_designer_call_asserts_num_arms():
    from optimizer.center_designer import CenterDesigner

    cd = CenterDesigner(MagicMock())
    with pytest.raises(AssertionError):
        cd(None, num_arms=2)


def test_random_designer_init():
    from optimizer.random_designer import RandomDesigner

    mock_policy = MagicMock()
    rd = RandomDesigner(mock_policy)
    assert rd._policy == mock_policy


def test_random_designer_call():
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


def test_random_designer_call_with_telemetry():
    from optimizer.random_designer import RandomDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
    policy = default_policy(env_conf)

    mock_telemetry = MagicMock()

    rd = RandomDesigner(policy)
    rd(None, num_arms=1, telemetry=mock_telemetry)

    mock_telemetry.set_dt_fit.assert_called_once_with(0)
    mock_telemetry.set_dt_select.assert_called_once()


def test_sobol_designer_init():
    from optimizer.sobol_designer import SobolDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-3d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    sd = SobolDesigner(policy)
    assert sd._policy == policy
    assert sd.seed == 42 + 12345


def test_sobol_designer_call():
    from optimizer.sobol_designer import SobolDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    sd = SobolDesigner(policy)
    policies = sd(None, num_arms=5)

    assert len(policies) == 5
    assert len(sd.fig_last_arms) == 5


def test_sobol_designer_estimate():
    from optimizer.sobol_designer import SobolDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    sd = SobolDesigner(policy)
    result = sd.estimate([], [[1, 2], [3, 4]])
    assert result == [None, None]


def test_l_h_d_designer_call():
    from optimizer.lhd_designer import LHDDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    lhd = LHDDesigner(policy)
    policies = lhd(None, num_arms=2)

    assert len(policies) == 2


def test_optuna_designer_init():
    from optimizer.optuna_designer import OptunaDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    od = OptunaDesigner(policy)
    assert od._policy == policy


def test_optuna_designer_call():
    from optimizer.optuna_designer import OptunaDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    od = OptunaDesigner(policy)
    policies = od([], num_arms=2)

    assert len(policies) == 2


def test_ax_designer_init():
    from optimizer.ax_designer import AxDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    ad = AxDesigner(policy)
    assert ad._policy == policy
    assert ad._ax_client is None


def test_ax_designer_call_asserts_num_arms():
    from optimizer.ax_designer import AxDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    ad = AxDesigner(policy)
    with pytest.raises(AssertionError):
        ad([], num_arms=2)


def test_b_t_designer_init():
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


def test_b_t_designer_call_sobol_init():
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


def test_turbo_ref_designer_init():
    from optimizer.turbo_ref_designer import TuRBORefDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    trd = TuRBORefDesigner(policy, num_trust_regions=1, num_init=10)
    assert trd._policy == policy
    assert trd._num_init == 10


def test_turbo_e_n_n_designer_init():
    from optimizer.turbo_enn_designer import TurboENNDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-5d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    ted = TurboENNDesigner(policy, turbo_mode="turbo-enn", k=10)
    assert ted._policy == policy
    assert ted._k == 10


def test_ppo_designer_registered_under_opt_name_ppo():
    from optimizer.designers import Designers
    from optimizer.ppo_designer import PPOACDesigner, PPODesigner
    from problems.problem import build_problem

    problem = build_problem("pend", "actor-critic-mlp-16-8", problem_seed=0, noise_seed_0=0)
    policy = problem.build_policy()
    designer = Designers(policy, num_arms=1, env_conf=problem.env).create("ppo")
    assert isinstance(designer, PPODesigner)
    assert isinstance(designer, PPOACDesigner)


def test_ppo_designers_registered_under_split_names():
    from optimizer.designers import Designers
    from optimizer.ppo_designer import PPOACDesigner
    from optimizer.ppo_pg_designer import PPOPGDesigner
    from problems.problem import build_problem

    ac_problem = build_problem("pend", "actor-critic-mlp-16-8", problem_seed=0, noise_seed_0=0)
    ac_policy = ac_problem.build_policy()
    ac_designer = Designers(ac_policy, num_arms=1, env_conf=ac_problem.env).create("ppo-ac")
    assert isinstance(ac_designer, PPOACDesigner)

    pg_problem = build_problem("pend", "actor-mlp-16-8", problem_seed=0, noise_seed_0=0)
    pg_policy = pg_problem.build_policy()
    pg_designer = Designers(pg_policy, num_arms=1, env_conf=pg_problem.env).create("ppo-pg")
    assert isinstance(pg_designer, PPOPGDesigner)


def test_m_t_s_designer_init():
    from optimizer.mts_designer import MTSDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    md = MTSDesigner(policy, init_style="find")
    assert md._policy == policy
    assert md._init_style == "find"


def _sphere_policy_for_cma():
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    return env_conf, default_policy(env_conf)


def test_c_m_a_designer_init():
    from optimizer.cma_designer import CMAESDesigner

    _env_conf, policy = _sphere_policy_for_cma()
    cd = CMAESDesigner(policy)
    assert cd._policy == policy


def test_c_m_a_designer_call_asserts_num_arms():
    from optimizer.cma_designer import CMAESDesigner

    _env_conf, policy = _sphere_policy_for_cma()
    cd = CMAESDesigner(policy)
    with pytest.raises(AssertionError):
        cd([], num_arms=1)


def test_m_c_m_c_b_o_designer_init():
    from optimizer.mcmc_bo_designer import MCMCBODesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    md = MCMCBODesigner(policy, num_init=10)
    assert md._policy == policy
    assert md._num_init == 10


def test_vecchia_designer_init():
    from optimizer.vecchia_designer import VecchiaDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
    policy = default_policy(env_conf)

    vd = VecchiaDesigner(policy, num_candidates_per_arm=100)
    assert vd._policy == policy
    assert vd._num_candidates_per_arm == 100
