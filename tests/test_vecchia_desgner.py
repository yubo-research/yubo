import numpy as np

from optimizer.datum import Datum
from optimizer.trajectories import Trajectory
from optimizer.vecchia_designer import VecchiaDesigner
from problems.env_conf import default_policy, get_env_conf


def _mk_policy(env_tag="f:ackley-5d", seed=123):
    ec = get_env_conf(env_tag, problem_seed=seed)
    return default_policy(ec)


def _mk_datum(designer, policy, ret):
    traj = Trajectory(float(ret), np.empty((0,)), np.empty((0,)))
    return Datum(designer, policy, None, traj)


def _params_array(policies):
    return np.stack([p.get_params() for p in policies], axis=0)


def test_vecchia_designer_no_data_returns_num_arms():
    policy = _mk_policy()
    d = VecchiaDesigner(policy)

    arms = d([], num_arms=4)
    assert len(arms) == 4
    P = _params_array(arms)
    assert P.shape == (4, policy.num_params())
    assert len(np.unique(P, axis=0)) == 4


def test_vecchia_designer_with_data_improves_shape_and_stays_in_bounds():
    policy = _mk_policy()
    d = VecchiaDesigner(policy)

    rng = np.random.default_rng(0)
    data = []
    for _ in range(6):
        p = policy.clone()
        params = rng.uniform(-1.0, 1.0, size=(policy.num_params(),))
        p.set_params(params)
        ret = float(np.linalg.norm(params))
        data.append(_mk_datum(d, p, ret))

    arms = d(data, num_arms=3)
    assert len(arms) == 3
    P = _params_array(arms)
    assert P.shape == (3, policy.num_params())
    assert np.all(P >= -1.0) and np.all(P <= 1.0)
    assert len(np.unique(P, axis=0)) == 3


def test_vecchia_designer_one_arm_no_data():
    policy = _mk_policy()
    d = VecchiaDesigner(policy)

    arms = d([], num_arms=1)
    assert len(arms) == 1
    P = _params_array(arms)
    assert P.shape == (1, policy.num_params())
    assert np.all(P >= -1.0) and np.all(P <= 1.0)
