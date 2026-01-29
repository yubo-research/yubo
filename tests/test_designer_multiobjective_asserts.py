import numpy as np
import pytest


def test_optuna_designer_rejects_vector_returns():
    from optimizer.datum import Datum
    from optimizer.optuna_designer import OptunaDesigner
    from optimizer.trajectories import Trajectory
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=0, noise_seed_0=17)
    policy = default_policy(env_conf)
    designer = OptunaDesigner(policy)

    p = designer([], num_arms=1)[0]
    traj = Trajectory(
        rreturn=np.array([1.0, 2.0]), states=None, actions=None, rreturn_se=None
    )
    datum = Datum(designer=designer, policy=p, expected_acqf=None, trajectory=traj)

    with pytest.raises(AssertionError):
        designer([datum], num_arms=1)
