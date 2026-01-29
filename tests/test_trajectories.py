import numpy as np


def test_trajectory_dataclass():
    from optimizer.trajectories import Trajectory

    traj = Trajectory(rreturn=1.5, states=np.array([1, 2]), actions=np.array([0, 1]))
    assert traj.rreturn == 1.5
    assert traj.rreturn_se is None
    assert traj.rreturn_est is None


def test_trajectory_get_decision_rreturn_no_est():
    from optimizer.trajectories import Trajectory

    traj = Trajectory(rreturn=1.5, states=np.array([1, 2]), actions=np.array([0, 1]))
    assert traj.get_decision_rreturn() == 1.5


def test_trajectory_get_decision_rreturn_with_est():
    from optimizer.trajectories import Trajectory

    traj = Trajectory(
        rreturn=1.5,
        states=np.array([1, 2]),
        actions=np.array([0, 1]),
        rreturn_est=2.0,
    )
    assert traj.get_decision_rreturn() == 2.0
