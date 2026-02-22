from types import SimpleNamespace

import gymnasium as gym
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


def test_collect_trajectory_handles_dict_pixel_observation():
    from optimizer.trajectories import collect_trajectory

    class _Policy:
        def __init__(self):
            self.last_obs = None

        def __call__(self, obs):
            self.last_obs = np.asarray(obs)
            return 0

    class _DummyEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self):
            self.observation_space = gym.spaces.Dict(
                {
                    "pixels": gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
                }
            )
            self.action_space = gym.spaces.Discrete(2)
            self._done = False

        def reset(self, *, seed=None, options=None):
            _ = seed
            _ = options
            self._done = False
            return {"pixels": np.zeros((84, 84, 3), dtype=np.uint8)}, {}

        def step(self, action):
            _ = action
            self._done = True
            return (
                {"pixels": np.zeros((84, 84, 3), dtype=np.uint8)},
                1.0,
                True,
                False,
                {},
            )

        def close(self):
            return None

    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(max_steps=3, num_frames_skip=1, transform_state=False),
        make=lambda **_kwargs: _DummyEnv(),
    )
    policy = _Policy()
    traj = collect_trajectory(env_conf, policy, noise_seed=0)
    assert isinstance(policy.last_obs, np.ndarray)
    assert policy.last_obs.shape == (84, 84, 3)
    assert traj.rreturn == 1.0
