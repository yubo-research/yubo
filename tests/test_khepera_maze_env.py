import matplotlib
import numpy as np

matplotlib.use("Agg")

from problems.khepera_maze_env import KheperaMazeEnv, khepera_maze_conf


def test_khepera_maze_env_basic():
    env = KheperaMazeEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    for t in range(10):
        action = np.random.uniform(-env.max_speed, env.max_speed, size=2)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        if done:
            break


def test_khepera_maze_env_reset_seed():
    env = KheperaMazeEnv()
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)


def test_khepera_maze_env_spaces():
    env = KheperaMazeEnv()
    assert hasattr(env, "observation_space")
    obs_space = env.observation_space
    assert hasattr(obs_space, "low")
    assert hasattr(obs_space, "high")
    assert obs_space.low.shape == (5,)
    assert obs_space.high.shape == (5,)


def test_khepera_maze_conf_config():
    conf = khepera_maze_conf()
    assert hasattr(conf, "env_name")
    assert hasattr(conf, "gym_conf")
    assert hasattr(conf, "policy_class")
    assert hasattr(conf, "action_space")
    assert hasattr(conf, "make")
    env = conf.make()
    assert isinstance(env, KheperaMazeEnv)


def test_khepera_maze_conf_gym_conf():
    conf = khepera_maze_conf()
    assert hasattr(conf, "gym_conf")
    gym_conf = conf.gym_conf
    assert hasattr(gym_conf, "state_space")
    state_space = gym_conf.state_space
    assert hasattr(state_space, "shape")
    assert state_space.shape == (5,)
