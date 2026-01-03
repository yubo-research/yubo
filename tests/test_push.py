import numpy as np


def test_push_function():
    from problems.push import Push

    push = Push()
    push.reset()
    assert push.num_dim == 14

    x = np.zeros(14)
    result = push(x)
    assert np.isfinite(result)


def test_push_env():
    from problems.env_conf import get_env_conf

    ec = get_env_conf("push", problem_seed=42)
    env = ec.make()
    env.reset(seed=0)
    action = np.zeros(14)
    state, reward, done, info = env.step(action)
    assert np.isfinite(reward)
    env.close()
