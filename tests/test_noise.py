def test_noise():
    import numpy as np

    from problems.env_conf import get_env_conf

    noise = 0.1
    env_conf = get_env_conf("f:ackley-2d", noise=noise, seed=1)
    env = env_conf.make()

    assert env._real_noise_level > 0.2

    r = []
    for _ in range(10):
        r.append(env.step(np.array([0.63]))[1])
    assert np.std(r) > 0.02
