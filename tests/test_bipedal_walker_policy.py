import numpy as np


def test_bipedal_walker_policy_outputs_bounded_actions():
    from types import SimpleNamespace

    from problems.bipedal_walker_policy import BipedalWalkerPolicy

    env_conf = SimpleNamespace(
        env_name="BipedalWalker-v3",
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(24,))),
        action_space=SimpleNamespace(shape=(4,)),
    )

    p = BipedalWalkerPolicy(env_conf)

    s = np.zeros(24, dtype=np.float64)
    a = p(s)
    assert a.shape == (4,)
    assert np.isfinite(a).all()
    assert a.min() >= -1.0 and a.max() <= 1.0

    s[2] = 0.3
    s[8] = 1.0
    s[13] = 0.0
    s[14:] = 1.0
    a = p(s)
    assert a.shape == (4,)
    assert np.isfinite(a).all()
    assert a.min() >= -1.0 and a.max() <= 1.0
