import numpy as np


def test_turbo_lunar_policy_init():
    from types import SimpleNamespace

    from problems.turbo_lunar_policy import TurboLunarPolicy

    env_conf = SimpleNamespace(problem_seed=42)
    p = TurboLunarPolicy(env_conf)
    assert p.num_params() == 12
    assert p.problem_seed == 42


def test_turbo_lunar_policy_set_get_params():
    from types import SimpleNamespace

    from problems.turbo_lunar_policy import TurboLunarPolicy

    env_conf = SimpleNamespace(problem_seed=42)
    p = TurboLunarPolicy(env_conf)

    x = np.zeros(12)
    p.set_params(x)
    assert np.allclose(p.get_params(), x)


def test_turbo_lunar_policy_clone():
    from types import SimpleNamespace

    from problems.turbo_lunar_policy import TurboLunarPolicy

    env_conf = SimpleNamespace(problem_seed=42)
    p = TurboLunarPolicy(env_conf)
    x = np.zeros(12)
    p.set_params(x)

    p2 = p.clone()
    assert np.allclose(p2.get_params(), x)


def test_turbo_lunar_policy_call():
    from types import SimpleNamespace

    from problems.turbo_lunar_policy import TurboLunarPolicy

    env_conf = SimpleNamespace(problem_seed=42)
    p = TurboLunarPolicy(env_conf)
    x = np.zeros(12)
    p.set_params(x)

    state = np.zeros(8)
    action = p(state)
    assert action in [0, 1, 2, 3]
