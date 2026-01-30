import numpy as np


def test_reactor_policy_factory_init():
    from problems.reactor_policy import ReactorPolicyFactory

    factory = ReactorPolicyFactory(
        num_modes=3,
        memory_dim=6,
        delta_hidden=8,
        return_metrics=False,
    )
    assert factory._num_modes == 3
    assert factory._memory_dim == 6


def test_reactor_policy_wants_vector_return():
    from types import SimpleNamespace

    from problems.reactor_policy import ReactorPolicy

    env_conf = SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(24,))),
        action_space=SimpleNamespace(shape=(4,)),
        env_name="BipedalWalker-v3",
    )

    p = ReactorPolicy(env_conf, num_modes=3, memory_dim=6, return_metrics=False)
    assert p.wants_vector_return() is False

    p2 = ReactorPolicy(env_conf, num_modes=3, memory_dim=6, return_metrics=True)
    assert p2.wants_vector_return() is True


def test_reactor_policy_roundtrip_and_bounds():
    from types import SimpleNamespace

    from problems.reactor_policy import ReactorPolicy

    env_conf = SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(24,))),
        action_space=SimpleNamespace(shape=(4,)),
        env_name="BipedalWalker-v3",
    )

    p = ReactorPolicy(env_conf, num_modes=3, memory_dim=6)
    n = p.num_params()

    x = np.random.uniform(-1.0, 1.0, size=(n,)).astype(np.float64)
    p.set_params(x)
    y = p.get_params()
    assert np.allclose(x, y, atol=0.0)

    p.set_params(np.zeros_like(x))
    o = np.zeros((24,), dtype=np.float64)
    a0 = p(o)
    assert a0.shape == (4,)
    assert float(a0.min()) >= -1.0 and float(a0.max()) <= 1.0
    m0 = p.metrics()
    assert m0.shape == (8,)
    assert float(m0.min()) >= -1.0
    assert float(m0.max()) <= 0.0
    _ = p(o)
    p.reset_state()
    a1 = p(o)
    assert np.allclose(a0, a1, atol=1e-12)
    m1 = p.metrics()
    assert m1.shape == (8,)
    assert float(m1.min()) >= -1.0
    assert float(m1.max()) <= 0.0
