import numpy as np


def test_control_policy_params_roundtrip_and_reset_state():
    from types import SimpleNamespace

    from problems.control_policy import ControlPolicy

    env_conf = SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(24,))),
        action_space=SimpleNamespace(shape=(4,)),
    )

    p = ControlPolicy(env_conf, use_layer_norm=True)
    n = p.num_params()

    x = np.random.uniform(-1.0, 1.0, size=(n,)).astype(np.float64)
    p.set_params(x)
    y = p.get_params()
    assert y.shape == x.shape
    assert np.allclose(y, x, atol=1e-6)

    p.set_params(np.zeros_like(x))
    s = np.zeros((24,), dtype=np.float32)
    a0 = p(s)
    _ = p(s)
    p.reset_state()
    a1 = p(s)
    assert np.allclose(a0, a1, atol=1e-6)
