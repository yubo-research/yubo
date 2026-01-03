import numpy as np


def test_policy_params_roundtrip_bw_policy():
    from types import SimpleNamespace

    from problems.bipedal_walker_policy import BipedalWalkerPolicy

    env_conf = SimpleNamespace(
        env_name="BipedalWalker-v3",
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(24,))),
        action_space=SimpleNamespace(shape=(4,)),
    )
    p = BipedalWalkerPolicy(env_conf)

    np.random.seed(0)
    x = np.random.uniform(-1.0, 1.0, size=(p.num_params(),))
    assert x.min() >= -1.0 and x.max() <= 1.0

    p.set_params(x)
    y = p.get_params()
    assert y.min() >= -1.0 and y.max() <= 1.0
    np.testing.assert_allclose(y, x, atol=0.0, rtol=0.0)


def test_bw_policy_internal_mapping_uses_center_scale_but_get_params_returns_raw():
    from types import SimpleNamespace

    from problems.bipedal_walker_policy import BipedalWalkerPolicy

    def map_pm1(v, lo, hi):
        return lo + (v + 1.0) * 0.5 * (hi - lo)

    env_conf = SimpleNamespace(
        env_name="BipedalWalker-v3",
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(24,))),
        action_space=SimpleNamespace(shape=(4,)),
    )

    p = BipedalWalkerPolicy(env_conf)

    x = np.linspace(-1.0, 1.0, num=p.num_params(), dtype=np.float64)
    p.set_params(x)
    np.testing.assert_allclose(p.get_params(), x, atol=0.0, rtol=0.0)

    x_eff = np.clip(p._x_center + p._x_scale * x, -1.0, 1.0)
    exp_speed = map_pm1(float(x_eff[0]), 0.15, 0.50)
    exp_hip_swing = map_pm1(float(x_eff[1]), 0.6, 1.5)
    exp_act_scale = map_pm1(float(x_eff[13]), 0.25, 1.0)
    exp_swap_timeout = int(round(map_pm1(float(x_eff[14]), 5.0, 50.0)))

    assert abs(float(p._speed) - exp_speed) < 1e-12
    assert abs(float(p._hip_swing) - exp_hip_swing) < 1e-12
    assert abs(float(p._act_scale) - exp_act_scale) < 1e-12
    assert int(p._swap_timeout) == exp_swap_timeout


def test_policy_params_roundtrip_linear_policy():
    from types import SimpleNamespace

    from problems.linear_policy import LinearPolicy

    env_conf = SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(6,))),
        action_space=SimpleNamespace(shape=(3,)),
    )
    p = LinearPolicy(env_conf)

    np.random.seed(1)
    x = np.random.uniform(-1.0, 1.0, size=(p.num_params(),))
    assert x.min() >= -1.0 and x.max() <= 1.0

    p.set_params(x)
    y = p.get_params()
    assert y.min() >= -1.0 and y.max() <= 1.0
    np.testing.assert_allclose(y, x, atol=0.0, rtol=0.0)


def test_policy_params_roundtrip_mlp_policy():
    from types import SimpleNamespace

    from problems.mlp_policy import MLPPolicy

    env_conf = SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))),
        action_space=SimpleNamespace(shape=(2,)),
    )
    p = MLPPolicy(env_conf, hidden_sizes=(8,))

    np.random.seed(2)
    x = np.random.uniform(-1.0, 1.0, size=(p.num_params(),))
    assert x.min() >= -1.0 and x.max() <= 1.0

    p.set_params(x)
    y = p.get_params()
    assert y.min() >= -1.0 and y.max() <= 1.0
    np.testing.assert_allclose(y, x, atol=1e-6, rtol=0.0)
