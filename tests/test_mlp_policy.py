import numpy as np


def _flat_params(policy):
    with np.errstate(all="ignore"):
        return np.concatenate(
            [p.data.detach().cpu().numpy().reshape(-1) for p in policy.parameters()]
        )


def test_mlp_policy_set_params_is_delta_from_init():
    from types import SimpleNamespace

    from problems.mlp_policy import MLPPolicy

    env_conf = SimpleNamespace(
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))),
        action_space=SimpleNamespace(shape=(2,)),
    )

    p = MLPPolicy(env_conf, hidden_sizes=(8,))
    init_flat = p._flat_params_init.copy()
    np.testing.assert_allclose(p.get_params(), 0.0, atol=0.0, rtol=0.0)

    delta_1 = np.random.uniform(-0.1, 0.1, size=init_flat.shape)
    p.set_params(delta_1)
    np.testing.assert_allclose(p.get_params(), delta_1, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(
        _flat_params(p), init_flat + delta_1 * p._const_scale, atol=1e-6, rtol=0.0
    )

    delta_2 = np.random.uniform(-0.1, 0.1, size=init_flat.shape)
    p.set_params(delta_2)
    np.testing.assert_allclose(p.get_params(), delta_2, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(
        _flat_params(p), init_flat + delta_2 * p._const_scale, atol=1e-6, rtol=0.0
    )
