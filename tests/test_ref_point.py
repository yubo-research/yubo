import numpy as np


def test_sobol_ref_point_reproducible():
    from analysis.ref_point import SobolRefPoint
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-5d", problem_seed=0, noise_seed_0=17)
    policy = default_policy(env_conf)

    rp = SobolRefPoint(num_cal=64, seed=123, noise_seed_0=999, std_margin_scale=0.1)
    r_1 = rp.compute(env_conf, policy=policy)
    r_2 = rp.compute(env_conf, policy=policy)

    assert r_1.shape == (1,)
    assert np.all(np.isfinite(r_1))
    assert np.allclose(r_1, r_2)


def test_sobol_ref_point_changes_with_seed():
    from analysis.ref_point import SobolRefPoint
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-5d", problem_seed=0, noise_seed_0=17)
    policy = default_policy(env_conf)

    r_1 = SobolRefPoint(num_cal=64, seed=1, noise_seed_0=999).compute(
        env_conf, policy=policy
    )
    r_2 = SobolRefPoint(num_cal=64, seed=2, noise_seed_0=999).compute(
        env_conf, policy=policy
    )
    assert not np.allclose(r_1, r_2)
