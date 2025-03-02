def test_lhd_designer():
    from optimizer.lhd_designer import LHDDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
    policy = default_policy(env_conf)

    lhdd = LHDDesigner(policy)
    assert len(lhdd(None, num_arms=1)) == 1
    assert len(lhdd(None, num_arms=1000)) == 1000
