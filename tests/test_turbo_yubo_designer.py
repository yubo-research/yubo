def test_turbo_yubo_designer_basic():
    from acq.turbo_yubo.turbo_yubo_config import TurboYUBOConfig
    from optimizer.turbo_yubo_designer import TurboYUBODesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
    policy = default_policy(env_conf)

    designer = TurboYUBODesigner(policy, config=TurboYUBOConfig())
    assert len(designer([], num_arms=1)) == 1
    assert len(designer([], num_arms=5)) == 5
