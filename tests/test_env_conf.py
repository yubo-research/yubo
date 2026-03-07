def test_gym_conf_init():
    from problems.env_conf import GymConf

    conf = GymConf()
    assert conf.max_steps == 1000
    assert conf.num_frames_skip == 30


def test_gym_conf_custom():
    from problems.env_conf import GymConf

    conf = GymConf(max_steps=500, num_frames_skip=10)
    assert conf.max_steps == 500
    assert conf.num_frames_skip == 10


def test_env_conf_init():
    from problems.env_conf import EnvConf

    # Use a real env_name that exists
    conf = EnvConf(env_name="f:ackley-3d")
    assert conf.env_name == "f:ackley-3d"
    assert conf.problem_seed is None
    assert conf.frozen_noise is True


def test_env_conf_custom():
    from problems.env_conf import EnvConf

    conf = EnvConf(
        env_name="f:ackley-3d",
        problem_seed=42,
        noise_level=0.1,
        noise_seed_0=0,
    )
    assert conf.problem_seed == 42
    assert conf.noise_level == 0.1


def test_get_env_conf_pure_function():
    from problems.env_conf import get_env_conf

    conf = get_env_conf("f:ackley-3d", problem_seed=0, noise_seed_0=0)
    assert conf.env_name == "f:ackley-3d"


def test_default_policy():
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:ackley-3d", problem_seed=0, noise_seed_0=0)
    policy = default_policy(env_conf)
    assert policy is not None


def test_resolve_rl_model_defaults_cheetah_sac():
    from problems.env_conf import resolve_rl_model_defaults

    cfg = resolve_rl_model_defaults("cheetah", algo="sac")
    assert cfg["backbone_name"] == "mlp"
    assert cfg["backbone_hidden_sizes"] == (256, 256)
    assert cfg["backbone_activation"] == "relu"
    assert cfg["critic_head_hidden_sizes"] == ()
    # Override is merged on top of inferred base, so base defaults stay available.
    assert cfg["actor_head_hidden_sizes"] == ()


def test_resolve_rl_model_defaults_cheetah_ppo_uses_critic_head():
    from problems.env_conf import resolve_rl_model_defaults

    cfg = resolve_rl_model_defaults("cheetah", algo="ppo")
    assert cfg["backbone_name"] == "mlp"
    assert cfg["backbone_hidden_sizes"] == (64, 64)
    assert cfg["critic_head_hidden_sizes"] == ()
    assert cfg["share_backbone"] is True


def test_resolve_rl_model_defaults_quadruped_run_infers_from_policy_class():
    from problems.env_conf import resolve_rl_model_defaults

    ppo_cfg = resolve_rl_model_defaults("quadruped-run-64x64", algo="ppo")
    assert ppo_cfg["backbone_name"] == "mlp"
    assert ppo_cfg["backbone_hidden_sizes"] == (64, 64)
    assert ppo_cfg["backbone_activation"] == "silu"
    assert ppo_cfg["backbone_layer_norm"] is True
    assert ppo_cfg["critic_head_hidden_sizes"] == ()

    sac_cfg = resolve_rl_model_defaults("quadruped-run-64x64", algo="sac")
    assert sac_cfg["backbone_name"] == "mlp"
    assert sac_cfg["backbone_hidden_sizes"] == (64, 64)
    assert sac_cfg["backbone_activation"] == "silu"
    assert sac_cfg["backbone_layer_norm"] is True
    assert sac_cfg["critic_head_hidden_sizes"] == ()
