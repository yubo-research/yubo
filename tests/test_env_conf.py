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
