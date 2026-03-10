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
    assert cfg["backbone_hidden_sizes"] == (256, 256)
    assert cfg["backbone_activation"] == "relu"
    assert cfg["head_activation"] == "relu"
    assert "actor_head_hidden_sizes" not in cfg


def test_resolve_rl_model_defaults_cheetah_ppo_uses_explicit_model():
    from problems.env_conf import resolve_rl_model_defaults

    cfg = resolve_rl_model_defaults("cheetah", algo="ppo")
    assert cfg["backbone_hidden_sizes"] == (64, 64)
    assert cfg["backbone_layer_norm"] is True
    assert cfg["share_backbone"] is True
    assert cfg["log_std_init"] == -0.5


def test_resolve_rl_model_defaults_quadruped_run_uses_explicit_model():
    from problems.env_conf import resolve_rl_model_defaults

    ppo_cfg = resolve_rl_model_defaults("dm_control/quadruped-run-v0", algo="ppo")
    assert ppo_cfg["backbone_hidden_sizes"] == (64, 64)
    assert ppo_cfg["backbone_layer_norm"] is True
    assert ppo_cfg["share_backbone"] is True

    sac_cfg = resolve_rl_model_defaults("dm_control/quadruped-run-v0", algo="sac")
    assert sac_cfg["backbone_hidden_sizes"] == (256, 256)
    assert sac_cfg["backbone_activation"] == "relu"
    assert sac_cfg["backbone_layer_norm"] is True
    assert sac_cfg["head_activation"] == "relu"


def test_get_env_conf_applies_atari_preprocess_overrides():
    import problems.env_conf as env_conf_module

    class _FakeBindings:
        resolve_atari_from_tag = staticmethod(lambda _tag: ("ALE/Pong-v5", lambda _env_conf: object()))

    old_bindings = env_conf_module._ATARI_DM_BINDINGS
    env_conf_module._ATARI_DM_BINDINGS = _FakeBindings()
    try:
        conf = env_conf_module.get_env_conf(
            "atari:Pong:mlp16",
            problem_seed=0,
            atari_preprocess={
                "repeat_action_probability": 0.2,
                "use_minimal_action_set": False,
                "color_averaging": True,
                "grayscale_newaxis": False,
            },
        )
    finally:
        env_conf_module._ATARI_DM_BINDINGS = old_bindings
    assert isinstance(conf.atari_preprocess, dict)
    assert conf.atari_preprocess["repeat_action_probability"] == 0.2
    assert conf.atari_preprocess["use_minimal_action_set"] is False
    assert conf.atari_preprocess["color_averaging"] is True
    assert conf.atari_preprocess["grayscale_newaxis"] is False


def test_env_conf_ale_make_uses_atari_preprocess_options():
    import numpy as np
    from gymnasium import spaces

    import problems.env_conf as env_conf_module

    captured = {}

    class _FakeEnv:
        observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        action_space = spaces.Discrete(6)

        def close(self):
            return

    def _fake_make_atari_env(env_name, *, render_mode=None, max_episode_steps=0, preprocess=None):
        captured["env_name"] = env_name
        captured["render_mode"] = render_mode
        captured["max_episode_steps"] = max_episode_steps
        captured["preprocess"] = preprocess
        return _FakeEnv()

    class _FakeAtariPreprocessOptions:
        def __init__(
            self,
            *,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=True,
            scale_obs=False,
            repeat_action_probability=0.0,
            use_minimal_action_set=True,
            color_averaging=False,
        ):
            self.terminal_on_life_loss = terminal_on_life_loss
            self.grayscale_obs = grayscale_obs
            self.grayscale_newaxis = grayscale_newaxis
            self.scale_obs = scale_obs
            self.repeat_action_probability = repeat_action_probability
            self.use_minimal_action_set = use_minimal_action_set
            self.color_averaging = color_averaging

    class _FakeBindings:
        resolve_dm_control_from_tag = staticmethod(lambda tag, use_pixels: (str(tag), object()))
        resolve_atari_from_tag = staticmethod(lambda tag: (str(tag), lambda _env_conf: object()))
        make_atari_preprocess_options = staticmethod(lambda **kwargs: _FakeAtariPreprocessOptions(**kwargs))
        make_dm_control_env = staticmethod(lambda *args, **kwargs: _FakeEnv())
        make_atari_env = staticmethod(_fake_make_atari_env)

    old_bindings = env_conf_module._ATARI_DM_BINDINGS
    env_conf_module._ATARI_DM_BINDINGS = _FakeBindings()
    try:
        conf = env_conf_module.EnvConf(
            "ALE/Pong-v5",
            max_steps=321,
            atari_preprocess={
                "terminal_on_life_loss": True,
                "grayscale_newaxis": False,
                "scale_obs": True,
                "repeat_action_probability": 0.15,
                "use_minimal_action_set": False,
                "color_averaging": True,
            },
        )
        env = conf.make(render_mode="rgb_array")
        env.close()
    finally:
        env_conf_module._ATARI_DM_BINDINGS = old_bindings
    preprocess = captured["preprocess"]
    assert captured["env_name"] == "ALE/Pong-v5"
    assert captured["max_episode_steps"] == 321
    assert preprocess.terminal_on_life_loss is True
    assert preprocess.grayscale_newaxis is False
    assert preprocess.scale_obs is True
    assert preprocess.repeat_action_probability == 0.15
    assert preprocess.use_minimal_action_set is False
    assert preprocess.color_averaging is True


def test_get_env_conf_rejects_atari_preprocess_for_non_atari():
    import pytest

    from problems.env_conf import get_env_conf

    with pytest.raises(ValueError, match="only valid for Atari envs"):
        get_env_conf("pend", atari_preprocess={"repeat_action_probability": 0.2})
