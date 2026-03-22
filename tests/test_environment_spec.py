"""Tests for problems/environment_spec.py module."""

from __future__ import annotations

from types import SimpleNamespace


def test_gym_conf_dataclass():
    from problems.environment_spec import GymConf

    conf = GymConf()
    assert conf.max_steps == 1000
    assert conf.num_frames_skip == 30
    assert conf.state_space is None
    assert conf.transform_state is True

    conf2 = GymConf(max_steps=500, num_frames_skip=10, transform_state=False)
    assert conf2.max_steps == 500
    assert conf2.num_frames_skip == 10
    assert conf2.transform_state is False


def test_environment_spec_dataclass_basic():
    from problems.environment_spec import EnvironmentSpec

    spec = EnvironmentSpec(env_name="test-env")
    assert spec.env_name == "test-env"
    assert spec.gym_conf is None
    assert spec.kwargs == {}
    assert spec.from_pixels is False
    assert spec.pixels_only is True


def test_environment_spec_post_init_pure_function():
    from problems.environment_spec import _PURE_FUNCTION_MAX_STEPS, EnvironmentSpec

    spec_f = EnvironmentSpec(env_name="f:sphere-2d")
    assert spec_f.max_steps == _PURE_FUNCTION_MAX_STEPS

    spec_g = EnvironmentSpec(env_name="g:sphere-3d")
    assert spec_g.max_steps == _PURE_FUNCTION_MAX_STEPS


def test_environment_spec_post_init_default_max_steps():
    from problems.environment_spec import _DEFAULT_MAX_STEPS, EnvironmentSpec

    spec = EnvironmentSpec(env_name="custom-env")
    assert spec.max_steps == _DEFAULT_MAX_STEPS


def test_environment_spec_post_init_no_default_for_dm_atari():
    from problems.environment_spec import EnvironmentSpec

    spec_ale = EnvironmentSpec(env_name="ALE/Pong-v5")
    assert spec_ale.max_steps is None

    spec_dm = EnvironmentSpec(env_name="dm_control/walker-run-v0")
    assert spec_dm.max_steps is None


def test_environment_runtime_properties():
    from problems.environment_spec import EnvironmentRuntime, EnvironmentSpec, GymConf

    gym_conf = GymConf(max_steps=200)
    spec = EnvironmentSpec(env_name="Pendulum-v1", gym_conf=gym_conf)
    runtime = EnvironmentRuntime(spec=spec, problem_seed=42)

    assert runtime.env_name == "Pendulum-v1"
    assert runtime.gym_conf is gym_conf
    assert runtime.max_steps == 200
    assert runtime.kwargs == {}
    assert runtime.problem_seed == 42


def test_environment_runtime_max_steps_from_spec():
    from problems.environment_spec import EnvironmentRuntime, EnvironmentSpec

    spec = EnvironmentSpec(env_name="f:sphere-2d", max_steps=1)
    runtime = EnvironmentRuntime(spec=spec)
    assert runtime.max_steps == 1


def test_environment_runtime_kwargs():
    from problems.environment_spec import EnvironmentRuntime, EnvironmentSpec

    spec = EnvironmentSpec(env_name="test", kwargs={"continuous": True})
    runtime = EnvironmentRuntime(spec=spec)
    assert runtime.kwargs == {"continuous": True}


def test_get_environment_spec_gym():
    from problems.environment_spec import get_environment_spec

    spec = get_environment_spec("cheetah")
    assert spec.env_name == "HalfCheetah-v5"
    assert spec.gym_conf is not None


def test_get_environment_spec_pure_function():
    from problems.environment_spec import get_environment_spec

    spec = get_environment_spec("f:sphere-2d")
    assert spec.env_name == "f:sphere-2d"
    assert spec.max_steps == 1


def test_get_environment_spec_unknown_returns_direct():
    from problems.environment_spec import get_environment_spec

    spec = get_environment_spec("custom-unknown-env")
    assert spec.env_name == "custom-unknown-env"


def test_materialize_env():
    from problems.environment_spec import EnvironmentSpec, materialize_env

    spec = EnvironmentSpec(env_name="f:sphere-2d")
    runtime = materialize_env(
        spec,
        problem_seed=123,
        noise_seed_0=456,
        noise_level=0.1,
        frozen_noise=True,
    )
    assert runtime.problem_seed == 123
    assert runtime.noise_seed_0 == 456
    assert runtime.noise_level == 0.1
    assert runtime.frozen_noise is True
    assert runtime.spec is spec


def test_materialize_env_make_pure_function():
    from problems.environment_spec import EnvironmentSpec, materialize_env

    spec = EnvironmentSpec(env_name="f:sphere-2d")
    runtime = materialize_env(spec, problem_seed=0)
    env = runtime.make()
    assert env is not None
    obs, _ = env.reset(seed=0)
    assert obs == 0
    env.close()


def test_needs_atari_dm_bindings_false():
    from problems.environment_spec import needs_atari_dm_bindings

    assert needs_atari_dm_bindings("f:sphere-2d") is False
    assert needs_atari_dm_bindings("g:ackley-3d") is False
    assert needs_atari_dm_bindings("custom-env") is False


def test_needs_atari_dm_bindings_true_prefixes():
    from problems.environment_spec import needs_atari_dm_bindings

    assert needs_atari_dm_bindings("dm:walker-walk") is True
    assert needs_atari_dm_bindings("dm_control/walker-run-v0") is True
    assert needs_atari_dm_bindings("atari:Pong") is True
    assert needs_atari_dm_bindings("ALE/Pong-v5") is True


def test_needs_atari_dm_bindings_known_tags():
    from problems.environment_spec import needs_atari_dm_bindings

    assert needs_atari_dm_bindings("atari-pong") is True
    assert needs_atari_dm_bindings("dm_control/quadruped-run-v0") is True


def test_needs_atari_dm_bindings_gym_tag_dm():
    from problems.environment_spec import needs_atari_dm_bindings

    assert needs_atari_dm_bindings("quadruped-run-64x64") is True


def test_parse_tag_options_no_options():
    from problems.environment_spec import parse_tag_options

    tag, frozen, pix = parse_tag_options("cheetah", None)
    assert tag == "cheetah"
    assert frozen is False
    assert pix is None


def test_parse_tag_options_fn_suffix():
    from problems.environment_spec import parse_tag_options

    tag, frozen, pix = parse_tag_options("cheetah:fn", None)
    assert tag == "cheetah"
    assert frozen is True
    assert pix is None


def test_parse_tag_options_pixels_suffix():
    from problems.environment_spec import parse_tag_options

    tag, frozen, pix = parse_tag_options("dm:walker-walk:pixels", None)
    assert tag == "dm:walker-walk"
    assert frozen is False
    assert pix is True


def test_parse_tag_options_both_suffixes():
    from problems.environment_spec import parse_tag_options

    tag, frozen, pix = parse_tag_options("dm:walker-walk:pixels:fn", None)
    assert tag == "dm:walker-walk"
    assert frozen is True
    assert pix is True


def test_parse_tag_options_from_pixels_override():
    from problems.environment_spec import parse_tag_options

    tag, frozen, pix = parse_tag_options("test:pixels", False)
    assert pix is False


def test_register_atari_dm_bindings_loader():
    from problems.environment_spec import register_atari_dm_bindings_loader

    call_count = {"n": 0}

    def loader():
        call_count["n"] += 1
        return SimpleNamespace()

    register_atari_dm_bindings_loader(loader)
    assert call_count["n"] == 0


def test_environment_runtime_make_with_noise():
    from problems.environment_spec import EnvironmentSpec, materialize_env

    spec = EnvironmentSpec(env_name="f:sphere-2d")
    runtime = materialize_env(spec, problem_seed=0, noise_level=0.01)
    env = runtime.make()
    assert env is not None
    env.close()


def test_gym_spec_helper():
    from problems.environment_spec import _gym_spec

    spec = _gym_spec("TestEnv-v0")
    assert spec.env_name == "TestEnv-v0"
    assert spec.gym_conf is not None
    assert spec.gym_conf.max_steps == 1000


def test_environment_spec_copy_from_get():
    from problems.environment_spec import get_environment_spec

    spec1 = get_environment_spec("cheetah")
    spec2 = get_environment_spec("cheetah")
    assert spec1 is not spec2
    spec1.max_steps = 999
    assert spec2.max_steps != 999
