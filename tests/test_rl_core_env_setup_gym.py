"""Tests for rl.core.env_setup (SAC gym setup)."""

from types import SimpleNamespace

import numpy as np
import pytest

from rl.core.env_setup import ContinuousGymEnvSetup, build_continuous_gym_env_setup


def test_continuous_gym_env_setup_dataclass():
    setup = ContinuousGymEnvSetup(
        env_conf=object(),
        problem_seed=1,
        noise_seed_0=2,
        act_dim=2,
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        obs_lb=None,
        obs_width=None,
    )
    assert setup.problem_seed == 1
    assert setup.act_dim == 2


def test_build_continuous_gym_env_setup_success():
    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(transform_state=False),
        action_space=SimpleNamespace(
            shape=(2,),
            low=np.asarray([-1.0, 0.0]),
            high=np.asarray([1.0, 2.0]),
        ),
        ensure_spaces=lambda: None,
    )
    out = build_continuous_gym_env_setup(
        env_tag="pend",
        seed=7,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=False,
        get_env_conf_fn=lambda *_args, **_kwargs: env_conf,
        obs_scale_from_env_fn=lambda _env: (None, None),
    )
    assert out.env_conf is env_conf
    assert out.act_dim == 2
    assert np.allclose(out.action_low, np.asarray([-1.0, 0.0], dtype=np.float32))
    assert np.allclose(out.action_high, np.asarray([1.0, 2.0], dtype=np.float32))
    assert out.obs_lb is None
    assert out.obs_width is None


def test_build_continuous_gym_env_setup_with_obs_scale():
    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(
                low=np.array([-1.0]),
                high=np.array([1.0]),
                shape=(1,),
            ),
        ),
        action_space=SimpleNamespace(
            shape=(1,),
            low=np.asarray([-1.0]),
            high=np.asarray([1.0]),
        ),
        ensure_spaces=lambda: None,
    )
    out = build_continuous_gym_env_setup(
        env_tag="pend",
        seed=0,
        problem_seed=0,
        noise_seed_0=0,
        from_pixels=False,
        pixels_only=False,
        get_env_conf_fn=lambda *_args, **_kwargs: env_conf,
        obs_scale_from_env_fn=lambda ec: (
            np.array(ec.gym_conf.state_space.low, dtype=np.float32),
            np.array(ec.gym_conf.state_space.high - ec.gym_conf.state_space.low, dtype=np.float32),
        ),
    )
    assert out.obs_lb is not None
    assert out.obs_width is not None
    assert np.allclose(out.obs_lb, np.array([-1.0]))
    assert np.allclose(out.obs_width, np.array([2.0]))


def test_build_continuous_gym_env_setup_requires_continuous_box():
    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(transform_state=False),
        action_space=SimpleNamespace(shape=(2,)),
        ensure_spaces=lambda: None,
    )
    with pytest.raises(ValueError, match="continuous Box action space"):
        build_continuous_gym_env_setup(
            env_tag="bad",
            seed=0,
            problem_seed=None,
            noise_seed_0=None,
            from_pixels=False,
            pixels_only=False,
            get_env_conf_fn=lambda *_args, **_kwargs: env_conf,
            obs_scale_from_env_fn=lambda _env: (None, None),
        )
