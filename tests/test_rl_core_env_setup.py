from types import SimpleNamespace

import numpy as np
import pytest

from rl.core.env_setup import build_continuous_gym_env_setup


def test_build_continuous_gym_env_setup_success_normalizes_bounds():
    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(transform_state=False),
        action_space=SimpleNamespace(
            shape=(3,),
            low=np.asarray([-np.inf], dtype=np.float32),
            high=np.asarray([np.inf], dtype=np.float32),
        ),
        ensure_spaces=lambda: None,
    )
    out = build_continuous_gym_env_setup(
        env_tag="pend",
        seed=7,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
        get_env_conf_fn=lambda *_args, **_kwargs: env_conf,
        obs_scale_from_env_fn=lambda _env: (None, None),
    )
    assert out.env_conf is env_conf
    assert out.act_dim == 3
    assert np.allclose(out.action_low, np.asarray([-1.0, -1.0, -1.0], dtype=np.float32))
    assert np.allclose(out.action_high, np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
    assert out.obs_lb is None
    assert out.obs_width is None


def test_build_continuous_gym_env_setup_allows_non_gym_backend_contract():
    env_conf = SimpleNamespace(
        gym_conf=None,
        action_space=SimpleNamespace(shape=(1,), low=np.asarray([-1.0]), high=np.asarray([1.0])),
        ensure_spaces=lambda: None,
    )
    out = build_continuous_gym_env_setup(
        env_tag="bad",
        seed=0,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
        get_env_conf_fn=lambda *_args, **_kwargs: env_conf,
        obs_scale_from_env_fn=lambda _env: (None, None),
    )
    assert out.env_conf is env_conf


def test_build_continuous_gym_env_setup_requires_continuous_box_space():
    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(transform_state=False),
        action_space=SimpleNamespace(shape=(2,)),
        ensure_spaces=lambda: None,
    )
    with pytest.raises(ValueError, match="continuous Box action space"):
        _ = build_continuous_gym_env_setup(
            env_tag="bad",
            seed=0,
            problem_seed=None,
            noise_seed_0=None,
            from_pixels=False,
            pixels_only=True,
            get_env_conf_fn=lambda *_args, **_kwargs: env_conf,
            obs_scale_from_env_fn=lambda _env: (None, None),
        )
