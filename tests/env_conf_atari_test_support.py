"""Atari/dm fakes for env_conf tests (kiss: type() instead of class statements)."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


def fake_bindings_resolve_atari():
    return type(
        "_FakeBindings",
        (),
        {"resolve_atari_from_tag": staticmethod(lambda _tag: ("ALE/Pong-v5", lambda _env_conf: object()))},
    )()


def fake_bindings_pong_stack():
    captured: dict = {}

    def _fake_make_atari_env(env_name, *, render_mode=None, max_episode_steps=0, preprocess=None):
        captured["env_name"] = env_name
        captured["render_mode"] = render_mode
        captured["max_episode_steps"] = max_episode_steps
        captured["preprocess"] = preprocess
        return _fake_env_cls()()

    FakeEnv = _fake_env_cls()
    FakeAtariPreprocessOptions = _fake_preprocess_options_cls()
    return (
        type(
            "_FakeBindings",
            (),
            {
                "resolve_dm_control_from_tag": staticmethod(lambda tag, use_pixels: (str(tag), object())),
                "resolve_atari_from_tag": staticmethod(lambda tag: (str(tag), lambda _env_conf: object())),
                "make_atari_preprocess_options": staticmethod(lambda **kwargs: FakeAtariPreprocessOptions(**kwargs)),
                "make_dm_control_env": staticmethod(lambda *args, **kwargs: FakeEnv()),
                "make_atari_env": staticmethod(_fake_make_atari_env),
            },
        )(),
        captured,
    )


def _fake_env_cls():
    return type(
        "_FakeEnv",
        (),
        {
            "observation_space": spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),
            "action_space": spaces.Discrete(6),
            "close": lambda self: None,
        },
    )


def _fake_preprocess_options_cls():
    def _init(
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

    return type("_FakeAtariPreprocessOptions", (), {"__init__": _init})
