from types import SimpleNamespace

import pytest
from rl_puffer_vector_env_stubs import _FakeAtari, _FakeVector

from rl.pufferlib import vector_env


def test_vector_backend_from_name_invalid():
    with pytest.raises(ValueError, match="vector_backend"):
        _ = vector_env._vector_backend_from_name(_FakeVector, "bad")


def test_make_vector_env_atari_path_uses_framestack_and_backend_kwargs():
    fake_vector = _FakeVector()
    fake_atari = _FakeAtari()
    cfg = SimpleNamespace(
        env_tag="atari:Pong",
        framestack=4,
        vector_backend="multiprocessing",
        vector_num_workers=2,
        vector_batch_size=3,
        vector_overwork=True,
        num_envs=8,
        seed=9,
    )

    out = vector_env.make_vector_env(
        cfg,
        import_pufferlib_modules_fn=lambda: (object(), fake_vector, fake_atari),
        is_atari_env_tag_fn=lambda tag: tag.startswith("atari:"),
        to_puffer_game_name_fn=lambda _tag: "pong",
        resolve_gym_env_name_fn=lambda _tag: ("CartPole-v1", {}),
    )
    assert out == "vec-env"
    assert fake_atari.games == ["pong"]
    assert len(fake_vector.calls) == 1
    call = fake_vector.calls[0]
    assert call["env_kwargs"] == {"framestack": 4}
    assert call["backend"] is fake_vector.Multiprocessing
    assert call["num_envs"] == 8
    assert call["seed"] == 9
    assert call["kwargs"]["num_workers"] == 2
    assert call["kwargs"]["batch_size"] == 3
    assert call["kwargs"]["overwork"] is True


def test_make_vector_env_gym_path_uses_empty_env_kwargs():
    fake_vector = _FakeVector()
    fake_atari = _FakeAtari()
    cfg = SimpleNamespace(
        env_tag="Pendulum-v1",
        env_conf=SimpleNamespace(env_tag="Pendulum-v1"),
        framestack=4,
        vector_backend="serial",
        vector_num_workers=None,
        vector_batch_size=None,
        vector_overwork=False,
        num_envs=2,
        seed=1,
    )

    out = vector_env.make_vector_env(
        cfg,
        import_pufferlib_modules_fn=lambda: (object(), fake_vector, fake_atari),
        is_atari_env_tag_fn=lambda _tag: False,
        to_puffer_game_name_fn=lambda _tag: "unused",
        resolve_gym_env_name_fn=lambda _tag: ("Pendulum-v1", {"g": 9.81}),
    )
    assert out == "vec-env"
    assert fake_atari.games == []
    assert len(fake_vector.calls) == 1
    call = fake_vector.calls[0]
    assert call["env_kwargs"] == {}
    assert call["backend"] is fake_vector.Serial
    assert call["kwargs"] == {}
    assert callable(call["env_creator"])


def test_make_gymnasium_env_accepts_environment_runtime_make(monkeypatch):
    calls = {}

    class FakeRuntime:
        def make(self, **kwargs):
            calls.update(kwargs)
            return "gym-env"

    fake_puffer = SimpleNamespace(EpisodeStats=lambda env: ("stats", env))
    fake_emulation = SimpleNamespace(GymnasiumPufferEnv=lambda **kwargs: ("puffer", kwargs))
    monkeypatch.setitem(__import__("sys").modules, "pufferlib", fake_puffer)
    monkeypatch.setitem(__import__("sys").modules, "pufferlib.emulation", fake_emulation)
    fake_puffer.emulation = fake_emulation

    out = vector_env._make_gymnasium_env(env_conf=FakeRuntime(), render_mode="rgb_array", buf="buf", seed=3)

    assert calls == {"render_mode": "rgb_array"}
    assert out == ("puffer", {"env": ("stats", "gym-env"), "buf": "buf", "seed": 3})


def test_make_vector_env_dm_control_path_uses_dm_creator(monkeypatch):
    fake_vector = _FakeVector()
    fake_atari = _FakeAtari()
    cfg = SimpleNamespace(
        env_tag="dm_control/quadruped-run-v0",
        env_conf=SimpleNamespace(env_tag="dm_control/quadruped-run-v0"),
        from_pixels=False,
        pixels_only=True,
        framestack=1,
        vector_backend="serial",
        vector_num_workers=None,
        vector_batch_size=None,
        vector_overwork=False,
        num_envs=2,
        seed=7,
    )

    captured = {}

    def _fake_dm_creator(**kwargs):
        captured.update(kwargs)
        return "dm-env"

    monkeypatch.setattr(vector_env, "_make_dm_control_env", _fake_dm_creator)

    out = vector_env.make_vector_env(
        cfg,
        import_pufferlib_modules_fn=lambda: (object(), fake_vector, fake_atari),
        is_atari_env_tag_fn=lambda _tag: False,
        to_puffer_game_name_fn=lambda _tag: "unused",
        resolve_gym_env_name_fn=lambda _tag: ("dm_control/quadruped-run-v0", {}),
    )
    assert out == "vec-env"
    call = fake_vector.calls[0]
    assert call["env_kwargs"] == {}
    assert callable(call["env_creator"])
    assert call["env_creator"]() == "dm-env"
    assert captured["domain"] == "quadruped"
    assert captured["task"] == "run"
    assert captured["from_pixels"] is False
    assert captured["pixels_only"] is True
