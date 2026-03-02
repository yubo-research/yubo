from types import SimpleNamespace

import pytest

from rl.pufferlib import vector_env


class _FakeVector:
    class Serial:
        pass

    class Multiprocessing:
        pass

    def __init__(self):
        self.calls = []

    def make(self, env_creator, *, env_kwargs, backend, num_envs, seed, **kwargs):
        self.calls.append(
            {
                "env_creator": env_creator,
                "env_kwargs": env_kwargs,
                "backend": backend,
                "num_envs": num_envs,
                "seed": seed,
                "kwargs": kwargs,
            }
        )
        return "vec-env"


class _FakeAtari:
    def __init__(self):
        self.games = []

    def env_creator(self, game_name):
        self.games.append(game_name)

        def _creator():
            return game_name

        return _creator


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
