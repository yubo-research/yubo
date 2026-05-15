from types import SimpleNamespace

from common.experiment_seeds import resolve_run_seeds
from rl.core.env_setup import build_env_setup


def test_resolve_run_seeds_prefers_explicit_values():
    out = resolve_run_seeds(seed=7, problem_seed=11, noise_seed_0=22)
    assert out.problem_seed == 11
    assert out.noise_seed_0 == 22


def test_build_env_setup_forwards_parameters(monkeypatch):
    captured = {}

    def _fake_get_env_conf(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(gym_conf=SimpleNamespace(), ensure_spaces=lambda: None)

    monkeypatch.setattr("rl.core.env_setup.maybe_register_atari_dm_backends", lambda *_args, **_kwargs: None)
    out = build_env_setup(
        env_tag="pend",
        seed=0,
        problem_seed=5,
        noise_seed_0=9,
        from_pixels=True,
        pixels_only=False,
        get_env_conf_fn=_fake_get_env_conf,
    )
    assert captured["args"] == ("pend",)
    assert captured["kwargs"]["problem_seed"] == 5
    assert captured["kwargs"]["noise_seed_0"] == 9
    assert captured["kwargs"]["from_pixels"] is True
    assert captured["kwargs"]["pixels_only"] is False
    assert out.problem_seed == 5
    assert out.noise_seed_0 == 9
    assert out.env_conf is not None


def test_resolve_run_seeds_from_seed_is_int():
    out = resolve_run_seeds(seed=3, problem_seed=None, noise_seed_0=None)
    assert isinstance(out.problem_seed, int)
    assert isinstance(out.noise_seed_0, int)
