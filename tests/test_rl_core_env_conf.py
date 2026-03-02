from types import SimpleNamespace

from rl.core.env_conf import build_seeded_env_conf, build_seeded_env_conf_from_run, resolve_run_seeds


def test_resolve_run_seeds_prefers_explicit_values():
    out = resolve_run_seeds(seed=7, problem_seed=11, noise_seed_0=22)
    assert out.problem_seed == 11
    assert out.noise_seed_0 == 22


def test_build_seeded_env_conf_forwards_parameters():
    captured = {}

    def _fake_get_env_conf(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(gym_conf=SimpleNamespace(), ensure_spaces=lambda: None)

    out = build_seeded_env_conf(
        env_tag="pend",
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


def test_build_seeded_env_conf_from_run_uses_seed_resolution():
    out = build_seeded_env_conf_from_run(
        env_tag="pend",
        seed=3,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
        get_env_conf_fn=lambda *_args, **_kwargs: SimpleNamespace(gym_conf=SimpleNamespace(), ensure_spaces=lambda: None),
    )
    assert isinstance(out.problem_seed, int)
    assert isinstance(out.noise_seed_0, int)
    assert out.env_conf is not None
