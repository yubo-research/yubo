from types import SimpleNamespace

from rl.core.envs import conf_for_run, seeded_conf, seeds


def test_seeds_prefers_explicit_values():
    out = seeds(seed=7, problem_seed=11, noise_seed_0=22)
    assert out.problem_seed == 11
    assert out.noise_seed_0 == 22


def test_seeded_conf_forwards_parameters():
    captured = {}

    def _fake_get_env_conf(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(gym_conf=SimpleNamespace(), ensure_spaces=lambda: None)

    out = seeded_conf(
        env_tag="cheetah",
        problem_seed=5,
        noise_seed_0=9,
        obs_mode="mixed",
        get_env_conf_fn=_fake_get_env_conf,
    )
    assert captured["args"] == ("cheetah",)
    assert captured["kwargs"]["problem_seed"] == 5
    assert captured["kwargs"]["noise_seed_0"] == 9
    assert captured["kwargs"]["obs_mode"] == "mixed"
    assert out.problem_seed == 5
    assert out.noise_seed_0 == 9
    assert out.env_conf is not None


def test_conf_for_run_uses_seed_resolution():
    out = conf_for_run(
        env_tag="cheetah",
        seed=3,
        problem_seed=None,
        noise_seed_0=None,
        obs_mode="image",
        get_env_conf_fn=lambda *_args, **_kwargs: SimpleNamespace(gym_conf=SimpleNamespace(), ensure_spaces=lambda: None),
    )
    assert isinstance(out.problem_seed, int)
    assert isinstance(out.noise_seed_0, int)
    assert out.env_conf is not None
