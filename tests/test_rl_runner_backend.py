import pytest

from rl import builtins, runner


def test_extract_algo_cfg_uses_algo_table():
    builtins.register_all()
    cfg = {
        "rl": {
            "algo": "sac",
            "sac": {"exp_dir": "_tmp/test"},
        }
    }
    algo_name, algo_cfg = runner._extract_algo_cfg(cfg)
    assert algo_name == "sac"
    assert algo_cfg["exp_dir"] == "_tmp/test"


def test_extract_algo_cfg_rejects_backend():
    builtins.register_all()
    cfg = {
        "rl": {
            "algo": "sac",
            "backend": "torchrl",
            "sac": {"exp_dir": "_tmp/test"},
        }
    }
    with pytest.raises(ValueError, match=r"\[rl\]\.backend is no longer supported"):
        runner._extract_algo_cfg(cfg)


def test_extract_algo_cfg_rejects_any_backend():
    builtins.register_all()
    cfg = {
        "rl": {
            "algo": "ppo",
            "backend": "unknown_backend",
            "ppo": {},
        }
    }
    with pytest.raises(ValueError, match=r"\[rl\]\.backend is no longer supported"):
        runner._extract_algo_cfg(cfg)


def test_extract_run_cfg_supports_zero_based_num_reps():
    cfg = {"rl": {"run": {"num_reps": 3, "workers": 2}}}
    seeds, workers = runner._extract_run_cfg(cfg)
    assert seeds == [0, 1, 2]
    assert workers == 2


def test_extract_run_cfg_rejects_explicit_seeds():
    cfg = {"rl": {"run": {"seeds": [7, 8], "num_reps": 5}}}
    with pytest.raises(ValueError, match=r"\[rl.run\]\.seeds is removed"):
        _ = runner._extract_run_cfg(cfg)


def test_extract_run_cfg_rejects_non_positive_num_reps():
    cfg = {"rl": {"run": {"num_reps": 0}}}
    with pytest.raises(ValueError, match=r"\[rl.run\]\.num_reps must be >= 1"):
        runner._extract_run_cfg(cfg)


def test_extract_video_cfg_maps_run_video_to_algorithm_keys():
    cfg = {
        "rl": {
            "run": {
                "video": {
                    "enable": True,
                    "num_episodes": 4,
                    "prefix": "policy",
                }
            }
        }
    }
    assert runner._extract_video_cfg(cfg) == {
        "video_enable": True,
        "video_num_episodes": 4,
        "video_prefix": "policy",
    }


def test_extract_video_cfg_rejects_non_table():
    cfg = {"rl": {"run": {"video": True}}}
    with pytest.raises(ValueError, match=r"\[rl.run.video\] must be a table"):
        runner._extract_video_cfg(cfg)


def test_run_from_cfg_applies_run_video_to_algorithm_config(monkeypatch):
    captured = {}

    class _Config:
        @classmethod
        def from_dict(cls, raw):
            captured.update(raw)
            return raw

    monkeypatch.setattr(
        "rl.registry.get_algo",
        lambda _algo_name: type("_Algo", (), {"config_cls": _Config, "train_fn": staticmethod(lambda cfg: cfg)})(),
    )
    cfg = {
        "rl": {
            "algo": "sac",
            "sac": {"env_tag": "cheetah"},
            "run": {"video": {"enable": True, "num_video_episodes": 2}},
        }
    }
    runner._run_from_cfg(cfg)
    assert captured["video_enable"] is True
    assert captured["video_num_video_episodes"] == 2
