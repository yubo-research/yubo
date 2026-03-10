import pytest

from rl import builtins, runner
from rl.registry import get_algo


def test_extract_algo_cfg_with_backend_uses_algo_table():
    builtins.register_all()
    cfg = {
        "rl": {
            "algo": "ppo",
            "backend": "pufferlib",
            "ppo": {"exp_dir": "_tmp/test"},
        }
    }
    algo_name, backend, algo_cfg = runner._extract_algo_cfg(cfg)
    assert algo_name == "ppo"
    assert backend == "pufferlib"
    assert algo_cfg["exp_dir"] == "_tmp/test"
    assert get_algo(algo_name, backend=backend).name == "ppo"


def test_extract_algo_cfg_with_backend_ignores_legacy_impl_table():
    builtins.register_all()
    cfg = {
        "rl": {
            "algo": "ppo",
            "backend": "pufferlib",
            "ppo_puffer": {"exp_dir": "_tmp/test_fallback"},
        }
    }
    _, _, algo_cfg = runner._extract_algo_cfg(cfg)
    assert algo_cfg == {}


def test_extract_algo_cfg_unknown_backend_raises():
    builtins.register_all()
    cfg = {
        "rl": {
            "algo": "ppo",
            "backend": "unknown_backend",
            "ppo": {},
        }
    }
    with pytest.raises(ValueError, match="Unknown backend"):
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
