import pytest

from rl.algos import builtins, runner
from rl.algos.registry import get_algo


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
    assert get_algo(algo_name, backend=backend).name == "ppo_puffer"


def test_extract_algo_cfg_with_backend_falls_back_to_resolved_table():
    builtins.register_all()
    cfg = {
        "rl": {
            "algo": "ppo",
            "backend": "pufferlib",
            "ppo_puffer": {"exp_dir": "_tmp/test_fallback"},
        }
    }
    _, _, algo_cfg = runner._extract_algo_cfg(cfg)
    assert algo_cfg["exp_dir"] == "_tmp/test_fallback"


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
