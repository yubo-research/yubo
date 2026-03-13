import pytest

import rl.runner as runner
from rl.runner_helpers import parse_runtime_args, seeded_exp_dir, split_config_and_args


def test_seeded_exp_dir_suffix():
    assert seeded_exp_dir("_tmp/ppo", 7).endswith("seed_7")


def test_parse_runtime_args_rejects_seeds_flag():
    with pytest.raises(SystemExit, match="--seeds is removed"):
        _ = parse_runtime_args(["--seeds", "1,3-4", "--workers", "2", "--set", "x=1"])


def test_split_config_and_args_accepts_local_prefix():
    config, rest = split_config_and_args(
        [
            "local",
            "--config",
            "configs/rl/atari/ppo_pong_cpu.toml",
            "--set",
            "rl.ppo.seed=2",
        ]
    )
    assert config == "configs/rl/atari/ppo_pong_cpu.toml"
    assert rest == ["--set", "rl.ppo.seed=2"]


def test_split_config_and_args_still_accepts_legacy_form():
    config, rest = split_config_and_args(["--config", "configs/rl/atari/ppo_pong_cpu.toml"])
    assert config == "configs/rl/atari/ppo_pong_cpu.toml"
    assert rest == []


def test_split_config_and_args_usage_mentions_local():
    with pytest.raises(SystemExit, match=r"Usage: runner.py \[local\] --config"):
        split_config_and_args([])


def test_apply_runner_overrides_routes_flat_keys_into_experiment():
    cfg = {
        "experiment": {
            "opt_name": "ppo",
            "env_tag": "cheetah",
            "num_reps": 10,
            "eval_interval": 10,
        }
    }
    out = runner._apply_runner_overrides(cfg, {"num_reps": 1, "eval_interval": 2, "experiment.local_workers": 3})
    assert out["experiment"]["num_reps"] == 1
    assert out["experiment"]["eval_interval"] == 2
    assert out["experiment"]["local_workers"] == 3
    assert "num_reps" not in out
