import pytest

from rl.algos.runner_helpers import parse_runtime_args, parse_seeds, seeded_exp_dir, split_config_and_args


def test_parse_seeds_range():
    assert parse_seeds("1,3-5") == [1, 3, 4, 5]


def test_seeded_exp_dir_suffix():
    assert seeded_exp_dir("_tmp/ppo", 7).endswith("seed_7")


def test_parse_runtime_args_extracts_seeds_and_workers():
    runtime = parse_runtime_args(["--seeds", "1,3-4", "--workers", "2", "--set", "x=1"])
    assert runtime.seeds_raw == "1,3-4"
    assert runtime.workers == 2
    assert runtime.workers_cli_set is True
    assert runtime.cleaned == ["--set", "x=1"]


def test_split_config_and_args_accepts_local_prefix():
    config, rest = split_config_and_args(["local", "--config", "configs/rl/atari/ppo_pong_cpu.toml", "--set", "rl.ppo.seed=2"])
    assert config == "configs/rl/atari/ppo_pong_cpu.toml"
    assert rest == ["--set", "rl.ppo.seed=2"]


def test_split_config_and_args_still_accepts_legacy_form():
    config, rest = split_config_and_args(["--config", "configs/rl/atari/ppo_pong_cpu.toml"])
    assert config == "configs/rl/atari/ppo_pong_cpu.toml"
    assert rest == []


def test_split_config_and_args_usage_mentions_local():
    with pytest.raises(SystemExit, match=r"Usage: runner.py \[local\] --config"):
        split_config_and_args([])
