from rl.algos.runner import _parse_runtime_args, _parse_seeds, _seeded_exp_dir


def test_parse_seeds_range():
    assert _parse_seeds("1,3-5") == [1, 3, 4, 5]


def test_seeded_exp_dir_suffix():
    assert _seeded_exp_dir("_tmp/ppo", 7).endswith("seed_7")


def test_parse_runtime_args_extracts_seeds_and_workers():
    runtime = _parse_runtime_args(["--seeds", "1,3-4", "--workers", "2", "--set", "x=1"])
    assert runtime.seeds_raw == "1,3-4"
    assert runtime.workers == 2
    assert runtime.workers_cli_set is True
    assert runtime.cleaned == ["--set", "x=1"]
