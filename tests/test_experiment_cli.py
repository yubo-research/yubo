import textwrap

import pytest

from experiments.experiment import load_experiment_config


def _write_toml(tmp_path, content: str):
    path = tmp_path / "exp.toml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_load_experiment_config_cli_only():
    cfg = load_experiment_config(
        [
            "--exp-dir=_tmp/exp",
            "--env-tag=f:sphere-2d",
            "--opt-name=random",
            "--num-arms=4",
            "--num-rounds=10",
            "--num-reps=3",
        ]
    )
    assert cfg.exp_dir == "_tmp/exp"
    assert cfg.env_tag == "f:sphere-2d"
    assert cfg.opt_name == "random"
    assert cfg.num_arms == 4
    assert cfg.num_rounds == 10
    assert cfg.num_reps == 3
    assert cfg.b_trace is True


def test_load_experiment_config_toml_experiment_table(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        env_tag = "f:ackley-5d"
        opt_name = "ucb"
        num_arms = 8
        num_rounds = 20
        num_reps = 5
        b_trace = false
        max_total_seconds = 30.5
        """,
    )
    cfg = load_experiment_config(["--config", str(toml_path)])
    assert cfg.exp_dir == "_tmp/exp"
    assert cfg.env_tag == "f:ackley-5d"
    assert cfg.opt_name == "ucb"
    assert cfg.num_arms == 8
    assert cfg.num_rounds == 20
    assert cfg.num_reps == 5
    assert cfg.b_trace is False
    assert cfg.max_total_seconds == 30.5


def test_load_experiment_config_toml_root_and_hyphen_keys(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        "exp-dir" = "_tmp/root"
        "env-tag" = "f:rastrigin-3d"
        "opt-name" = "random"
        "num-arms" = 2
        "num-rounds" = 4
        "num-reps" = 1
        """,
    )
    cfg = load_experiment_config([f"--config={toml_path}"])
    assert cfg.exp_dir == "_tmp/root"
    assert cfg.env_tag == "f:rastrigin-3d"
    assert cfg.opt_name == "random"
    assert cfg.num_arms == 2
    assert cfg.num_rounds == 4
    assert cfg.num_reps == 1


def test_load_experiment_config_cli_overrides_toml(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        env_tag = "f:sphere-2d"
        opt_name = "random"
        num_arms = 4
        num_rounds = 6
        num_reps = 1
        b_trace = false
        """,
    )
    cfg = load_experiment_config(
        [
            f"--config={toml_path}",
            "--num-reps=9",
            "--b-trace=true",
        ]
    )
    assert cfg.num_reps == 9
    assert cfg.b_trace is True


def test_load_experiment_config_unknown_toml_key_raises(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        env_tag = "f:sphere-2d"
        opt_name = "random"
        num_arms = 4
        num_rounds = 6
        num_reps = 1
        not_a_real_key = 123
        """,
    )
    with pytest.raises(ValueError, match="Unknown key"):
        load_experiment_config([f"--config={toml_path}"])


def test_load_experiment_config_missing_required_raises(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        opt_name = "random"
        num_reps = 1
        """,
    )
    with pytest.raises(ValueError, match="Missing required fields"):
        load_experiment_config([f"--config={toml_path}"])
