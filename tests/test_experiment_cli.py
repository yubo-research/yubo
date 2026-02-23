import sys
import textwrap
import types
from dataclasses import dataclass

import pytest
from click.testing import CliRunner

from experiments.experiment import cli, load_experiment_config


def _write_toml(tmp_path, content: str):
    path = tmp_path / "exp.toml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


@dataclass
class _StubExperimentConfig:
    exp_dir: str
    env_tag: str
    opt_name: str
    num_arms: int
    num_rounds: int
    num_reps: int
    num_denoise: int | None = None
    num_denoise_passive: int | None = None
    max_proposal_seconds: float | None = None
    max_total_seconds: float | None = None
    b_trace: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "_StubExperimentConfig":
        return cls(**d)


def _install_stub_experiment_sampler(monkeypatch) -> list[tuple[object, object]]:
    """
    Avoid importing the real `experiments.experiment_sampler` (which pulls in torch, etc).
    We only want to validate CLI + TOML parsing here.
    """

    calls: list[tuple[object, object]] = []

    def _sampler(config, distributor_fn):
        calls.append((config, distributor_fn))

    mod = types.ModuleType("experiments.experiment_sampler")
    mod.ExperimentConfig = _StubExperimentConfig
    mod.sampler = _sampler
    mod.scan_local = object()

    monkeypatch.setitem(sys.modules, "experiments.experiment_sampler", mod)
    return calls


def test_cli_help_does_not_import_heavy_deps():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "local" in result.output


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
        runtime_device = "cpu"
        local_workers = 4
        """,
    )
    cfg = load_experiment_config(config_toml_path=str(toml_path))
    assert cfg.exp_dir == "_tmp/exp"
    assert cfg.env_tag == "f:ackley-5d"
    assert cfg.opt_name == "ucb"
    assert cfg.num_arms == 8
    assert cfg.num_rounds == 20
    assert cfg.num_reps == 5
    assert cfg.b_trace is False
    assert cfg.max_total_seconds == 30.5
    assert cfg.runtime_device == "cpu"
    assert cfg.local_workers == 4


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
    cfg = load_experiment_config(config_toml_path=str(toml_path))
    assert cfg.exp_dir == "_tmp/root"
    assert cfg.env_tag == "f:rastrigin-3d"
    assert cfg.opt_name == "random"
    assert cfg.num_arms == 2
    assert cfg.num_rounds == 4
    assert cfg.num_reps == 1


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
        load_experiment_config(config_toml_path=str(toml_path))


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
        load_experiment_config(config_toml_path=str(toml_path))


def test_cli_local_runs_with_stubbed_sampler(monkeypatch, tmp_path):
    calls = _install_stub_experiment_sampler(monkeypatch)
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        env_tag = "f:sphere-2d"
        opt_name = "random"
        num_arms = 4
        num_rounds = 10
        num_reps = 3
        """,
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["local", str(toml_path)])
    assert result.exit_code == 0, result.output

    assert len(calls) == 1
    config, distributor_fn = calls[0]
    assert isinstance(config, _StubExperimentConfig)
    assert distributor_fn is sys.modules["experiments.experiment_sampler"].scan_local


def test_load_experiment_config_toml_optimizer_table(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        env_tag = "f:ackley-2d"
        num_arms = 2
        num_rounds = 3
        num_reps = 1

        [optimizer]
        name = "turbo-enn-fit"

        [optimizer.params]
        acq_type = "ucb"
        num_candidates = 64
        """,
    )
    cfg = load_experiment_config(config_toml_path=str(toml_path))
    assert cfg.opt_name == "turbo-enn-fit/acq_type=ucb/num_candidates=64"


def test_load_experiment_config_toml_optimizer_unknown_key_raises(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        env_tag = "f:ackley-2d"
        num_arms = 2
        num_rounds = 3
        num_reps = 1

        [optimizer]
        name = "turbo-enn-fit"
        bad_key = 1
        """,
    )
    with pytest.raises(ValueError, match="Unknown key"):
        load_experiment_config(config_toml_path=str(toml_path))


def test_load_experiment_config_toml_optimizer_invalid_param_type_raises(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        env_tag = "f:ackley-2d"
        num_arms = 2
        num_rounds = 3
        num_reps = 1

        [optimizer]
        name = "turbo-enn-fit"

        [optimizer.params]
        bad = [1, 2]
        """,
    )
    with pytest.raises(TypeError, match="must be int/float/str/bool"):
        load_experiment_config(config_toml_path=str(toml_path))


def test_load_experiment_config_toml_opt_name_and_optimizer_both_set_raises(tmp_path):
    toml_path = _write_toml(
        tmp_path,
        """
        [experiment]
        exp_dir = "_tmp/exp"
        env_tag = "f:ackley-2d"
        opt_name = "random"
        num_arms = 2
        num_rounds = 3
        num_reps = 1

        [optimizer]
        name = "turbo-enn-fit"
        """,
    )
    with pytest.raises(ValueError, match="both experiment\\.opt_name and \\[optimizer\\]\\.name"):
        load_experiment_config(config_toml_path=str(toml_path))
