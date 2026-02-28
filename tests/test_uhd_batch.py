"""Tests for ops/uhd_batch.py core utilities."""

from __future__ import annotations

import json
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def test_config_hash_excludes_seeds():
    from ops.uhd_batch import _config_hash

    base = {"env_tag": "mnist", "num_rounds": 100, "optimizer": "mezo"}
    h1 = _config_hash(base)
    h2 = _config_hash({**base, "problem_seed": 42})
    h3 = _config_hash({**base, "noise_seed_0": 420})
    assert h1 == h2 == h3


def test_config_hash_differs_on_content():
    from ops.uhd_batch import _config_hash

    h1 = _config_hash({"env_tag": "mnist", "num_rounds": 100})
    h2 = _config_hash({"env_tag": "mnist", "num_rounds": 200})
    assert h1 != h2


def test_dict_to_toml_roundtrip():
    import tomllib

    from ops.uhd_batch import _dict_to_toml

    cfg = {
        "env_tag": "mnist",
        "num_rounds": 1000,
        "lr": 0.001,
        "problem_seed": 42,
        "be_sigma_range": [1e-5, 1e-1],
    }
    toml_text = _dict_to_toml(cfg)
    parsed = tomllib.loads(toml_text)["uhd"]
    assert parsed["env_tag"] == "mnist"
    assert parsed["num_rounds"] == 1000
    assert parsed["lr"] == 0.001
    assert parsed["problem_seed"] == 42
    assert parsed["be_sigma_range"] == [1e-5, 1e-1]


def test_dict_to_toml_skips_none():
    from ops.uhd_batch import _dict_to_toml

    toml_text = _dict_to_toml({"a": 1, "b": None, "c": "x"})
    assert "b" not in toml_text


def test_dict_to_toml_bool():
    import tomllib

    from ops.uhd_batch import _dict_to_toml

    toml_text = _dict_to_toml({"flag": True})
    parsed = tomllib.loads(toml_text)["uhd"]
    assert parsed["flag"] is True


def test_parse_eval_lines_basic():
    from ops.uhd_batch import _parse_eval_lines

    log = textwrap.dedent("""\
        UHD-Simple: num_params = 455306, optimizer = mezo
        EVAL: i_iter = 0 sigma = 0.001000 mu = -2.3270 se = 0.0037 y_best = -2.3270
        EVAL: i_iter = 50 sigma = 0.001000 mu = -2.3024 se = 0.0039 y_best = -2.2976 test_acc = 0.0972
    """)
    records = _parse_eval_lines(log)
    assert len(records) == 2
    assert records[0]["i_iter"] == 0
    assert records[0]["rreturn"] == pytest.approx(-2.3270)
    assert records[1]["i_iter"] == 50
    assert records[1]["rreturn"] == pytest.approx(-2.3024)


def test_parse_eval_lines_empty():
    from ops.uhd_batch import _parse_eval_lines

    assert _parse_eval_lines("no eval lines here\n") == []
    assert _parse_eval_lines("") == []


def test_write_trace(tmp_path):
    from ops.uhd_batch import _write_trace

    tp = tmp_path / "traces" / "00000.jsonl"
    records = [
        {"i_iter": 0, "rreturn": -2.3, "dt_prop": 0.0, "dt_eval": 0.0},
        {"i_iter": 1, "rreturn": -2.2, "dt_prop": 0.0, "dt_eval": 0.0},
    ]
    _write_trace(tp, records)

    assert tp.exists()
    assert tp.with_suffix(".done").exists()
    lines = tp.read_text().strip().split("\n")
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["i_iter"] == 0
    assert r0["rreturn"] == -2.3


def test_write_config(tmp_path):
    from ops.uhd_batch import _write_config

    exp_dir = tmp_path / "abc123"
    cfg = {"env_tag": "mnist", "num_rounds": 100, "problem_seed": 42, "optimizer": "mezo"}
    _write_config(exp_dir, cfg)

    config_path = exp_dir / "config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["opt_name"] == "mezo"
    assert config["num_arms"] == 1
    assert "problem_seed" not in config
    assert config["env_tag"] == "mnist"


def test_gen_missing_reps_all_missing(tmp_path):
    from ops.uhd_batch import _gen_missing_reps

    exp_dir = tmp_path / "exp"
    reps = list(_gen_missing_reps(exp_dir, 3, 18))
    assert len(reps) == 3
    assert reps[0] == (0, 18, 180, exp_dir / "traces" / "00000.jsonl")
    assert reps[1] == (1, 19, 190, exp_dir / "traces" / "00001.jsonl")
    assert reps[2] == (2, 20, 200, exp_dir / "traces" / "00002.jsonl")


def test_gen_missing_reps_skips_done(tmp_path):
    from ops.uhd_batch import _gen_missing_reps

    exp_dir = tmp_path / "exp"
    traces_dir = exp_dir / "traces"
    traces_dir.mkdir(parents=True)
    (traces_dir / "00001.done").touch()

    reps = list(_gen_missing_reps(exp_dir, 3, 18))
    assert len(reps) == 2
    assert reps[0][0] == 0
    assert reps[1][0] == 2


def test_load_toml(tmp_path):
    from ops.uhd_batch import _load_toml

    toml_path = tmp_path / "test.toml"
    toml_path.write_text('[uhd]\nenv_tag = "mnist"\nnum-rounds = 100\n')
    cfg = _load_toml(str(toml_path))
    assert cfg["env_tag"] == "mnist"
    assert cfg["num_rounds"] == 100


def test_experiment_dir_deterministic(tmp_path):
    from ops.uhd_batch import _experiment_dir

    cfg = {"env_tag": "mnist", "num_rounds": 100}
    d1 = _experiment_dir(str(tmp_path), cfg)
    d2 = _experiment_dir(str(tmp_path), cfg)
    assert d1 == d2
    assert d1.parent == tmp_path


# ---------------------------------------------------------------------------
# Modal function tests (mocked)
# ---------------------------------------------------------------------------

_FAKE_EVAL = "EVAL: i_iter = 0 sigma = 0.001 mu = -2.33 se = 0.01 y_best = -2.33\n"


def test_uhd_batch_worker():
    from ops.uhd_batch import uhd_batch_worker

    raw_fn = uhd_batch_worker.get_raw_f()
    fake_dict = {}
    completed = SimpleNamespace(stdout=_FAKE_EVAL, stderr="", returncode=0)

    with patch("ops.uhd_batch._results_dict", return_value=fake_dict), patch("subprocess.run", return_value=completed):
        raw_fn(("k1", {"env_tag": "x", "num_rounds": 1}))

    assert "k1" in fake_dict
    assert "EVAL:" in fake_dict["k1"]


def test_uhd_batch_resubmitter():
    from ops.uhd_batch import uhd_batch_resubmitter

    raw_fn = uhd_batch_resubmitter.get_raw_f()
    fake_submitted = {}
    mock_worker = MagicMock()

    with patch("ops.uhd_batch._submitted_dict", return_value=fake_submitted), patch("modal.Function.from_name", return_value=mock_worker):
        raw_fn([("k1", {"a": 1}), ("k2", {"b": 2})])

    assert fake_submitted == {"k1": True, "k2": True}
    mock_worker.spawn_map.assert_called_once()
    spawned = mock_worker.spawn_map.call_args[0][0]
    assert len(spawned) == 2


def test_uhd_batch_resubmitter_skips_submitted():
    from ops.uhd_batch import uhd_batch_resubmitter

    raw_fn = uhd_batch_resubmitter.get_raw_f()
    fake_submitted = {"k1": True}
    mock_worker = MagicMock()

    with patch("ops.uhd_batch._submitted_dict", return_value=fake_submitted), patch("modal.Function.from_name", return_value=mock_worker):
        raw_fn([("k1", {"a": 1}), ("k2", {"b": 2})])

    spawned = mock_worker.spawn_map.call_args[0][0]
    assert len(spawned) == 1
    assert spawned[0][0] == "k2"


def test_uhd_batch_deleter():
    from ops.uhd_batch import uhd_batch_deleter

    raw_fn = uhd_batch_deleter.get_raw_f()
    fake_dict = {"k1": "data", "k2": "data", "k3": "data"}

    with patch("ops.uhd_batch._results_dict", return_value=fake_dict):
        raw_fn(["k1", "k3", "nonexistent"])

    assert "k1" not in fake_dict
    assert "k2" in fake_dict
    assert "k3" not in fake_dict


# ---------------------------------------------------------------------------
# CLI command tests (mocked)
# ---------------------------------------------------------------------------


def test_local_cmd(tmp_path):
    from click.testing import CliRunner

    from ops.uhd_batch import cli, local_cmd  # noqa: F811

    assert local_cmd is not None
    toml_path = tmp_path / "test.toml"
    toml_path.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 10\n')

    with patch("ops.uhd_batch._batch_local") as mock_bl:
        result = CliRunner().invoke(cli, ["local", str(toml_path), "--num-reps", "3"])

    assert result.exit_code == 0, result.output
    mock_bl.assert_called_once()
    cfg = mock_bl.call_args[0][0]
    assert cfg["env_tag"] == "mnist"
    assert mock_bl.call_args[0][1] == 3


def test_modal_cmd(tmp_path):
    from click.testing import CliRunner

    from ops.uhd_batch import cli, modal_cmd  # noqa: F811

    assert modal_cmd is not None
    toml_path = tmp_path / "test.toml"
    toml_path.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 10\n')

    with patch("ops.uhd_batch._batch_modal") as mock_bm:
        result = CliRunner().invoke(cli, ["modal", str(toml_path), "--num-reps", "5"])

    assert result.exit_code == 0, result.output
    mock_bm.assert_called_once()
    cfg = mock_bm.call_args[0][0]
    assert cfg["env_tag"] == "mnist"
    assert mock_bm.call_args[0][1] == 5


def test_collect_cmd():
    from click.testing import CliRunner

    from ops.uhd_batch import cli, collect_cmd  # noqa: F811

    assert collect_cmd is not None
    with patch("ops.uhd_batch._collect") as mock_c:
        result = CliRunner().invoke(cli, ["collect", "--results-dir", "/tmp/res"])

    assert result.exit_code == 0, result.output
    mock_c.assert_called_once_with("/tmp/res")


def test_status_cmd():
    from click.testing import CliRunner

    from ops.uhd_batch import cli, status_cmd  # noqa: F811

    assert status_cmd is not None
    mock_rd = MagicMock()
    mock_rd.len.return_value = 7
    mock_sd = MagicMock()
    mock_sd.len.return_value = 12

    with patch("ops.uhd_batch._results_dict", return_value=mock_rd), patch("ops.uhd_batch._submitted_dict", return_value=mock_sd):
        result = CliRunner().invoke(cli, ["status"])

    assert result.exit_code == 0, result.output
    assert "results_available = 7" in result.output
    assert "submitted = 12" in result.output


def test_cleanup_cmd():
    from click.testing import CliRunner

    from ops.uhd_batch import cleanup_cmd, cli  # noqa: F811

    assert cleanup_cmd is not None
    with patch("modal.Dict.delete") as mock_del:
        result = CliRunner().invoke(cli, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert mock_del.call_count == 2
    deleted = {c.args[0] for c in mock_del.call_args_list}
    assert deleted == {"uhd_batch_results", "uhd_batch_submitted"}
    assert "Deleted dict: uhd_batch_results" in result.output
