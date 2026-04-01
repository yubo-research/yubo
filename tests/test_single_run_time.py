"""Tests for ops/single_run_time.py CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner


def test_single_run_time_help():
    from ops.single_run_time import cli

    runner = CliRunner()
    r = runner.invoke(cli, ["--help"])
    assert r.exit_code == 0
    assert "deploy" in r.output
    assert "submit" in r.output
    assert "collect" in r.output
    assert "stop" in r.output


def _ctx(cli):
    return click.Context(cli)


@patch("ops.single_run_time.run_modal_deploy")
def test_deploy_cmd_callback(mock_run):
    from ops.single_run_time import cli

    cli.get_command(_ctx(cli), "deploy").callback()
    mock_run.assert_called_once()


@patch("ops.single_run_time.subprocess.run")
def test_deploy_via_runner(mock_run):
    from ops.single_run_time import cli

    mock_run.return_value = MagicMock()
    runner = CliRunner()
    r = runner.invoke(cli, ["deploy"])
    assert r.exit_code == 0
    mock_run.assert_called_once()


def test_deploy_cmd_missing_modal_app():
    from ops import single_run_time as srt

    fake_root = Path("/this/path/does/not/exist/repo")
    with patch.object(srt, "_REPO_ROOT", fake_root):
        with pytest.raises(click.ClickException, match="Modal app not found"):
            srt.cli.get_command(_ctx(srt.cli), "deploy").callback()


def test_load_prep_rejects_bare_name():
    from ops.single_run_time import _load_prep

    with pytest.raises(click.ClickException, match="module.path.function_name"):
        _load_prep("nope")


@patch("ops.single_run_time.run_modal_submit")
def test_submit_cmd_callback(mock_submit):
    from ops.single_run_time import cli

    cli.get_command(_ctx(cli), "submit").callback(
        batch_tag="tag1",
        prep="experiments.batch_preps.prep_tlunar",
        results_dir="results",
        force=False,
    )
    mock_submit.assert_called_once_with("tag1", "experiments.batch_preps.prep_tlunar", "results", False)


@patch("experiments.modal_timing_sweep.submit_configs")
def test_submit_via_runner(mock_submit):
    from ops.single_run_time import cli

    with patch("ops.single_run_time._load_prep", return_value=lambda _: []):
        runner = CliRunner()
        r = runner.invoke(
            cli,
            ["submit", "tag1", "--prep", "experiments.batch_preps.prep_tlunar"],
        )
    assert r.exit_code == 0
    mock_submit.assert_called_once_with("tag1", [], force=False)


@patch("ops.single_run_time.run_modal_collect")
def test_collect_cmd_callback(mock_collect):
    from ops.single_run_time import cli

    cli.get_command(_ctx(cli), "collect").callback()
    mock_collect.assert_called_once()


@patch("experiments.modal_timing_sweep.collect")
def test_collect_via_runner(mock_collect):
    from ops.single_run_time import cli

    with patch("ops.single_run_time._require_modal"):
        runner = CliRunner()
        r = runner.invoke(cli, ["collect"])
    assert r.exit_code == 0
    mock_collect.assert_called_once()


@patch("ops.single_run_time.run_modal_stop")
def test_stop_cmd_callback(mock_stop):
    from ops.single_run_time import cli

    cli.get_command(_ctx(cli), "stop").callback()
    mock_stop.assert_called_once()


@patch("ops.single_run_time.subprocess.run")
def test_stop_via_runner(mock_run):
    from ops.single_run_time import cli

    mock_run.side_effect = [MagicMock(returncode=0), MagicMock(returncode=0)]
    runner = CliRunner()
    r = runner.invoke(cli, ["stop"])
    assert r.exit_code == 0
    assert mock_run.call_count == 2
