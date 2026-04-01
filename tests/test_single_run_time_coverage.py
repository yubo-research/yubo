"""Module-level coverage for ops/single_run_time.py (kiss test_coverage gate)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

import ops.single_run_time as srt


@patch("ops.single_run_time.subprocess.run")
def test_run_modal_deploy(mock_run):
    mock_run.return_value = MagicMock()
    srt.run_modal_deploy()
    mock_run.assert_called_once()


def test_run_modal_deploy_missing_script():
    fake_root = Path("/nonexistent/repo/root")
    with patch.object(srt, "_REPO_ROOT", fake_root):
        with pytest.raises(click.ClickException, match="Modal app not found"):
            srt.run_modal_deploy()


@patch("experiments.modal_timing_sweep.submit_configs")
def test_run_modal_submit(mock_submit):
    with patch.object(srt, "_load_prep", return_value=lambda _: []):
        srt.run_modal_submit("t", "experiments.batch_preps.prep_tlunar", "results", False)
    mock_submit.assert_called_once_with("t", [], force=False)


def test_run_modal_submit_rejects_non_list():
    with (
        patch.object(srt, "_require_modal"),
        patch.object(srt, "_load_prep", return_value=lambda _: ()),
    ):
        with pytest.raises(click.ClickException, match="list of ExperimentConfig"):
            srt.run_modal_submit("t", "x.y.z", "results", False)


def test_run_modal_submit_rejects_num_reps_not_one():
    from experiments.experiment_sampler import ExperimentConfig

    bad = ExperimentConfig(
        exp_dir="results/x",
        env_tag="g:sphere-1d",
        opt_name="random",
        num_arms=1,
        num_reps=2,
        num_rounds=1,
    )
    with (
        patch.object(srt, "_require_modal"),
        patch.object(srt, "_load_prep", return_value=lambda _: [bad]),
    ):
        with pytest.raises(click.ClickException, match="num_reps==1"):
            srt.run_modal_submit("t", "x.y.z", "results", False)


def test_run_modal_submit_rejects_non_experiment_config_item():
    with (
        patch.object(srt, "_require_modal"),
        patch.object(srt, "_load_prep", return_value=lambda _: [object()]),
    ):
        with pytest.raises(click.ClickException, match="item 0 must be ExperimentConfig"):
            srt.run_modal_submit("t", "x.y.z", "results", False)


@patch("experiments.modal_timing_sweep.collect")
def test_run_modal_collect(mock_collect):
    with patch.object(srt, "_require_modal"):
        srt.run_modal_collect()
    mock_collect.assert_called_once()


@patch("ops.single_run_time.subprocess.run")
def test_run_modal_stop(mock_run):
    mock_run.side_effect = [
        MagicMock(returncode=0),
        MagicMock(returncode=0),
    ]
    srt.run_modal_stop()
    assert mock_run.call_count == 2
    c0 = mock_run.call_args_list[0]
    assert c0[0][0][:3] == ["modal", "app", "stop"]
    assert c0[0][0][3] == "yubo_timing_sweep"
    assert c0[1].get("check") is False
    c1 = mock_run.call_args_list[1]
    assert c1[0][0][:2] == ["modal", "run"]
    assert c1[0][0][-2:] == ["--cmd", "clean_up"]
    assert c1[1].get("check") is True


@patch("ops.single_run_time.subprocess.run")
def test_run_modal_stop_app_stop_nonzero_still_runs_clean_up(mock_run, capsys):
    mock_run.side_effect = [
        MagicMock(returncode=1),
        MagicMock(returncode=0),
    ]
    srt.run_modal_stop()
    assert mock_run.call_count == 2
    err = capsys.readouterr().err
    assert "modal app stop exited 1" in err


def test_deploy_cmd_thin_wrapper():
    with patch.object(srt, "run_modal_deploy") as m:
        srt.deploy_cmd.callback()
    m.assert_called_once()


def test_submit_cmd_thin_wrapper():
    with patch.object(srt, "run_modal_submit") as m:
        srt.submit_cmd.callback(
            batch_tag="a",
            prep="b.c",
            results_dir="r",
            force=True,
        )
    m.assert_called_once_with("a", "b.c", "r", True)


def test_collect_cmd_thin_wrapper():
    with patch.object(srt, "run_modal_collect") as m:
        srt.collect_cmd.callback()
    m.assert_called_once()


def test_stop_cmd_thin_wrapper():
    with patch.object(srt, "run_modal_stop") as m:
        srt.stop_cmd.callback()
    m.assert_called_once()
