"""Direct coverage tests for ops/modal_enn_image.py and CLI thin wrappers.

Module-level imports ensure kiss static test_coverage linking.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

import ops.modal_enn_image as mei
from ops.modal_enn_image import (
    ENN_MODAL_BUILD_COMMANDS,
    ENN_MODAL_IGNORE,
    add_enn_to_image,
    enn_project_root,
)
from ops.single_run_time import cli as srt_cli
from ops.single_run_time import collect_cmd, deploy_cmd, stop_cmd, submit_cmd
from ops.uhd_batch_cli import cli as uhd_cli
from ops.uhd_batch_cli import deploy_cmd as uhd_deploy_cmd
from ops.uhd_batch_cli import stop_cmd as uhd_stop_cmd


class _RecordingImage:
    """Minimal fluent stub for modal.Image."""

    def __init__(self) -> None:
        self.ops: list[tuple[str, tuple, dict]] = []

    def add_local_dir(self, *args, **kwargs):
        self.ops.append(("add_local_dir", args, kwargs))
        return self

    def run_commands(self, *commands):
        self.ops.append(("run_commands", commands, {}))
        return self


def test_enn_project_root_resolves_sibling(tmp_path: Path):
    repo = tmp_path / "yubo"
    repo.mkdir()
    assert enn_project_root(repo) == tmp_path / "enn"
    assert mei.enn_project_root(repo) == tmp_path / "enn"


def test_add_enn_to_image_noop_when_missing(tmp_path: Path):
    image = _RecordingImage()
    out = add_enn_to_image(image, tmp_path / "no_such_enn")
    assert out is image
    assert image.ops == []


def test_add_enn_to_image_noop_when_not_directory(tmp_path: Path):
    enn_file = tmp_path / "enn"
    enn_file.write_text("not a dir")
    image = _RecordingImage()
    out = add_enn_to_image(image, enn_file)
    assert out is image
    assert image.ops == []


def test_add_enn_to_image_copies_and_builds(tmp_path: Path):
    enn_root = tmp_path / "enn"
    enn_root.mkdir()
    image = _RecordingImage()
    out = add_enn_to_image(image, enn_root, remote_path="/root/enn")
    assert out is image
    assert len(image.ops) == 2
    assert image.ops[0][0] == "add_local_dir"
    assert image.ops[0][2]["remote_path"] == "/root/enn"
    assert image.ops[0][2]["ignore"] is ENN_MODAL_IGNORE
    assert image.ops[1][0] == "run_commands"
    assert image.ops[1][1] == ENN_MODAL_BUILD_COMMANDS


def test_single_run_time_deploy_cmd():
    with patch("ops.single_run_time.run_modal_deploy") as mock:
        deploy_cmd.callback()
    mock.assert_called_once()


def test_single_run_time_submit_cmd():
    with patch("ops.single_run_time.run_modal_submit") as mock:
        submit_cmd.callback(
            batch_tag="b",
            prep="x.y.z",
            results_dir="results",
            force=False,
        )
    mock.assert_called_once_with("b", "x.y.z", "results", False)


def test_single_run_time_collect_cmd():
    with patch("ops.single_run_time.run_modal_collect") as mock:
        collect_cmd.callback()
    mock.assert_called_once()


def test_single_run_time_stop_cmd():
    with patch("ops.single_run_time.run_modal_stop") as mock:
        stop_cmd.callback()
    mock.assert_called_once()


def test_single_run_time_cli_deploy_via_runner():
    runner = CliRunner()
    with patch("ops.single_run_time.run_modal_deploy"):
        result = runner.invoke(srt_cli, ["deploy"])
    assert result.exit_code == 0


def test_uhd_batch_cli_deploy_cmd():
    with patch("ops.uhd_batch_cli._deploy_uhd_batch_app") as mock:
        uhd_deploy_cmd.callback()
    mock.assert_called_once()


def test_uhd_batch_cli_stop_cmd():
    with patch("ops.uhd_batch_cli._stop_uhd_batch") as mock:
        uhd_stop_cmd.callback()
    mock.assert_called_once()


def test_uhd_batch_cli_deploy_via_runner():
    runner = CliRunner()
    with patch("ops.uhd_batch_cli._deploy_uhd_batch_app"):
        result = runner.invoke(uhd_cli, ["deploy"])
    assert result.exit_code == 0


def test_uhd_batch_cli_stop_via_runner():
    runner = CliRunner()
    with patch("ops.uhd_batch_cli._stop_uhd_batch"):
        result = runner.invoke(uhd_cli, ["stop"])
    assert result.exit_code == 0
