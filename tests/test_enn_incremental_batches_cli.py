from __future__ import annotations

import pytest
from click.testing import CliRunner

import ops.enn_incremental_batches as batches_mod
from tests.modal_cli_test_helpers import (
    assert_modal_command_calls,
    capture_subprocess_run,
)


@pytest.fixture
def captured_modal(monkeypatch):
    return capture_subprocess_run(batches_mod, monkeypatch)


def test_enn_incremental_batches_get_impl_path():
    assert batches_mod._get_impl_path() == "experiments/modal_enn_incremental_batches_impl.py"


def test_enn_incremental_batches_help_shows_exp_type_metavar():
    runner = CliRunner()

    res = runner.invoke(batches_mod.cli, ["deploy", "--help"])

    assert res.exit_code == 0
    assert "Usage: cli deploy [OPTIONS] EXP_TYPE TAG" in res.output


@pytest.mark.parametrize(
    ("invoke_args", "expected_modal_cmds", "modal_tag"),
    [
        (
            ["deploy", "add_method", "t1"],
            [["modal", "deploy", "experiments/modal_enn_incremental_batches_impl.py"]],
            "add_method-t1",
        ),
        (
            ["deploy", "fit_method", "t1"],
            [["modal", "deploy", "experiments/modal_enn_incremental_batches_impl.py"]],
            "fit_method-t1",
        ),
        (
            ["submit", "add_method", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "add_method-t1",
                    "--cmd",
                    "submit",
                    "--output-dir",
                    "results/enn_incremental",
                    "--index-driver",
                    "all",
                    "--num-reps",
                    "10",
                    "--d",
                    "10",
                    "--problem-seed",
                    "17",
                ],
            ],
            "add_method-t1",
        ),
        (
            ["submit", "fit_method", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "fit_method-t1",
                    "--cmd",
                    "submit",
                    "--output-dir",
                    "results/enn_incremental",
                    "--index-driver",
                    "all",
                    "--num-reps",
                    "10",
                    "--d",
                    "10",
                    "--problem-seed",
                    "17",
                ],
            ],
            "fit_method-t1",
        ),
        (
            ["submit-force", "fit_method", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "fit_method-t1",
                    "--cmd",
                    "submit-force",
                    "--output-dir",
                    "results/enn_incremental",
                    "--index-driver",
                    "all",
                    "--num-reps",
                    "10",
                    "--d",
                    "10",
                    "--problem-seed",
                    "17",
                ],
            ],
            "fit_method-t1",
        ),
        (
            [
                "submit",
                "add_method",
                "t1",
                "--index-driver",
                "HNSW",
                "--d",
                "7",
            ],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "add_method-t1",
                    "--cmd",
                    "submit",
                    "--output-dir",
                    "results/enn_incremental",
                    "--index-driver",
                    "hnsw",
                    "--num-reps",
                    "10",
                    "--d",
                    "7",
                    "--problem-seed",
                    "17",
                ],
            ],
            "add_method-t1",
        ),
        (
            [
                "submit",
                "add_method",
                "t1",
                "--num-reps",
                "3",
                "--problem-seed",
                "23",
            ],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "add_method-t1",
                    "--cmd",
                    "submit",
                    "--output-dir",
                    "results/enn_incremental",
                    "--index-driver",
                    "all",
                    "--num-reps",
                    "3",
                    "--d",
                    "10",
                    "--problem-seed",
                    "23",
                ],
            ],
            "add_method-t1",
        ),
        (
            ["collect", "add_method", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "add_method-t1",
                    "--cmd",
                    "collect",
                    "--output-dir",
                    "results/enn_incremental",
                ],
            ],
            "add_method-t1",
        ),
        (
            ["status", "add_method", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "add_method-t1",
                    "--cmd",
                    "status",
                ]
            ],
            "add_method-t1",
        ),
        (
            ["stop", "add_method", "t1"],
            [
                ["modal", "app", "stop", "yubo-enn-incremental-add_method-t1"],
                [
                    "modal",
                    "dict",
                    "delete",
                    "--yes",
                    "--allow-missing",
                    "enn_incremental_results_add_method-t1",
                ],
                [
                    "modal",
                    "dict",
                    "delete",
                    "--yes",
                    "--allow-missing",
                    "enn_incremental_submitted_add_method-t1",
                ],
            ],
            "add_method-t1",
        ),
        (
            ["collect", "fit_method", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "fit_method-t1",
                    "--cmd",
                    "collect",
                    "--output-dir",
                    "results/enn_incremental",
                ],
            ],
            "fit_method-t1",
        ),
        (
            ["status", "fit_method", "t1"],
            [
                [
                    "modal",
                    "run",
                    "experiments/modal_enn_incremental_batches_impl.py::batches",
                    "--tag",
                    "fit_method-t1",
                    "--cmd",
                    "status",
                ]
            ],
            "fit_method-t1",
        ),
        (
            ["stop", "fit_method", "t1"],
            [
                ["modal", "app", "stop", "yubo-enn-incremental-fit_method-t1"],
                [
                    "modal",
                    "dict",
                    "delete",
                    "--yes",
                    "--allow-missing",
                    "enn_incremental_results_fit_method-t1",
                ],
                [
                    "modal",
                    "dict",
                    "delete",
                    "--yes",
                    "--allow-missing",
                    "enn_incremental_submitted_fit_method-t1",
                ],
            ],
            "fit_method-t1",
        ),
    ],
)
def test_enn_incremental_batches_commands_set_modal_tag_and_args(captured_modal, invoke_args, expected_modal_cmds, modal_tag):
    runner = CliRunner()
    res = runner.invoke(batches_mod.cli, invoke_args)
    assert res.exit_code == 0, res.output
    assert_modal_command_calls(captured_modal, expected_modal_cmds, modal_tag=modal_tag)


def test_enn_incremental_batches_local_fit_writes_json(tmp_path, monkeypatch):
    pytest.importorskip("enn")
    runner = CliRunner()
    out = tmp_path / "out"
    res = runner.invoke(
        batches_mod.cli,
        [
            "local-fit",
            "sphere",
            "3",
            "0",
            "flat",
            "--d",
            "2",
            "--problem-seed",
            "17",
            "--num-reps",
            "1",
            "--output-dir",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    dest = out / "enn_fit_D2_sphere_N3_pseed17_nrep1_rep0_flat.json"
    assert dest.exists()


def test_enn_incremental_batches_propagates_subprocess_exit_code(monkeypatch):
    class _Bad:
        returncode = 7

    monkeypatch.setattr(batches_mod.subprocess, "run", lambda *a, **k: _Bad())

    runner = CliRunner()
    res = runner.invoke(batches_mod.cli, ["status", "add_method", "x"])
    assert res.exit_code == 7
