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


def test_local_default_checkpoints_match_batch_checkpoint_grid():
    from analysis.fitting_time.fitting_time_enn_incremental import (
        enn_incremental_checkpoint_ns,
    )

    assert batches_mod._resolve_checkpoints(None) == enn_incremental_checkpoint_ns()
    assert batches_mod._resolve_checkpoints("") == enn_incremental_checkpoint_ns()


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
            ["deploy", "fit_ind", "t1"],
            [["modal", "deploy", "experiments/modal_enn_incremental_batches_impl.py"]],
            "fit_ind-t1",
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
def test_enn_incremental_batches_subprocess_commands(captured_modal, invoke_args, expected_modal_cmds, modal_tag):
    runner = CliRunner()
    res = runner.invoke(batches_mod.cli, invoke_args)
    assert res.exit_code == 0, res.output
    assert_modal_command_calls(captured_modal, expected_modal_cmds, modal_tag=modal_tag)


@pytest.mark.parametrize(
    ("invoke_args", "expected_tag", "expected_cmd", "expected_kwargs"),
    [
        (
            ["submit", "add_method", "t1"],
            "add_method-t1",
            "submit",
            {
                "output_dir": "results/enn_incremental",
                "index_driver": "all",
                "num_reps": 10,
                "d_dims": 10,
                "problem_seed": 17,
            },
        ),
        (
            ["submit", "fit_method", "t1"],
            "fit_method-t1",
            "submit",
            {
                "output_dir": "results/enn_incremental",
                "index_driver": "all",
                "num_reps": 10,
                "d_dims": 10,
                "problem_seed": 17,
            },
        ),
        (
            ["submit-force", "fit_method", "t1"],
            "fit_method-t1",
            "submit-force",
            {
                "output_dir": "results/enn_incremental",
                "index_driver": "all",
                "num_reps": 10,
                "d_dims": 10,
                "problem_seed": 17,
            },
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
            "add_method-t1",
            "submit",
            {
                "output_dir": "results/enn_incremental",
                "index_driver": "hnsw",
                "num_reps": 10,
                "d_dims": 7,
                "problem_seed": 17,
            },
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
            "add_method-t1",
            "submit",
            {
                "output_dir": "results/enn_incremental",
                "index_driver": "all",
                "num_reps": 3,
                "d_dims": 10,
                "problem_seed": 23,
            },
        ),
        (
            ["collect", "add_method", "t1"],
            "add_method-t1",
            "collect",
            {"output_dir": "results/enn_incremental"},
        ),
        (
            ["status", "add_method", "t1"],
            "add_method-t1",
            "status",
            {},
        ),
        (
            ["collect", "fit_method", "t1"],
            "fit_method-t1",
            "collect",
            {"output_dir": "results/enn_incremental"},
        ),
        (
            ["status", "fit_method", "t1"],
            "fit_method-t1",
            "status",
            {},
        ),
    ],
)
def test_enn_incremental_batches_client_commands(monkeypatch, captured_modal, invoke_args, expected_tag, expected_cmd, expected_kwargs):
    calls: list[tuple] = []

    def _record(tag, cmd, **kwargs):
        calls.append((tag, cmd, kwargs))

    monkeypatch.setattr(batches_mod, "_run_client_command", _record)
    runner = CliRunner()
    res = runner.invoke(batches_mod.cli, invoke_args)
    assert res.exit_code == 0, res.output
    assert captured_modal == []
    assert len(calls) == 1
    tag, cmd, kwargs = calls[0]
    assert tag == expected_tag
    assert cmd == expected_cmd
    for key, val in expected_kwargs.items():
        assert kwargs[key] == val


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


def test_enn_incremental_batches_local_fit_ind_writes_json(tmp_path):
    pytest.importorskip("enn")
    runner = CliRunner()
    out = tmp_path / "out"
    res = runner.invoke(
        batches_mod.cli,
        [
            "local-fit-ind",
            "sphere",
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
            "--checkpoints",
            "1,3",
        ],
    )
    assert res.exit_code == 0, res.output
    dest = out / "enn_fit_ind_D2_sphere_pseed17_nrep1_rep0_flat.json"
    assert dest.exists()


def test_enn_incremental_batches_propagates_client_exit_code(monkeypatch):
    import sys

    def _exit(_tag, _cmd, **_kwargs):
        sys.exit(7)

    monkeypatch.setattr(batches_mod, "_run_client_command", _exit)

    runner = CliRunner()
    res = runner.invoke(batches_mod.cli, ["status", "add_method", "x"])
    assert res.exit_code == 7
