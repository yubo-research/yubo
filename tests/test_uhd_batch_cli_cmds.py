"""CLI command tests for ops/uhd_batch_cli (split from test_uhd_batch for kiss size limits)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_local_cmd_uses_config_num_reps(tmp_path):
    from click.testing import CliRunner

    from ops.uhd_batch import cli

    toml_path = tmp_path / "test.toml"
    toml_path.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 10\nnum_reps = 7\n')

    with patch("ops.uhd_batch._batch_local") as mock_bl:
        result = CliRunner().invoke(cli, ["local", str(toml_path)])

    assert result.exit_code == 0, result.output
    assert mock_bl.call_args[0][1] == 7


def test_deploy_cmd():
    from click.testing import CliRunner

    from ops.uhd_batch import cli, deploy_cmd  # noqa: F811

    assert deploy_cmd is not None
    with patch("ops.uhd_batch_cli._deploy_uhd_batch_app") as mock_deploy:
        result = CliRunner().invoke(cli, ["deploy"])

    assert result.exit_code == 0, result.output
    mock_deploy.assert_called_once()


def test_modal_cmd(tmp_path):
    from click.testing import CliRunner

    from ops.uhd_batch import cli

    toml_path = tmp_path / "test.toml"
    toml_path.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 10\n')

    with patch("ops.uhd_batch_cli._resolve_batch_modal") as mock_resolve:
        mock_bm = MagicMock()
        mock_resolve.return_value = mock_bm
        result = CliRunner().invoke(cli, ["modal", str(toml_path), "--num-reps", "2"])

    assert result.exit_code == 0, result.output
    mock_bm.assert_called_once()


def test_submit_cmd_config(tmp_path):
    from click.testing import CliRunner

    from ops.uhd_batch import cli, submit_cmd  # noqa: F811

    assert submit_cmd is not None
    toml_path = tmp_path / "test.toml"
    toml_path.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 10\n')

    with (
        patch("ops.uhd_batch_cli._ensure_uhd_batch_app"),
        patch("ops.uhd_batch_cli._batch_modal") as mock_bm,
    ):
        result = CliRunner().invoke(
            cli,
            [
                "submit",
                "--config",
                str(toml_path),
                "--num-reps",
                "5",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_bm.assert_called_once()
    cfg = mock_bm.call_args[0][0]
    assert cfg["env_tag"] == "mnist"
    assert mock_bm.call_args[0][1] == 5


def test_submit_cmd_prep(tmp_path):
    from click.testing import CliRunner

    from ops.uhd_batch import cli, submit_cmd  # noqa: F811

    assert submit_cmd is not None

    with (
        patch("ops.uhd_batch_cli._ensure_uhd_batch_app"),
        patch("ops.uhd_batch_cli._batch_modal") as mock_batch,
    ):
        result = CliRunner().invoke(
            cli,
            [
                "submit",
                "--prep",
                "experiments.uhd_batch_preps.prep_uhd_batch_cheetah",
                "--results-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert mock_batch.call_count == 7
    assert "Submit complete: 7 configs" in result.output


def test_modal_cmd_uses_config_num_reps(tmp_path):
    from click.testing import CliRunner

    from ops.uhd_batch import cli

    toml_path = tmp_path / "test.toml"
    toml_path.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 10\nnum_reps = 9\n')

    with patch("ops.uhd_batch_cli._resolve_batch_modal") as mock_resolve:
        mock_bm = MagicMock()
        mock_resolve.return_value = mock_bm
        result = CliRunner().invoke(cli, ["modal", str(toml_path)])

    assert result.exit_code == 0, result.output
    assert mock_bm.call_args[0][1] == 9


def test_cleanup_cmd():
    from click.testing import CliRunner

    from ops.uhd_batch import cli

    with patch("ops.uhd_batch_cli._require_modal"), patch("modal.Dict.delete") as mock_delete:
        result = CliRunner().invoke(cli, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert mock_delete.call_count == 2


def test_batch_cmd(tmp_path):
    from click.testing import CliRunner

    from ops.uhd_batch import cli

    with (
        patch("ops.uhd_batch_cli._require_modal"),
        patch("ops.uhd_batch_cli._ensure_uhd_batch_app"),
        patch("ops.uhd_batch_cli._load_prep_configs", return_value=[({"env_tag": "mnist"}, 3)]),
        patch("ops.uhd_batch_cli._batch_modal") as mock_bm,
    ):
        result = CliRunner().invoke(cli, ["batch", "experiments.uhd_batch_preps.prep_uhd_batch_cheetah"])

    assert result.exit_code == 0, result.output
    mock_bm.assert_called_once_with({"env_tag": "mnist"}, 3, "results/uhd")


def test_collect_cmd():
    from click.testing import CliRunner

    from ops.uhd_batch import cli, collect_cmd  # noqa: F811

    assert collect_cmd is not None
    with patch("ops.uhd_batch_cli._collect") as mock_c:
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

    with (
        patch("ops.uhd_batch_cli._results_dict", return_value=mock_rd),
        patch("ops.uhd_batch_cli._submitted_dict", return_value=mock_sd),
    ):
        result = CliRunner().invoke(cli, ["status"])

    assert result.exit_code == 0, result.output
    assert "app = yubo_uhd_batch" in result.output
    assert "results_available = 7" in result.output
    assert "submitted = 12" in result.output


def test_stop_cmd():
    from click.testing import CliRunner

    from ops.uhd_batch import cli, stop_cmd  # noqa: F811

    assert stop_cmd is not None
    with patch("ops.uhd_batch_cli._stop_uhd_batch") as mock_stop:
        result = CliRunner().invoke(cli, ["stop"])

    assert result.exit_code == 0, result.output
    mock_stop.assert_called_once()
