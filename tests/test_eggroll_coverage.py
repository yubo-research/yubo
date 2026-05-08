import json

from click.testing import CliRunner


def test_eggroll_paper_manifest_is_complete():
    from experiments.eggroll_coverage import _paper_coverage

    coverage = _paper_coverage()

    assert coverage.expected == 41
    assert coverage.present == coverage.expected
    assert coverage.missing == []


def test_eggroll_coverage_validate_json_reports_all_statuses():
    from experiments.eggroll_coverage import cli

    result = CliRunner().invoke(cli, ["validate", "--json-output"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["paper"]["present"] == payload["paper"]["expected"]
    assert payload["paper"]["missing"] == []
    assert payload["counts"]["ok"] > 0
    assert payload["counts"]["asset_blocked"] > 0
    assert "adapter_blocked" not in payload["counts"]

    by_path = {row["path"]: row for row in payload["results"]}
    assert by_path["configs/bo/eggroll/paper/rl/cartpole_v1_eggroll.toml"]["status"] == "ok"
    assert by_path["configs/bo/eggroll/paper/qwen/qwen3_4b_deepscaler_rlvr_eggroll.toml"]["status"] == "ok"
