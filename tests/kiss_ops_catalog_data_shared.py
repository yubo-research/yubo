import json

from click.testing import CliRunner


def run_kiss_data_locator_optimizers(tmp_path):
    from analysis.data_locator import DataLocator

    results_dir = tmp_path / "results"
    exp_dir = results_dir / "exp_a"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(json.dumps({"opt_name": "random", "env_tag": "f:ackley-2d"}))
    dl = DataLocator(
        results_path=str(results_dir),
        exp_dir="",
        opt_names=["random", "sobol"],
    )
    assert dl.optimizers() == ["random"]


def run_kiss_ops_catalog_and_data_cli(tmp_path):
    import ops.catalog as catalog
    import ops.data as data_cli

    runner = CliRunner()
    catalog.cli.callback()
    catalog.environments.callback()
    res = runner.invoke(catalog.cli, ["environments"])
    assert res.exit_code == 0

    results_dir = tmp_path / "results"
    exp_dir = results_dir / "abc123"
    traces = exp_dir / "traces"
    traces.mkdir(parents=True)
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "opt_name": "random",
                "env_tag": "f:ackley-2d",
                "num_arms": 1,
                "num_rounds": 1,
            }
        )
    )
    (traces / "00000.jsonl").write_text("{}\n")

    res = runner.invoke(data_cli.cli, ["ls", str(results_dir)])
    assert res.exit_code == 0
    assert "abc123" in res.output
    data_cli.cli.callback()
    data_cli.ls.callback(results_dir, False)

    res = runner.invoke(data_cli.cli, ["rm", str(results_dir), "abc123", "-f"])
    assert res.exit_code == 0
    assert not exp_dir.exists()

    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(json.dumps({"opt_name": "random", "env_tag": "f:ackley-2d"}))
    data_cli.rm.callback(results_dir, ("abc123",), True)
    assert not exp_dir.exists()
