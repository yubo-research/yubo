from click.testing import CliRunner


def test_kiss_coverage_smoke_designers_parse_and_context():
    # Minimal coverage smoke for newly-added code units.
    from optimizer.designer_registry import _SimpleContext
    from optimizer.designer_spec import parse_designer_spec

    spec = parse_designer_spec("ts_sweep/num_candidates=10/num_keep=5")
    assert spec.base == "ts_sweep"
    assert spec.specific["num_candidates"] == 10
    assert spec.general["num_keep"] == 5

    ctx = _SimpleContext(
        policy=object(),
        num_arms=1,
        bt=lambda *a, **k: None,
        num_keep=None,
        keep_style=None,
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=1,
    )
    assert ctx.num_arms == 1


def test_kiss_coverage_smoke_ops_catalog_cli():
    # Use Click runner so coverage sees the command bodies.
    import ops.catalog as catalog

    runner = CliRunner()
    res = runner.invoke(catalog.cli, ["environments"])
    assert res.exit_code == 0, res.output
    assert "f:ackley" in res.output
    assert "ant" in res.output
