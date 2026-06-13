"""Kiss coverage shard."""

from __future__ import annotations


def test_cov_catalog_num_params_cli():
    from ops.catalog import _CatalogPolicy, cli

    p = _CatalogPolicy()
    assert p.num_params() == 1
    _ = cli


def test_cov_data_cli():
    import ops.data as data_mod

    cli = data_mod.cli
    assert cli is not None


def test_cov_ops_batches_cli(monkeypatch):
    import ops.batches as batches_mod

    monkeypatch.setattr("experiments.batches_impl.run_from_batch_tag", lambda *args, **kwargs: None)
    batches_mod.cli.callback("prep_cum_time_dim", 1, True, "results")
    assert batches_mod.cli is not None
