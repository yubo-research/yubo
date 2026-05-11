import pytest
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


def test_kiss_coverage_smac_rf_fit():
    import numpy as np

    pytest.importorskip("smac")
    from analysis.fitting_time.smac_rf import SMACRFConfig, SMACRFSurrogate

    rng = np.random.default_rng(11)
    x = rng.uniform(0.0, 1.0, size=(22, 1))
    y = (x[:, 0] ** 2).ravel()
    m = SMACRFSurrogate(SMACRFConfig(n_trees=6, seed=9))
    info = m.fit(x, y)
    assert info["meta"]["n_trees"] == 6
    mean, var = m.predict(x[:3])
    assert mean.shape == (3,) and var.shape == (3,)


def test_kiss_coverage_smoke_dngo_fit():
    import numpy as np

    from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate

    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=(18, 1))
    y = np.sin(2 * np.pi * x[:, 0]).reshape(-1, 1)
    m = DNGOSurrogate(DNGOConfig(hidden_width=16, feature_dim=8, num_epochs=30, seed=7))
    info = m.fit(x, y)
    assert np.isfinite(info["marginal_log_likelihood"])


def test_kiss_coverage_smoke_fitting_time_functions():
    import numpy as np
    import torch

    from analysis.fitting_time.fitting_time import (
        fit_dngo,
        fit_enn,
        fit_exact_gp,
        fit_svgp_default,
        fit_svgp_linear,
    )

    rng = np.random.default_rng(0)
    x_np = rng.uniform(0.0, 1.0, size=(30, 2)).astype(np.float64)
    y_np = (x_np[:, 0:1] ** 2 + 0.01 * rng.standard_normal((30, 1))).astype(np.float64)
    x_te = x_np[:7]
    dt, yh, pv = fit_enn(x_np, y_np, x_te)
    assert isinstance(dt, float) and dt >= 0.0
    assert yh.shape == (7, 1)
    assert pv.shape == (7, 1)
    dt, yh, pv = fit_dngo(x_np, y_np, x_te)
    assert isinstance(dt, float) and dt >= 0.0
    assert yh.shape == (7, 1)
    assert pv.shape == (7, 1)
    xt = torch.tensor(x_np, dtype=torch.float64)
    yt = torch.tensor(y_np, dtype=torch.float64)
    x_tt = torch.tensor(x_te, dtype=torch.float64)
    dt, yh, pv = fit_exact_gp(xt, yt, x_tt)
    assert isinstance(dt, float) and dt >= 0.0
    assert tuple(yh.shape) == (7, 1)
    assert tuple(pv.shape) == (7, 1)
    dt, yh, pv = fit_svgp_default(xt, yt, x_tt)
    assert isinstance(dt, float) and dt >= 0.0
    assert tuple(yh.shape) == (7, 1)
    assert tuple(pv.shape) == (7, 1)
    dt, yh, pv = fit_svgp_linear(xt, yt, x_tt)
    assert isinstance(dt, float) and dt >= 0.0
    assert tuple(yh.shape) == (7, 1)
    assert tuple(pv.shape) == (7, 1)


def test_kiss_coverage_smoke_fitting_time_smac_rf():
    import numpy as np

    pytest.importorskip("smac")
    from analysis.fitting_time.fitting_time import fit_smac_rf

    rng = np.random.default_rng(1)
    x_np = rng.uniform(0.0, 1.0, size=(30, 2)).astype(np.float64)
    y_np = (x_np[:, 0] ** 2).astype(np.float64)
    x_te = x_np[:7]
    dt, yh, pv = fit_smac_rf(x_np, y_np, x_te)
    assert isinstance(dt, float) and dt >= 0.0
    assert yh.shape == (7, 1)
    assert pv.shape == (7, 1)


def test_kiss_coverage_smoke_ops_catalog_cli():
    # Use Click runner so coverage sees the command bodies.
    import ops.catalog as catalog

    runner = CliRunner()
    res = runner.invoke(catalog.cli, ["environments"])
    assert res.exit_code == 0, res.output
    assert "f:ackley" in res.output
    assert "ant" in res.output
