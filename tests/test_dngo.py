import numpy as np
import pytest
import torch


def test_dngo_net_basis_and_forward():
    from analysis.fitting_time.dngo import _DNGONet

    net = _DNGONet(dim_in=3, hidden_width=8, feature_dim=5, num_middle_layers=2)
    x = torch.randn(7, 3)
    b = net.basis(x)
    out = net.forward(x)
    assert b.shape == (7, 5)
    assert out.shape == (7, 1)


def test_marginal_log_likelihood_and_neg():
    from analysis.fitting_time.dngo import _marginal_log_likelihood, _neg_mll

    rng = np.random.default_rng(3)
    n, k = 25, 6
    phi = rng.standard_normal((n, k))
    y = rng.standard_normal(n)
    mll, parts = _marginal_log_likelihood(0.0, 0.0, phi, y)
    assert np.isfinite(mll)
    assert parts["m_w"].shape == (k,)
    assert parts["chol_a"].shape == (k, k)
    nll = _neg_mll(np.array([0.0, 0.0]), phi, y)
    assert np.isfinite(nll)


def test_marginal_log_likelihood_cholesky_fail_returns_neg_inf():
    from unittest.mock import patch

    from analysis.fitting_time.dngo import _marginal_log_likelihood

    phi = np.ones((4, 2), dtype=np.float64)
    y = np.zeros(4, dtype=np.float64)
    with patch("numpy.linalg.cholesky", side_effect=np.linalg.LinAlgError("singular")):
        mll, parts = _marginal_log_likelihood(0.0, 0.0, phi, y)
    assert mll == -1e25
    assert parts == {}


def test_dngo_fit_optimize_failure_raises():
    from unittest.mock import patch

    from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate

    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 1.0, size=(12, 1))
    y = np.sin(x[:, 0]).reshape(-1, 1)

    def bad_mll(*_a, **_k):
        return -1e25, {}

    model = DNGOSurrogate(DNGOConfig(num_epochs=5, hidden_width=8, feature_dim=4, seed=0))
    with patch("analysis.fitting_time.dngo._marginal_log_likelihood", side_effect=bad_mll):
        with pytest.raises(RuntimeError, match="marginal likelihood"):
            model.fit(x, y)


def test_dngo_package_import():
    from analysis.fitting_time import DNGOConfig, DNGOSurrogate

    assert DNGOConfig is not None
    assert DNGOSurrogate is not None


def test_dngo_fit_reports_val_mse_when_early_stopping_splits():
    from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate

    rng = np.random.default_rng(7)
    n = 50
    x = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float64)
    y = (x[:, 0:1] + 0.1 * rng.standard_normal((n, 1))).astype(np.float64)
    model = DNGOSurrogate(
        DNGOConfig(
            hidden_width=24,
            feature_dim=12,
            num_middle_layers=2,
            num_epochs=30,
            learning_rate=1e-2,
            seed=0,
            val_fraction=0.2,
            min_obs_for_val_split=30,
        )
    )
    info = model.fit(x, y)
    assert "early_stopping_best_val_mse_normalized_y" in info
    assert np.isfinite(info["early_stopping_best_val_mse_normalized_y"])


def test_dngo_mse_readout_mean_matches_network_forward():
    from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate

    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=(25, 1)).astype(np.float64)
    y = (np.sin(2 * np.pi * x[:, 0]) + 0.05 * rng.standard_normal(25)).reshape(-1, 1)
    x_te = x[:6]
    model = DNGOSurrogate(
        DNGOConfig(
            hidden_width=24,
            feature_dim=12,
            num_middle_layers=2,
            num_epochs=60,
            learning_rate=1e-2,
            seed=0,
            use_mse_readout_for_mean=True,
        )
    )
    model.fit(x, y)
    xn = (x_te - model._mx) / model._sx
    with torch.no_grad():
        mu_nn = model._net(torch.from_numpy(xn).float()).numpy().ravel() * model._sy + model._my
    mu_p, _ = model.predict(x_te)
    np.testing.assert_allclose(mu_p, mu_nn, rtol=1e-5, atol=1e-5)


def test_dngo_fit_predict_1d():
    from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate

    rng = np.random.default_rng(0)
    n = 40
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    y = np.sin(2 * np.pi * x[:, 0]) + 0.1 * rng.standard_normal(n)
    y = y.reshape(-1, 1)

    model = DNGOSurrogate(
        DNGOConfig(
            hidden_width=32,
            feature_dim=16,
            num_epochs=80,
            learning_rate=1e-2,
            seed=1,
        )
    )
    info = model.fit(x, y)
    assert np.isfinite(info["marginal_log_likelihood"])
    assert "log_alpha" in info

    x_test = np.linspace(0.0, 1.0, 25, dtype=np.float64).reshape(-1, 1)
    mean, var = model.predict(x_test)
    assert mean.shape == (25,)
    assert var.shape == (25,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert np.all(var > 0)


def test_dngo_y_shape_1d_array():
    from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate

    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, size=(15, 2))
    y = (x[:, 0] + 0.5 * x[:, 1]).ravel()

    model = DNGOSurrogate(DNGOConfig(hidden_width=16, feature_dim=8, num_epochs=40, seed=2))
    model.fit(x, y)
    mean, var = model.predict(x[:3])
    assert mean.shape == (3,)
    assert var.shape == (3,)


def test_dngo_length_mismatch_raises():
    from analysis.fitting_time.dngo import DNGOSurrogate

    model = DNGOSurrogate()
    with pytest.raises(ValueError, match="length mismatch"):
        model.fit(np.zeros((5, 1)), np.zeros((3, 1)))


def test_dngo_predict_before_fit_raises():
    from analysis.fitting_time.dngo import DNGOSurrogate

    model = DNGOSurrogate()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.zeros((2, 1)))
