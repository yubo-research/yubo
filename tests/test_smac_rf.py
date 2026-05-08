import numpy as np
import pytest

pytest.importorskip("smac")
pytest.importorskip("ConfigSpace")
pytest.importorskip("pyrfr")

from analysis.fitting_time.smac_rf import SMACRFConfig, SMACRFSurrogate


def test_smac_rf_package_import():
    from analysis.fitting_time import SMACRFConfig as C
    from analysis.fitting_time import SMACRFSurrogate as S

    assert C is not None
    assert S is not None


def test_smac_rf_fit_predict_1d():
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    y = np.sin(2 * np.pi * x[:, 0]) + 0.1 * rng.standard_normal(n)

    model = SMACRFSurrogate(SMACRFConfig(n_trees=10, seed=1))
    info = model.fit(x, y)
    assert "meta" in info
    assert info["meta"]["n_trees"] == 10

    x_test = np.linspace(0.0, 1.0, 20, dtype=np.float64).reshape(-1, 1)
    mean, var = model.predict(x_test)
    assert mean.shape == (20,)
    assert var.shape == (20,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert np.all(var > 0)


def test_smac_rf_multivariate():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((30, 4))
    y = x[:, 0] - 0.5 * x[:, 1]

    model = SMACRFSurrogate(SMACRFConfig(n_trees=5, seed=3))
    model.fit(x, y)
    mean, var = model.predict(x[:5])
    assert mean.shape == (5,)
    assert var.shape == (5,)


def test_smac_rf_explicit_bounds():
    x = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    lo = np.array([0.0])
    hi = np.array([2.0])
    model = SMACRFSurrogate(SMACRFConfig(bounds=(lo, hi), n_trees=4, seed=0))
    model.fit(x, y)
    mean, _ = model.predict(np.array([[1.0]]))
    assert mean.shape == (1,)


def test_smac_rf_ratio_features_above_one_raises():
    model = SMACRFSurrogate(SMACRFConfig(ratio_features=1.5, n_trees=3, seed=0))
    x = np.random.rand(12, 1)
    y = np.random.randn(12)
    with pytest.raises(ValueError, match="ratio_features"):
        model.fit(x, y)


def test_smac_rf_length_mismatch_raises():
    model = SMACRFSurrogate()
    with pytest.raises(ValueError, match="length mismatch"):
        model.fit(np.zeros((5, 1)), np.zeros(3))


def test_smac_rf_predict_before_fit_raises():
    model = SMACRFSurrogate()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.zeros((2, 1)))
