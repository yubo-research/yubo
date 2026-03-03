import numpy as np
import pytest
from enn.turbo.types.appendable_array import AppendableArray

import optimizer.enn_turbo_optimizer as eno
from optimizer.enn_turbo_optimizer import TurboOptimizer


def _appendable(rows: list[list[float]]) -> AppendableArray:
    arr = AppendableArray()
    for row in rows:
        arr.append(np.asarray(row, dtype=float))
    return arr


def test_trim_trailing_obs_clamps_prev_num_obs_for_tr_state():
    opt = object.__new__(TurboOptimizer)
    opt._x_obs = _appendable([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]])
    opt._y_obs = _appendable([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]])
    opt._y_tr_list = [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]]
    opt._yvar_obs = AppendableArray()
    opt._trailing_obs = 4
    opt._incumbent_idx = 0

    class _TRState:
        prev_num_obs = 6

    opt._tr_state = _TRState()

    opt._trim_trailing_obs()

    assert len(opt._y_obs) <= 4
    assert int(opt._tr_state.prev_num_obs) == len(opt._y_obs)


def test_predict_mu_sigma_fills_missing_sigma():
    class _Posterior:
        mu = np.array([1.5], dtype=float)
        sigma = None

    class _Surrogate:
        def predict(self, _x):
            return _Posterior()

    opt = type("O", (), {})()
    opt._num_dim = 2
    opt._bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt._surrogate = _Surrogate()

    mu, sigma = eno.predict_mu_sigma(opt, np.array([0.25, 0.75], dtype=float))
    assert mu.shape == (1, 1)
    assert sigma.shape == (1, 1)
    assert float(sigma[0, 0]) == pytest.approx(0.0)


def test_predict_mu_sigma_validates_input_shape():
    opt = type("O", (), {})()
    opt._num_dim = 2
    opt._bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt._surrogate = object()
    with pytest.raises(ValueError):
        _ = eno.predict_mu_sigma(opt, np.array([[0.1, 0.2, 0.3]], dtype=float))


def test_scalarize_returns_none_on_missing_or_error():
    assert eno.scalarize(type("O", (), {"_tr_state": None})(), np.array([[1.0]], dtype=float)) is None

    class _BadTR:
        def scalarize(self, _y, *, clip=False):
            _ = clip
            raise RuntimeError("fail")

    assert eno.scalarize(type("O", (), {"_tr_state": _BadTR()})(), np.array([[1.0]], dtype=float)) is None

    class _GoodTR:
        def scalarize(self, y, *, clip=False):
            return np.asarray(y, dtype=float) + (10.0 if clip else 1.0)

    out = eno.scalarize(type("O", (), {"_tr_state": _GoodTR()})(), np.array([[2.0]], dtype=float), clip=True)
    np.testing.assert_allclose(out, np.array([[12.0]], dtype=float))


def test_create_optimizer_uses_built_surrogate_and_acq(monkeypatch):
    captured = {}
    base_acq = object()

    class _FakeTurboOptimizer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(eno, "_build_surrogate", lambda _cfg: "surrogate")
    monkeypatch.setattr(eno, "build_acquisition_optimizer", lambda _cfg: base_acq)
    monkeypatch.setattr(eno, "TurboOptimizer", _FakeTurboOptimizer)

    cfg = type("Cfg", (), {})()
    cfg.surrogate = object()
    cfg.acquisition = object()
    cfg.acq_optimizer = object()

    opt = eno.create_optimizer(
        bounds=np.array([[0.0, 1.0]], dtype=float),
        config=cfg,
        rng=np.random.default_rng(0),
    )

    assert isinstance(opt, _FakeTurboOptimizer)
    assert captured["surrogate"] == "surrogate"
    assert captured["acquisition_optimizer"] is base_acq
