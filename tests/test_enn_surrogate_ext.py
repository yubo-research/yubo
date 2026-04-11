import types

import numpy as np

import optimizer.enn_surrogate_ext as ext


class _FakeENN:
    def __init__(self):
        self.train_x = np.array(
            [
                [0.1, 0.2],
                [0.4, 0.2],
                [0.1, 0.6],
            ],
            dtype=float,
        )
        self.train_y = np.array([[1.0], [2.0], [3.5]], dtype=float)
        self.num_outputs = 1
        self.calls = 0

    def __len__(self):
        return self.train_x.shape[0]

    def _compute_posterior_internals(self, x, params, flags):
        _ = x, params, flags
        self.calls += 1
        return types.SimpleNamespace(
            idx=np.array([[1, 2]], dtype=int),
            w_normalized=np.array([[[0.75], [0.25]]], dtype=float),
        )


class _GradientTRState:
    def __init__(self):
        self.config = types.SimpleNamespace(pc_rotation_mode=None)
        self.calls = []

    def needs_gradient_signal(self):
        return True

    def observe_local_geometry(self, **kwargs):
        self.calls.append(kwargs)


class _IsoTRState:
    def __init__(self):
        self.config = types.SimpleNamespace(pc_rotation_mode=None, geometry="enn_iso")
        self.calls = []

    def needs_local_geometry(self):
        return False

    def observe_local_geometry(self, **kwargs):
        self.calls.append(kwargs)


def test_update_trust_region_gradient_fallback_reuses_single_local_posterior(monkeypatch):
    surrogate = ext.GeometryENNSurrogate(config=types.SimpleNamespace())
    surrogate._enn = _FakeENN()
    surrogate._params = types.SimpleNamespace(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    tr_state = _GradientTRState()
    x_center = surrogate._enn.train_x[0]
    y_obs = surrogate._enn.train_y[:, 0]
    monkeypatch.setattr(ext, "_gradient_mu_impl", lambda **kwargs: None)

    surrogate.update_trust_region(
        tr_state=tr_state,
        x_center=x_center,
        y_obs=y_obs,
        incumbent_idx=0,
        rng=np.random.default_rng(0),
    )

    assert surrogate._enn.calls == 1
    assert len(tr_state.calls) == 1
    observed = tr_state.calls[0]
    np.testing.assert_allclose(observed["delta_x"], surrogate._enn.train_x[[1, 2]] - x_center)
    np.testing.assert_allclose(observed["weights"], np.array([0.75, 0.25], dtype=float))
    np.testing.assert_allclose(observed["delta_y"], np.array([1.0, 2.5], dtype=float))


def test_update_trust_region_identity_geometry_skips_local_posterior():
    surrogate = ext.GeometryENNSurrogate(config=types.SimpleNamespace())
    surrogate._enn = _FakeENN()
    surrogate._params = types.SimpleNamespace(
        k_num_neighbors=2,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )
    tr_state = _IsoTRState()
    surrogate.update_trust_region(
        tr_state=tr_state,
        x_center=surrogate._enn.train_x[0],
        y_obs=surrogate._enn.train_y[:, 0],
        incumbent_idx=0,
        rng=np.random.default_rng(0),
    )

    assert surrogate._enn.calls == 0
    assert tr_state.calls == []


def test_true_ellipsoid_rho_hook_skips_identity_geometry(monkeypatch):
    surrogate = ext.GeometryENNSurrogate(config=types.SimpleNamespace())
    surrogate.predict = lambda x: (_ for _ in ()).throw(AssertionError("predict should not run"))

    called = []

    class _TRState:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(geometry="enn_iso")

        def observe_incumbent_transition(self, **kwargs):
            called.append(kwargs)

    surrogate._maybe_update_true_ellipsoid_rho(
        tr_state=_TRState(),
        x_center=np.array([0.1, 0.2], dtype=float),
        y_obs=np.array([1.0, 2.0], dtype=float),
        incumbent_idx=0,
    )

    assert called == []
