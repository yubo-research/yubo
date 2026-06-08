from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from tests.kiss_turbo_gp_helper import make_fake_gp


def test_kiss_tidy_d_turbo_refs(monkeypatch):
    import turbo_m_ref.turbo_1_ask_tell_core as atc
    import turbo_m_ref.turbo_1_candidates as tc
    import turbo_m_ref.turbo_1_core as t1c

    atc.validate_init_args(
        np.zeros(2),
        np.ones(2),
        n_init=2,
        batch_size=1,
        verbose=False,
        use_ard=False,
        max_cholesky_size=0,
        n_training_steps=30,
        device="cpu",
        dtype="float32",
    )
    assert atc.standardize_fX(np.array([1.0, 2.0, 9.0])).sigma > 0
    xc = np.random.default_rng(0).random((6, 2)).astype(np.float64)
    yc = np.random.default_rng(1).standard_normal((6, 2)).astype(np.float64)
    assert atc.select_candidates(2, 2, xc, yc).shape == (2, 2)
    self_at = SimpleNamespace(
        dim=2,
        n_cand=20,
        min_cuda=10**9,
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_cholesky_size=0,
        use_ard=False,
        batch_size=1,
    )

    gp = make_fake_gp()
    monkeypatch.setattr(
        atc,
        "train_gp_model",
        lambda self, X, fX, n_training_steps, hypers, device, dtype: gp,
    )
    monkeypatch.setattr(
        atc,
        "sample_candidates",
        lambda gp, X_cand, device, dtype, batch_size, max_cholesky_size: np.zeros((len(X_cand), batch_size)),
    )
    atc.init_hypers(self_at)
    atc.init_counters_and_tr(self_at, batch_size=1, length_fixed=False)
    atc.device_dtype_for(self_at, 1)
    atc.trust_region_bounds(self_at, np.random.random((4, 2)), np.random.random(4), gp, 0.5)
    assert atc.create_candidates(self_at, np.random.random((4, 2)), np.random.random(4), 0.5, 30, {}).X_cand is not None

    t1c.validate_init_args(
        np.zeros(2),
        np.ones(2),
        n_init=2,
        max_evals=100,
        batch_size=1,
        verbose=False,
        use_ard=False,
        max_cholesky_size=0,
        n_training_steps=30,
        dtype="float32",
    )
    self_c = SimpleNamespace(dim=2, use_ard=False, _surrogate_type="none", batch_size=1)
    t1c.init_hypers(self_c)
    t1c.init_counters_and_tr(self_c, batch_size=1)
    t1c.CandidatesResult(X_cand=np.zeros((1, 2)), y_cand=None, hypers={})
    t1c.make_X_cand(
        self_at,
        x_center=np.zeros((1, 2)),
        lb=np.zeros(2),
        ub=np.ones(2),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert (
        t1c.compute_y_cand(
            self_c,
            X=np.zeros((3, 2)),
            fX=np.zeros((3, 1)),
            X_cand=np.zeros((2, 2)),
            mu=0.0,
            sigma=1.0,
            gp=None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        is None
    )

    self_tc = SimpleNamespace(
        dim=2,
        min_cuda=10**9,
        max_cholesky_size=0,
        use_ard=False,
        batch_size=1,
        n_cand=32,
        device=torch.device("cpu"),
        dtype=torch.float64,
        _surrogate_type="none",
    )
    monkeypatch.setattr(tc, "train_gp", lambda train_x, train_y, use_ard, num_steps, hypers=None: gp)
    acq = SimpleNamespace(torch_random_choice=lambda x, k, replace=False: x[: int(k)])
    monkeypatch.setitem(__import__("sys").modules, "acq.acq_util", acq)
    X = np.linspace(0.1, 0.9, 8)[:, None]
    X = np.hstack([X, 1.0 - X])
    fX = (X[:, 0] ** 2).reshape(-1, 1)
    cr2 = tc.create_candidates(self_tc, X, fX, 0.5, 30, {})
    assert cr2.X_cand is not None
    sel = tc.select_candidates(
        self_tc,
        cr2.X_cand,
        SimpleNamespace(mu=np.arange(len(cr2.X_cand)), se=np.ones(len(cr2.X_cand))),
    )
    assert sel.shape[0] == 1
