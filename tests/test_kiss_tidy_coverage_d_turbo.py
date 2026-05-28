from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from tests.kiss_turbo_gp_helper import make_fake_gp
from turbo_m_ref.turbo_1_candidates import create_candidates, select_candidates
from turbo_m_ref.turbo_1_core import init_counters_and_tr, init_hypers


def test_kiss_tidy_d_turbo_candidates_and_core(monkeypatch):
    import turbo_m_ref.turbo_1_candidates as tc

    self_c = SimpleNamespace(dim=2, use_ard=False, _surrogate_type="none", batch_size=1)
    init_hypers(self_c)
    init_counters_and_tr(self_c, batch_size=1)

    gp = make_fake_gp()
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
    x = np.linspace(0.1, 0.9, 8)[:, None]
    x = np.hstack([x, 1.0 - x])
    fx = (x[:, 0] ** 2).reshape(-1, 1)
    cr2 = create_candidates(self_tc, x, fx, 0.5, 30, {})
    assert cr2.X_cand is not None
    sel = select_candidates(self_tc, cr2.X_cand, SimpleNamespace(mu=np.arange(len(cr2.X_cand)), se=np.ones(len(cr2.X_cand))))
    assert sel.shape[0] == 1
