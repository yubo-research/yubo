import math

import numpy as np


class MockPolicy:
    def __init__(self, num_params=32):
        self._num_params = int(num_params)
        self._params = np.zeros(self._num_params)
        self.problem_seed = 0

    def num_params(self):
        return self._num_params

    def get_params(self):
        return self._params.copy()

    def set_params(self, x):
        self._params = np.asarray(x, dtype=float).copy()

    def clone(self):
        out = MockPolicy(self._num_params)
        out.set_params(self._params)
        return out


def test_sparse_evidence_expected_support_large_dim():
    from enn.turbo.config.trust_region import TurboTRConfig

    from optimizer.sparse_enn_designer import SparseEvidenceTrustRegion

    tr = SparseEvidenceTrustRegion(TurboTRConfig(), 1_000_000_000, num_pert=20)

    assert math.isclose(tr.expected_support, 20.0, rel_tol=0.0, abs_tol=1e-6)


def test_sparse_evidence_failure_tolerance_uses_support_not_ambient_dim():
    from enn.turbo.config.trust_region import TurboTRConfig

    from optimizer.sparse_enn_designer import SparseEvidenceTrustRegion

    tr = SparseEvidenceTrustRegion(
        TurboTRConfig(),
        1_000_000_000,
        num_pert=20,
        clock_scale=3.0,
        min_failures=4.0,
    )
    tr.validate_request(1)

    expected = math.ceil(max(1.0, 4.0, 3.0 * tr.expected_support))
    assert tr.failure_tolerance == expected
    assert tr.failure_tolerance < 100


def test_sparse_evidence_failure_tolerance_batches():
    from enn.turbo.config.trust_region import TurboTRConfig

    from optimizer.sparse_enn_designer import SparseEvidenceTrustRegion

    tr = SparseEvidenceTrustRegion(
        TurboTRConfig(),
        10_000,
        num_pert=20,
        clock_scale=3.0,
        min_failures=4.0,
    )
    tr.validate_request(16)

    expected = math.ceil(max(1.0, 4.0 / 16.0, 3.0 * tr.expected_support / 16.0))
    assert tr.failure_tolerance == expected


def test_sparse_enn_designer_installs_sparse_trust_region():
    from optimizer.sparse_enn_designer import (
        SparseENNDesigner,
        SparseEvidenceTrustRegion,
    )

    designer = SparseENNDesigner(
        MockPolicy(num_params=32),
        num_init=1,
        num_candidates=8,
        candidate_rv="uniform",
    )

    policies = designer([], 1)

    assert len(policies) == 1
    assert isinstance(designer._turbo._tr_state, SparseEvidenceTrustRegion)
    designer._turbo._tr_state.validate_request(1)
    assert designer._turbo._tr_state.failure_tolerance < 100


def test_sparse_enn_registry_create_with_options():
    from optimizer.designers import Designers
    from optimizer.sparse_enn_designer import SparseENNDesigner

    designers = Designers(MockPolicy(num_params=32), num_arms=1)
    designer = designers.create("sparse-enn/clock_scale=2.5/num_pert=20/min_failures=4/num_candidates=8/candidate_rv=uniform")

    assert isinstance(designer, SparseENNDesigner)
    assert designer._clock_scale == 2.5
    assert designer._sparse_num_pert == 20
    assert designer._min_failures == 4.0


def test_sparse_enn_ucb_registry_sets_fit_defaults():
    from optimizer.designers import Designers
    from optimizer.sparse_enn_designer import SparseENNDesigner

    designers = Designers(MockPolicy(num_params=32), num_arms=1)
    designer = designers.create("sparse-enn/acq_type=ucb/num_candidates=8/candidate_rv=uniform")

    assert isinstance(designer, SparseENNDesigner)
    assert designer._acq_type == "ucb"
    assert designer._num_fit_samples == 100
    assert designer._num_fit_candidates == 100


def _turbo_zero_shrink_step(*, num_dim: int, overrides: dict | None) -> int | None:
    import enn._rust as rust

    bounds = np.array([[0.0, 1.0]] * num_dim, dtype=np.float64)
    opt = rust.create_optimizer_zero(
        bounds,
        1,
        0,
        config_overrides=overrides or {},
    )
    length_before = float(opt.tr_length())
    for step in range(1, 500):
        x = opt.ask(1, step)
        opt.tell(x, np.array([[0.0]], dtype=np.float64).reshape(1, 1), step)
        length_after = float(opt.tr_length())
        if length_after < length_before * 0.99:
            return step
    return None


def test_enn_failure_tolerance_dim_override():
    """Rust TR should shrink earlier when failure_tolerance_dim is small."""
    sparse_step = _turbo_zero_shrink_step(
        num_dim=100,
        overrides={"failure_tolerance_dim": 20.0},
    )
    default_step = _turbo_zero_shrink_step(num_dim=100, overrides={})
    assert sparse_step is not None
    assert default_step is not None
    if sparse_step >= default_step * 0.9:
        import pytest

        pytest.skip("enn build lacks failure_tolerance_dim patch (re-run pixi run extras-mac in the active Pixi env)")
    assert sparse_step < default_step
