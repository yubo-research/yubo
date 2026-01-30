import torch

from tests.test_util import make_simple_gp as _make_simple_gp


def test_acq_min_dist_init():
    from acq.acq_min_dist import AcqMinDist

    model = _make_simple_gp()
    acq = AcqMinDist(model, toroidal=False)
    assert acq is not None


def test_acq_min_dist_forward():
    from acq.acq_min_dist import AcqMinDist

    model = _make_simple_gp()
    acq = AcqMinDist(model, toroidal=False)
    X = torch.tensor([[[0.3, 0.3]]], dtype=torch.float64)
    result = acq(X)
    assert torch.isfinite(result).all()


def test_acq_min_dist_toroidal():
    from acq.acq_min_dist import AcqMinDist

    model = _make_simple_gp()
    acq = AcqMinDist(model, toroidal=True)
    X = torch.tensor([[[0.3, 0.3]]], dtype=torch.float64)
    result = acq(X)
    assert torch.isfinite(result).all()


def test_acq_min_dist_with_x_max():
    from acq.acq_min_dist import AcqMinDist

    model = _make_simple_gp()
    X_max = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    acq = AcqMinDist(model, toroidal=False, X_max=X_max)
    X = torch.tensor([[[0.3, 0.3]]], dtype=torch.float64)
    result = acq(X)
    assert torch.isfinite(result).all()


def test_acq_var_init():
    from acq.acq_var import AcqVar

    model = _make_simple_gp()
    acq = AcqVar(model)
    assert acq is not None


def test_acq_var_forward():
    from acq.acq_var import AcqVar

    model = _make_simple_gp()
    acq = AcqVar(model)
    X = torch.tensor([[[0.3, 0.3]]], dtype=torch.float64)
    result = acq(X)
    assert torch.isfinite(result).all()


def test_acq_ts_init():
    from acq.acq_ts import AcqTS

    model = _make_simple_gp()
    acq = AcqTS(model, num_candidates=100)
    assert acq is not None


def test_acq_ts_draw():
    from acq.acq_ts import AcqTS

    model = _make_simple_gp()
    acq = AcqTS(model, num_candidates=100)
    X_arms = acq.draw(num_arms=2)
    assert X_arms.shape == (2, 2)


def test_acq_sobol_init():
    from acq.acq_sobol import AcqSobol

    model = _make_simple_gp()
    acq = AcqSobol(model)
    assert acq is not None


def test_acq_sobol_draw():
    from acq.acq_sobol import AcqSobol

    model = _make_simple_gp()
    acq = AcqSobol(model)
    X_arms = acq.draw(num_arms=5)
    assert X_arms.shape == (5, 2)


def test_acq_pstar_init():
    from acq.acq_pstar import AcqPStar

    model = _make_simple_gp()
    acq = AcqPStar(model, num_X_samples=16, num_ts=32)
    assert acq is not None


def test_acq_dpp_init():
    from acq.acq_dpp import AcqDPP

    model = _make_simple_gp()
    acq = AcqDPP(model, num_X_samples=16)
    assert acq is not None
    assert acq._num_dim == 2
    assert acq._num_runs == 50


def test_acq_dpp_draw():
    from acq.acq_dpp import AcqDPP

    model = _make_simple_gp()
    acq = AcqDPP(model, num_X_samples=16, num_runs=5)
    X_arms = acq.draw(num_arms=2)
    assert X_arms.shape == (2, 2)


def test_acq_dpp_gp_init():
    from acq.acq_dpp import _GP

    model = _make_simple_gp()
    gp = _GP(model)
    assert gp.d == 2
    assert gp.model is not None


def test_acq_dpp_mean_var():
    from acq.acq_dpp import _GP

    model = _make_simple_gp()
    gp = _GP(model)
    X = torch.tensor([[0.3, 0.3], [0.5, 0.5]], dtype=torch.float64)
    mean, var = gp.mean_var(X)
    assert mean.shape == (2,)
    assert var.shape == (2, 2)


def test_acq_dpp_sample_from_pmax():
    from acq.acq_dpp import _GP

    model = _make_simple_gp()
    gp = _GP(model)
    X = torch.tensor([[0.3, 0.3], [0.5, 0.5], [0.7, 0.7]], dtype=torch.float64)
    x_max = gp.sample_from_pmax(X)
    assert x_max.shape == (2,)


def test_acq_mts_init():
    from acq.acq_mts import AcqMTS

    model = _make_simple_gp()
    acq = AcqMTS(model, num_iterations=3)
    assert acq is not None


def test_acq_mts_draw():
    from acq.acq_mts import AcqMTS

    model = _make_simple_gp()
    acq = AcqMTS(model, num_iterations=3)
    X_arms = acq.draw(num_arms=2)
    assert X_arms.shape == (2, 2)


def test_acq_bt_init():
    from acq.acq_bt import AcqBT
    from acq.acq_var import AcqVar

    acq = AcqBT(
        acq_factory=AcqVar,
        data=[],
        num_dim=2,
        device="cpu",
        dtype=torch.float64,
        num_keep=None,
        keep_style=None,
        model_spec=None,
    )
    assert acq is not None
    assert acq.model() is not None


def test_acq_tsroots_init():
    pytest = __import__("pytest")
    try:
        from acq.acq_tsroots import AcqTSRoots
    except ImportError:
        pytest.skip("tsroots not installed")
        return

    model = _make_simple_gp()
    acq = AcqTSRoots(model)
    assert acq is not None


def test_acq_tsroots_draw():
    pytest = __import__("pytest")
    try:
        from acq.acq_tsroots import AcqTSRoots
    except ImportError:
        pytest.skip("tsroots not installed")
        return

    model = _make_simple_gp()
    acq = AcqTSRoots(model)
    X_arms = acq.draw(num_arms=2)
    assert X_arms.shape == (2, 2)


def test_acq_iopt_init():
    from acq.acq_iopt import AcqIOpt

    model = _make_simple_gp()
    acq = AcqIOpt(model, num_X_samples=32)
    assert acq is not None


def test_acq_iopt_forward():
    from acq.acq_iopt import AcqIOpt

    model = _make_simple_gp()
    acq = AcqIOpt(model, num_X_samples=32)
    X = torch.tensor([[[0.3, 0.3]]], dtype=torch.float64)
    result = acq(X)
    assert torch.isfinite(result).all()


def test_acq_mtv_init():
    from acq.acq_mtv import AcqMTV

    model = _make_simple_gp()
    acq = AcqMTV(model, num_X_samples=32)
    assert acq is not None
