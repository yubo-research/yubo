import torch
from botorch.models import SingleTaskGP


def _make_simple_gp():
    """Create a simple GP model for testing."""
    X = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    Y = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float64)
    model = SingleTaskGP(X, Y)
    model.eval()
    return model


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


def test_acq_noisy_max_init():
    from acq.acq_noisy_max import AcqNoisyMax

    model = _make_simple_gp()
    acq = AcqNoisyMax(model, q=1)
    assert acq is not None


def test_acq_noisy_max_forward():
    from acq.acq_noisy_max import AcqNoisyMax

    model = _make_simple_gp()
    acq = AcqNoisyMax(model, q=1)
    X = torch.tensor([[[0.3, 0.3]]], dtype=torch.float64)
    result = acq(X)
    assert torch.isfinite(result).all()


def test_acq_pstar_init():
    from acq.acq_pstar import AcqPStar

    model = _make_simple_gp()
    acq = AcqPStar(model, num_X_samples=16, num_ts=32)
    assert acq is not None
