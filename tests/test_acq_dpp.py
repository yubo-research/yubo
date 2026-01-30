import torch
from botorch.models import SingleTaskGP


def _make_simple_gp():
    X = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    Y = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float64)
    model = SingleTaskGP(X, Y)
    model.eval()
    return model


def test_gp_init():
    from acq.acq_dpp import _GP

    model = _make_simple_gp()
    gp = _GP(model)
    assert gp.d == 2
    assert gp.model is not None


def test_acq_dpp_init():
    from acq.acq_dpp import AcqDPP

    model = _make_simple_gp()
    acq = AcqDPP(model, num_X_samples=16)
    assert acq is not None
    assert acq._num_dim == 2
