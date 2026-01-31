import numpy as np
import torch


def test_default_bounds():
    from acq.acq_util import default_bounds

    bounds = default_bounds(5)
    assert bounds.shape == (2, 5)
    assert torch.all(bounds[0] == 0.0)
    assert torch.all(bounds[1] == 1.0)


def test_keep_trailing():
    from acq.acq_util import keep_trailing

    Y = np.array([1, 2, 3, 4, 5])
    idx = keep_trailing(Y, 3)
    assert list(idx) == [2, 3, 4]


def test_keep_best():
    from acq.acq_util import keep_best

    Y = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    idx = keep_best(Y, 3)
    assert set(idx) == {1, 2, 4}


def test_unrebound():
    from acq.acq_util import unrebound

    rebounds = np.array([[0.0, 0.0], [2.0, 4.0]])
    X_r = np.array([[0.5, 0.5]])
    X = unrebound(X_r, rebounds)
    assert np.allclose(X, [[1.0, 2.0]])


def test_calc_p_max_from_Y():
    from acq.acq_util import calc_p_max_from_Y

    Y = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
    p_max = calc_p_max_from_Y(Y)
    assert p_max.shape == (3,)
    assert torch.isclose(p_max.sum(), torch.tensor(1.0))


def test_keep_data_none():
    from acq.acq_util import keep_data

    data = [1, 2, 3]
    result = keep_data(data, None, 2)
    assert result == [1, 2, 3]


def test_calc_p_max():
    from botorch.models import SingleTaskGP

    from acq.acq_util import calc_p_max

    X = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    Y = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float64)
    model = SingleTaskGP(X, Y)
    model.eval()

    X_test = torch.tensor([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]], dtype=torch.float64)
    p_max = calc_p_max(model, X_test, num_Y_samples=10)
    assert p_max.shape == (3,)
    assert torch.isclose(p_max.sum(), torch.tensor(1.0))
