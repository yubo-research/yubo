import numpy as np


def test_double_ackley_call():
    from problems.double_ackley import DoubleAckley

    da = DoubleAckley()
    x = np.zeros(20)
    y = da(x)
    assert y.shape == (2,)
    assert np.all(np.isfinite(y))


def test_double_ackley_nonzero():
    from problems.double_ackley import DoubleAckley

    da = DoubleAckley()
    x = np.random.rand(20) * 2 - 1
    y = da(x)
    assert y.shape == (2,)
    assert np.all(np.isfinite(y))
