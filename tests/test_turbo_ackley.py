import numpy as np


def test_turbo_ackley_call():
    from problems.turbo_ackley import TurboAckley

    ta = TurboAckley()
    x = np.zeros(5)
    y = ta(x)
    assert np.isfinite(y)


def test_turbo_ackley_nonzero():
    from problems.turbo_ackley import TurboAckley

    ta = TurboAckley()
    x = np.random.rand(5) * 2 - 1
    y = ta(x)
    assert np.isfinite(y)
