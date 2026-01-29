import numpy as np


def test_mk_2d_single():
    from problems.benchmark_util import mk_2d

    x = np.array([5.0])
    y = mk_2d(x)
    assert y.shape == (1,) or y.shape == (2,)


def test_mk_2d_multiple():
    from problems.benchmark_util import mk_2d

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = mk_2d(x)
    assert y.shape == (2,)


def test_mk_4d():
    from problems.benchmark_util import mk_4d

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = mk_4d(x)
    assert y.shape == (4,)
