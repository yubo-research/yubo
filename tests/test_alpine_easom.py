import numpy as np


def test_alpine_function():
    from problems.benchmark_functions_2 import Alpine

    x = np.array([1, 2, 3])
    a = Alpine()
    res = a(x)
    assert res > 3.5


def test_easom():
    from problems.benchmark_functions_2 import Easom

    x = 1
    y = 2

    e = Easom()
    res = e(x, y)
    assert res < 1
