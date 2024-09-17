import numpy as np

def test_alpine_function():

    from problems.alpine_easom_function import alpine

    x = np.array([1, 2, 3])
    a = alpine()
    res = a(x)
    assert res > 3.5

def test_easom():

    from problems.alpine_easom_function import easom

    x = 1
    y = 2

    e = easom()
    res = e(x, y)
    assert res < 1