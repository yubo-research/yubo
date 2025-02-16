import numpy as np


def test_booth():
    from problems.benchmark_functions_2 import Booth

    x = np.array([1, 2])
    b = Booth()
    res = b(x)
    assert res > 4


def test_himmelblau():
    from problems.benchmark_functions_2 import Himmelblau

    x = np.array([3, 4])
    b = Himmelblau()
    res = b(x)
    assert res > 5


def test_matyas():
    from problems.benchmark_functions_2 import Matyas

    x = np.array([5, 2])
    b = Matyas()
    res = b(x)
    assert res > 0


def test_zettl():
    from problems.benchmark_functions_2 import Zettl

    x = np.array([7, 3])
    b = Zettl()
    res = b(x)
    assert res > 10


def test_sum_squares():
    from problems.benchmark_functions_2 import Sum_Squares

    x = [9]
    b = Sum_Squares()
    res = b(x)
    assert res > 11


def test_salomonl():
    from problems.benchmark_functions_2 import Salomon

    x = np.array([2, 5])
    b = Salomon()
    res = b(x)
    assert res > 0


def test_whitley():
    from problems.benchmark_functions_2 import Whitley

    x = [25]
    b = Whitley()
    res = b(x)
    assert res > 33


def test_brown():
    from problems.benchmark_functions_2 import Brown

    x = [10, 4]
    b = Brown()
    res = b(x)
    assert res > 20


def test_zakharov():
    from problems.benchmark_functions_2 import Zakharov

    x = [1, 12]
    b = Zakharov()
    res = b(x)
    assert res > 30
