import numpy as np

def test_booth():

    from problems.pure_functions_2 import Booth

    x = np.array([1, 2])
    b = Booth()
    res = b(x)
    assert res > 4

def test_himmelblau():

    from problems.pure_functions_2 import Himmelblau

    x = np.array([3, 4])
    b = Himmelblau()
    res = b(x)
    assert res > 5

def test_matyas():

    from problems.pure_functions_2 import Matyas

    x = np.array([5, 2])
    b = Matyas()
    res = b(x)
    assert res < 7

def test_zettl():

    from problems.pure_functions_2 import Zettl

    x = np.array([7, 3])
    b = Zettl()
    res = b(x)
    assert res > 10

def test_sum_squares():

    from problems.pure_functions_2 import Sum_Squares

    x = 9
    b = Sum_Squares()
    res = b(x)
    assert res > 11

def test_perm():

    from problems.pure_functions_2 import Perm

    x = np.array([7, 3])
    b = Perm()
    res = b(x, beta=20)
    assert res > 15

def test_salomonl():

    from problems.pure_functions_2 import Salomon

    x = np.array([2, 5])
    b = Salomon()
    res = b(x)
    assert res < 8

def test_whitley():

    from problems.pure_functions_2 import Whitley

    x = 25
    b = Whitley()
    res = b(x)
    assert res > 33

def test_brown():

    from problems.pure_functions_2 import Brown

    x = 10
    b = Brown()
    res = b(x)
    assert res > 20

def test_zakharov():

    from problems.pure_functions_2 import Zakharov

    x = 12
    b = Zakharov()
    res = b(x)
    assert res > 30