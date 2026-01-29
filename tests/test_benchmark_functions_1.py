import numpy as np


def test_sphere3():
    from problems.benchmark_functions_1 import Sphere3

    f = Sphere3()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_sphere():
    from problems.benchmark_functions_1 import Sphere

    f = Sphere()
    x = np.zeros(5)
    y = f(x)
    assert y == 0.0


def test_ackley():
    from problems.benchmark_functions_1 import Ackley

    f = Ackley()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_beale():
    from problems.benchmark_functions_1 import Beale

    f = Beale()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_branin():
    from problems.benchmark_functions_1 import Branin

    f = Branin()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_bukin():
    from problems.benchmark_functions_1 import Bukin

    f = Bukin()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_cross_in_tray():
    from problems.benchmark_functions_1 import CrossInTray

    f = CrossInTray()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_drop_wave():
    from problems.benchmark_functions_1 import DropWave

    f = DropWave()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_dixon_price():
    from problems.benchmark_functions_1 import DixonPrice

    f = DixonPrice()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_egg_holder():
    from problems.benchmark_functions_1 import EggHolder

    f = EggHolder()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_griewank():
    from problems.benchmark_functions_1 import Griewank

    f = Griewank()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_grlee12():
    from problems.benchmark_functions_1 import GrLee12

    f = GrLee12()
    x = np.array([0.5])
    y = f(x)
    assert np.isfinite(y)


def test_hartmann():
    from problems.benchmark_functions_1 import Hartmann

    f = Hartmann()
    x = np.zeros(6)
    y = f(x)
    assert np.isfinite(y)


def test_holder_table():
    from problems.benchmark_functions_1 import HolderTable

    f = HolderTable()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_levy():
    from problems.benchmark_functions_1 import Levy

    f = Levy()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_michalewicz():
    from problems.benchmark_functions_1 import Michalewicz

    f = Michalewicz()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_powell():
    from problems.benchmark_functions_1 import Powell

    f = Powell()
    x = np.zeros(4)
    y = f(x)
    assert np.isfinite(y)


def test_rastrigin():
    from problems.benchmark_functions_1 import Rastrigin

    f = Rastrigin()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_rosenbrock():
    from problems.benchmark_functions_1 import Rosenbrock

    f = Rosenbrock()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_shubert():
    from problems.benchmark_functions_1 import Shubert

    f = Shubert()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_shekel():
    from problems.benchmark_functions_1 import Shekel

    f = Shekel()
    x = np.zeros(4)
    y = f(x)
    assert np.isfinite(y)


def test_six_hump_camel():
    from problems.benchmark_functions_1 import SixHumpCamel

    f = SixHumpCamel()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_stybtang():
    from problems.benchmark_functions_1 import StybTang

    f = StybTang()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_three_hump_camel():
    from problems.benchmark_functions_1 import ThreeHumpCamel

    f = ThreeHumpCamel()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_langerman():
    from problems.benchmark_functions_1 import Langerman

    f = Langerman()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_levy13():
    from problems.benchmark_functions_1 import Levy13

    f = Levy13()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_bohachevsky1():
    from problems.benchmark_functions_1 import Bohachevsky1

    f = Bohachevsky1()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_rotated_hyper_ellipsoid():
    from problems.benchmark_functions_1 import RotatedHyperEllipsoid

    f = RotatedHyperEllipsoid()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_sum_of_different_powers():
    from problems.benchmark_functions_1 import SumOfDifferentPowers

    f = SumOfDifferentPowers()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_trid():
    from problems.benchmark_functions_1 import Trid

    f = Trid()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_perm():
    from problems.benchmark_functions_1 import Perm

    f = Perm()
    x = np.zeros(4)
    y = f(x)
    assert np.isfinite(y)


def test_schaffer2():
    from problems.benchmark_functions_1 import Schaffer2

    f = Schaffer2()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_schaffer4():
    from problems.benchmark_functions_1 import Schaffer4

    f = Schaffer4()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_schwefel():
    from problems.benchmark_functions_1 import Schwefel

    f = Schwefel()
    x = np.zeros(5)
    y = f(x)
    assert np.isfinite(y)


def test_mccormick():
    from problems.benchmark_functions_1 import McCormick

    f = McCormick()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_power_sum():
    from problems.benchmark_functions_1 import PowerSum

    f = PowerSum()
    x = np.zeros(4)
    y = f(x)
    assert np.isfinite(y)


def test_perm_d_beta():
    from problems.benchmark_functions_1 import PermDBeta

    f = PermDBeta()
    x = np.zeros(4)
    y = f(x)
    assert np.isfinite(y)


def test_goldstein_price():
    from problems.benchmark_functions_1 import GoldsteinPrice

    f = GoldsteinPrice()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)


def test_colville():
    from problems.benchmark_functions_1 import Colville

    f = Colville()
    x = np.zeros(4)
    y = f(x)
    assert np.isfinite(y)


def test_dejong5():
    from problems.benchmark_functions_1 import DeJong5

    f = DeJong5()
    x = np.zeros(2)
    y = f(x)
    assert np.isfinite(y)
