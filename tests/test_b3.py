import numpy as np
import sys

sys.path.append('/Users/siddhantanandjadhav/BBO/bbo')
from problems.benchmark_functions_3 import SigOptWrapper, Adjiman, Cola, ArithmeticGeometricMean, BartelsConn, Bird, Corana, Dolan, LennardJones6, McCourt01, MixtureOfGaussians06, Pavianini, Xor, Zimmerman, Problem21, Problem09, McCourtBase

def test_Adjiman():

    x = np.random.uniform(-1, 1, 2)
    b = SigOptWrapper(Adjiman)
    res = b(x)
    assert res < 1 and res > -1

def test_ArithmeticGeometricMean():

    x = np.random.uniform(-1, 1, 2)
    b = SigOptWrapper(ArithmeticGeometricMean)
    res = b(x)
    assert res < 1 and res > -1

def test_Cola():

    x = np.random.uniform(-1, 1, 17)
    b = SigOptWrapper(Cola)
    res = b(x)
    assert res < 1 and res > -1

def test_BartelsConn():

    x = np.random.uniform(-1, 1, 2)
    b = SigOptWrapper(BartelsConn)
    res = b(x)
    assert res < 1 and res > -1

def test_Bird():

    x = np.random.uniform(-1, 1, 2)
    b = SigOptWrapper(Bird)
    res = b(x)
    assert res < 1 and res > -1

def test_Corona():

    x = np.random.uniform(-1, 1, 4)
    b = SigOptWrapper(Corana)
    res = b(x)
    assert res < 1 and res > -1

def test_Dolan():

    x = np.random.uniform(-1, 1, 5)
    b = SigOptWrapper(Dolan)
    res = b(x)
    assert res < 1 and res > -1

def test_LennardJones6():

    x = np.random.uniform(-1, 1, 6)
    b = SigOptWrapper(LennardJones6)
    res = b(x)
    assert res < 1 and res > -1

def test_MixtureOfGaussians06():

    x = np.random.uniform(-1, 1, 8)
    b = SigOptWrapper(MixtureOfGaussians06)
    res = b(x)
    assert res < 1 and res > -1

def test_Pavianini():

    x = np.random.uniform(-1, 1, 10)
    b = SigOptWrapper(Pavianini)
    res = b(x)
    assert res < 1 and res > -1

def test_Xor():

    x = np.random.uniform(-1, 1, 9)
    b = SigOptWrapper(Xor)
    res = b(x)
    assert res < 1 and res > -1

def test_Zimmerman():

    x = np.random.uniform(-1, 1, 2)
    b = SigOptWrapper(Zimmerman)
    res = b(x)
    assert res < 1 and res > -1

def test_Problem21():

    x = np.random.uniform(-1, 1, 1)
    b = SigOptWrapper(Problem21)
    res = b(x)
    assert res < 1 and res > -1

def test_Problem09():

    x = np.random.uniform(-1, 1, 1)
    b = SigOptWrapper(Problem09)
    res = b(x)
    assert res < 1 and res > -1
'''
def test_McCourt01():

    from problems.benchmark_functions_3 import SigOptWrapperMcCourt

    x = np.random.uniform(-1, 1, 7)
    b = SigOptWrapperMcCourt(McCourt01(McCourtBase))
    res = b(x)
    assert res < 1 and res > -1
'''