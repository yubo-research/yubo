import numpy as np


def test_to_unit_cube():
    from turbo_m_ref.utils import to_unit_cube

    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    result = to_unit_cube(x, lb, ub)
    assert result.shape == (2, 2)
    assert np.allclose(result[0], [0.5, 0.5])
    assert np.allclose(result[1], [1.0, 1.0])


def test_from_unit_cube():
    from turbo_m_ref.utils import from_unit_cube

    x = np.array([[0.5, 0.5], [1.0, 1.0]])
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    result = from_unit_cube(x, lb, ub)
    assert result.shape == (2, 2)
    assert np.allclose(result[0], [0.0, 0.0])
    assert np.allclose(result[1], [1.0, 1.0])


def test_latin_hypercube():
    from turbo_m_ref.utils import latin_hypercube

    np.random.seed(42)
    X = latin_hypercube(10, 5)
    assert X.shape == (10, 5)
    assert np.all(X >= 0) and np.all(X <= 1)
