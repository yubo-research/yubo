import numpy as np
import pytest


@pytest.mark.parametrize(
    "fn, x, row0, row1",
    [
        ("to", np.array([[0.0, 0.0], [1.0, 1.0]]), [0.5, 0.5], [1.0, 1.0]),
        ("from", np.array([[0.5, 0.5], [1.0, 1.0]]), [0.0, 0.0], [1.0, 1.0]),
    ],
)
def test_unit_cube_maps(fn, x, row0, row1):
    from turbo_m_ref.utils import from_unit_cube, to_unit_cube

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    mapper = to_unit_cube if fn == "to" else from_unit_cube
    result = mapper(x, lb, ub)
    assert result.shape == (2, 2)
    assert np.allclose(result[0], row0)
    assert np.allclose(result[1], row1)


def test_latin_hypercube():
    from turbo_m_ref.utils import latin_hypercube

    np.random.seed(42)
    X = latin_hypercube(10, 5)
    assert X.shape == (10, 5)
    assert np.all(X >= 0) and np.all(X <= 1)
