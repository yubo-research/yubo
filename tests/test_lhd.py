import numpy as np


def test_latin_hypercube_design_shape():
    from sampling.lhd import latin_hypercube_design

    X = latin_hypercube_design(10, 5, seed=42)
    assert X.shape == (10, 5)


def test_latin_hypercube_design_bounds():
    from sampling.lhd import latin_hypercube_design

    X = latin_hypercube_design(100, 3, seed=42)
    assert np.all(X >= 0)
    assert np.all(X <= 1)


def test_latin_hypercube_design_deterministic():
    from sampling.lhd import latin_hypercube_design

    X1 = latin_hypercube_design(10, 5, seed=42)
    X2 = latin_hypercube_design(10, 5, seed=42)
    assert np.allclose(X1, X2)


def test_latin_hypercube_design_different_seeds():
    from sampling.lhd import latin_hypercube_design

    X1 = latin_hypercube_design(10, 5, seed=42)
    X2 = latin_hypercube_design(10, 5, seed=43)
    assert not np.allclose(X1, X2)
