import numpy as np


def test_tight_bounding_box_1_basic():
    from sampling.tight_bounding_box import tight_bounding_box_1

    X_0 = np.array([0.5, 0.5])
    X = np.array([[0.5, 0.5], [0.4, 0.5], [0.6, 0.5], [0.1, 0.1], [0.9, 0.9]])
    num_keep = 3

    bounds = tight_bounding_box_1(X_0, X, num_keep)
    assert bounds.shape == (2, 2)
    assert np.all(bounds[0, :] <= bounds[1, :])


def test_tight_bounding_box_1_small_input():
    from sampling.tight_bounding_box import tight_bounding_box_1

    X_0 = np.array([0.5, 0.5])
    X = np.array([[0.5, 0.5]])
    num_keep = 3

    bounds = tight_bounding_box_1(X_0, X, num_keep)
    assert bounds.shape == (2, 2)


def test_tight_bounding_box_basic():
    from sampling.tight_bounding_box import tight_bounding_box

    X_0 = np.array([0.5, 0.5])
    X = np.array([[0.5, 0.5], [0.4, 0.5], [0.6, 0.5], [0.1, 0.1], [0.9, 0.9]])
    num_keep = 3

    idx, bounds = tight_bounding_box(X_0, X, num_keep)
    assert bounds.shape == (2, 2)
    assert len(idx) == num_keep


def test_tight_bounding_box_small_input():
    from sampling.tight_bounding_box import tight_bounding_box

    X_0 = np.array([0.5, 0.5])
    X = np.array([[0.5, 0.5]])
    num_keep = 3

    idx, bounds = tight_bounding_box(X_0, X, num_keep)
    assert bounds.shape == (2, 2)
    assert len(idx) == 1
