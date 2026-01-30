import numpy as np


def test_target_directions():
    from sampling.knn_tools import target_directions

    np.random.seed(42)
    x_0 = np.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
    u = target_directions(x_0)
    assert u.shape == x_0.shape
    norms = np.linalg.norm(u, axis=1)
    assert np.allclose(norms, 1.0)


def test_approx_ard():
    from sampling.knn_tools import approx_ard

    x_max = np.array([0.5, 0.5])
    y_max = 1.0
    x_neighbors = np.array([[0.4, 0.5], [0.5, 0.4], [0.6, 0.5]])
    y_neighbors = np.array([0.8, 0.9, 0.7])
    u = approx_ard(x_max, y_max, x_neighbors, y_neighbors)
    assert u.shape == (2,)


def test_random_directions():
    from sampling.knn_tools import random_directions

    u = random_directions(10, 5)
    assert u.shape == (10, 5)
    norms = np.linalg.norm(u, axis=1)
    assert np.allclose(norms, 1.0)


def test_random_corner():
    from sampling.knn_tools import random_corner

    np.random.seed(42)
    corners = random_corner(10, 5)
    assert corners.shape == (10, 5)
    assert np.all((corners == 0) | (corners == 1))


def test_far_clip():
    from sampling.knn_tools import far_clip

    x_0 = np.array([[0.5, 0.5]])
    u = np.array([[1.0, 0.0]])
    x = far_clip(x_0, u, k=2)
    assert x.shape == (1, 2)
    assert np.all(x >= 0) and np.all(x <= 1)


def test_single_coordinate_perturbation():
    from sampling.knn_tools import single_coordinate_perturbation

    np.random.seed(42)
    x_0 = np.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]])
    x_f = single_coordinate_perturbation(x_0)
    assert x_f.shape == x_0.shape


def test_raasp_knn():
    from sampling.knn_tools import raasp

    np.random.seed(42)
    x_0 = np.array([[0.5, 0.5, 0.5]])
    x = raasp(x_0, num_perturb=2)
    assert x.shape == x_0.shape


def test_clip_to_boundary():
    from sampling.knn_tools import clip_to_boundary

    x_0 = np.array([[0.5, 0.5]], dtype=np.float64)
    u = np.array([[1.0, 0.0]], dtype=np.float64)
    x = clip_to_boundary(x_0, u)
    assert x.shape == (1, 2)
    assert np.all(x >= 0) and np.all(x <= 1)


class MockENN:
    def __init__(self, X):
        self._X = np.array(X)

    def about_neighbors(self, x, k=1):
        x = np.atleast_2d(x)
        dists = np.linalg.norm(x[:, None, :] - self._X[None, :, :], axis=2)
        idx = np.argsort(dists, axis=1)[:, :k]
        sorted_dists = np.take_along_axis(dists, idx, axis=1)
        return idx, sorted_dists

    def idx_fast(self, x):
        x = np.atleast_2d(x)
        dists = np.linalg.norm(x[:, None, :] - self._X[None, :, :], axis=2)
        return np.argmin(dists, axis=1)


def test_nearest_neighbor():
    from sampling.knn_tools import nearest_neighbor

    X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    enn = MockENN(X)
    x = np.array([[0.4, 0.4], [0.8, 0.8]])
    idx, dist = nearest_neighbor(enn, x, p_boundary_is_neighbor=0.0)
    assert idx.shape == (2,)
    assert dist.shape == (2,)


def test_most_isolated():
    from sampling.knn_tools import most_isolated

    X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    enn = MockENN(X)
    x = np.array([[0.2, 0.2], [0.5, 0.5], [0.7, 0.7]])
    result = most_isolated(enn, x, p_boundary_is_neighbor=0.0)
    assert len(result) >= 1


class MockENNForFarthest(MockENN):
    def posterior(self, x):
        from types import SimpleNamespace

        n = x.shape[0]
        return SimpleNamespace(se=np.ones(n) * 0.5)


def test_farthest_neighbor_fast():
    from sampling.knn_tools import farthest_neighbor_fast

    X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    enn = MockENNForFarthest(X)
    x_0 = np.array([[0.3, 0.3], [0.7, 0.7]])
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = farthest_neighbor_fast(enn, x_0, u, num_steps=5, p_boundary_is_neighbor=0.0)
    assert x.shape == (2, 2)


def test_confidence_region_fast():
    from sampling.knn_tools import confidence_region_fast

    X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    enn = MockENNForFarthest(X)
    x_0 = np.array([[0.3, 0.3], [0.7, 0.7]])
    u = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = confidence_region_fast(enn, x_0, u, se_max=1.0, num_steps=5)
    assert x.shape == (2, 2)


def test_farthest_true_per_sample():
    from sampling.knn_tools import _farthest_true

    x = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
        [[0.7, 0.7], [0.8, 0.8], [0.9, 0.9]],
    ])
    a = np.array([
        [True, True, True],
        [True, True, True],
        [False, True, True],
    ])

    result = _farthest_true(x, a)

    assert result[0, 0] == 0.4, f"Sample 0 should use step 1, got {result[0]}"
    assert result[1, 0] == 0.8, f"Sample 1 should use step 2, got {result[1]}"
    assert result[2, 0] == 0.9, f"Sample 2 should use step 2, got {result[2]}"
