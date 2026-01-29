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
