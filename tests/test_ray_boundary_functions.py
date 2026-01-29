import numpy as np


def test_ray_boundary_np_basic():
    from sampling.ray_boundary import ray_boundary_np

    x_0 = np.array([[0.5, 0.5]], dtype=np.float64)
    u = np.array([[1.0, 0.0]], dtype=np.float64)
    x = ray_boundary_np(x_0, u)
    assert x.shape == (1, 2)
    assert np.isclose(x[0, 0], 1.0)
    assert np.isclose(x[0, 1], 0.5)


def test_ray_boundary_np_negative_direction():
    from sampling.ray_boundary import ray_boundary_np

    x_0 = np.array([[0.5, 0.5]], dtype=np.float64)
    u = np.array([[-1.0, 0.0]], dtype=np.float64)
    x = ray_boundary_np(x_0, u)
    assert np.isclose(x[0, 0], 0.0)
    assert np.isclose(x[0, 1], 0.5)


def test_ray_boundary_np_diagonal():
    from sampling.ray_boundary import ray_boundary_np

    x_0 = np.array([[0.5, 0.5]], dtype=np.float64)
    u = np.array([[1.0, 1.0]], dtype=np.float64)
    x = ray_boundary_np(x_0, u)
    assert np.all(x >= 0) and np.all(x <= 1)
