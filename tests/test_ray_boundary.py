def test_ray_boundary():
    import numpy as np

    from sampling.ray_boundary import ray_boundary

    x = np.random.uniform(size=(1, 7))
    u = np.random.normal(size=(1, 7))
    u = u / np.linalg.norm(u)

    x_f = ray_boundary(x, u)

    assert ((x_f.max() - 1) < 1e-6) or ((x_f.min() - 0) < 1e-6)
