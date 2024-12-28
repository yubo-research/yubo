def test_ray_boundary():
    import torch

    from sampling.ray_boundary import ray_boundary

    for _ in range(100):
        x = torch.rand(size=(3, 7), dtype=torch.double)
        u = torch.randn(size=(3, 7))
        u = u / torch.linalg.norm(u, axis=1, keepdims=True)

        x_f = ray_boundary(x, u)
        assert x_f.shape == x.shape

        assert torch.all(((x_f.max(axis=1).values - 1) < 1e-6) | ((x_f.min(axis=1).values - 0) < 1e-6))
