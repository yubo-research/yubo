def _test_tight_bounding_box(num_dim, num_samples, num_keep):
    import torch

    from sampling.tight_bounding_box import tight_bounding_box

    X = torch.rand(size=(num_samples, num_dim))
    X_0 = X[0, :]

    idx, bounds = tight_bounding_box(X_0, X, num_keep=num_keep, eps_bounds=1e-9)
    bounds = torch.tensor(bounds)
    assert len(idx) == num_keep

    X_keep = X[idx, :]

    assert torch.all(X_keep > bounds[0, :] - 1e-9)
    assert torch.all(X_keep < bounds[1, :] + 1e-9), torch.maximum(
        torch.tensor(0.0), X_keep - bounds[1, :]
    )


def test_tight_bounding_box():
    import numpy as np
    import torch

    np.random.seed(17)
    torch.manual_seed(17)

    for _ in range(100):
        num_dim = np.random.randint(1, 100)
        num_samples = np.random.randint(2, 100)
        num_keep = np.random.randint(1, num_samples)

        _test_tight_bounding_box(num_dim, num_samples, num_keep)
