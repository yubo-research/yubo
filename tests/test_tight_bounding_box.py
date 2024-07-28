def _test_tight_bounding_box(num_dim, num_samples, num_keep):
    import torch

    from sampling.tight_bounding_box import tight_bounding_box

    X = torch.rand(size=(num_samples, num_dim))
    X_0 = X[0, :]

    idx, bounds = tight_bounding_box(X_0, X, num_keep=num_keep)
    bounds = torch.tensor(bounds)
    assert len(idx) == num_keep
    assert torch.all(X[idx, :] > bounds[0, :])
    assert torch.all(X[idx, :] < bounds[1, :])


def test_tight_bounding_box():
    import numpy as np

    np.random.seed(17)

    for _ in range(30):
        num_dim = np.random.randint(100)
        num_samples = np.random.randint(2, 100)
        num_keep = 1 + np.random.randint(num_samples - 1)

        _test_tight_bounding_box(num_dim, num_samples, num_keep)
