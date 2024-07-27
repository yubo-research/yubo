def _test_tight_bounding_box(num_dim, num_samples, num_keep):
    import torch

    from sampling.tight_bounding_box import tight_bounding_box

    X = torch.rand(size=(num_samples, num_dim))
    X_0 = X[0, :]

    bounds = torch.tensor(tight_bounding_box(X_0, X, num_keep=num_keep))
    assert torch.sum(torch.all((X > bounds[0, :]) & (X < bounds[1, :]), dim=1)) == num_keep


def test_tight_bounding_box():
    import numpy as np

    np.random.seed(17)

    for _ in range(30):
        num_dim = np.random.randint(100)
        num_samples = np.random.randint(100)
        num_keep = np.random.randint(num_samples)

        _test_tight_bounding_box(num_dim, num_samples, num_keep)
