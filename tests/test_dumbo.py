import pytest


@pytest.mark.parametrize(
    "o, b, q, d",
    [
        (1, 5, 4, 3),
        (6, 1, 4, 3),
        (6, 5, 1, 3),
        (6, 5, 4, 1),
        (6, 5, 4, 3),
        (0, 5, 4, 3),
    ],
)
def test_dumbo_indexing(o, b, q, d):
    import torch

    from model.dumbo import DUMBOGP

    X_m = torch.rand(size=(o, d), dtype=torch.double)
    Y_m = 0.1 * X_m.sum(dim=-1, keepdim=True)

    assert Y_m.shape == (o, 1)

    for use_rank_distance in [False, True]:
        dgp = DUMBOGP(X_m, Y_m, use_rank_distance=use_rank_distance)

        X = torch.rand(size=(b, q, d))
        mvn = dgp.posterior(X)

        assert mvn.mean.shape == (b, q, 1)
        assert mvn.stddev.shape == (b, q)
        assert mvn.covariance_matrix.shape == (b, q, q)
