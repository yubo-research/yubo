"""
ipdb> x.shape
torch.Size([6, 3])
ipdb> mean_x.shape
torch.Size([6])
ipdb> covar_x.shape
torch.Size([6, 6])

ipdb> x.shape
torch.Size([5, 10, 3])
ipdb> mean_x.shape
torch.Size([5, 10])
ipdb> covar_x.shape
torch.Size([5, 10, 10])

"""


def _test_dumbo_indexing(o, b, q, d):
    import torch

    # from botorch.models.gp_regression import SingleTaskGP
    from model.dumbo import DUMBOGP

    X_m = torch.rand(size=(o, d), dtype=torch.double)
    Y_m = 0.1 * X_m.sum(dim=-1, keepdim=True)

    assert Y_m.shape == (o, 1)

    dgp = DUMBOGP(X_m, Y_m)

    X = torch.rand(size=(b, q, d))
    mvn = dgp.posterior(X)

    assert mvn.mean.shape == (b, q, 1)
    assert mvn.stddev.shape == (b, q)
    assert mvn.covariance_matrix.shape == (b, q, q)


def test_dumbo_one_obs():
    _test_dumbo_indexing(1, 5, 4, 3)


def test_dumbo_one_in_batch():
    _test_dumbo_indexing(6, 1, 4, 3)


def test_dumbo_one_arm():
    _test_dumbo_indexing(6, 5, 1, 3)


def test_dumbo_one_dim():
    _test_dumbo_indexing(6, 5, 4, 1)


def test_dumbo_full():
    _test_dumbo_indexing(6, 5, 4, 3)


def test_dumbo_no_obs():
    _test_dumbo_indexing(0, 5, 4, 3)
