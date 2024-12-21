def _test_dumbo_indexing(o, b, q, d, m):
    import torch

    from model.dumbo import DUMBOGP

    X_m = torch.rand(size=(o, d))
    Y_m = 0.1 * torch.arange(m) * X_m.sum(dim=-1, keepdim=True)

    assert Y_m.shape == (o, m)

    dgp = DUMBOGP(X_m, Y_m)

    X = torch.rand(size=(b, q, d))
    mvn = dgp.posterior(X)

    assert mvn.mean.shape == (m, b, q)
    assert mvn.stddev.shape == (m, b, q)


def test_dumbo_one_metric():
    _test_dumbo_indexing(6, 5, 4, 3, 1)


def test_dumbo_one_obs():
    _test_dumbo_indexing(1, 5, 4, 3, 2)


def test_dumbo_one_in_batch():
    _test_dumbo_indexing(6, 1, 4, 3, 2)


def test_dumbo_one_arm():
    _test_dumbo_indexing(6, 5, 1, 3, 2)


def test_dumbo_one_dim():
    _test_dumbo_indexing(6, 5, 4, 1, 2)


def test_dumbo_full():
    _test_dumbo_indexing(6, 5, 4, 3, 2)


def test_dumbo_no_obs():
    _test_dumbo_indexing(0, 5, 4, 3, 1)
