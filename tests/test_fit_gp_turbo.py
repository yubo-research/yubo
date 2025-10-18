import torch


def _mk_simple_data(num_samples=12, num_dim=3):
    torch.manual_seed(123)
    X = torch.rand(size=(num_samples, num_dim), dtype=torch.double)
    y = torch.sin(X.sum(dim=1)) + 0.05 * torch.randn(num_samples, dtype=torch.double)
    return X, y


def test_train_gp_shapes_and_device():
    from acq.fit_gp_turbo import train_gp

    X, y = _mk_simple_data()

    model = train_gp(train_x=X, train_y=y, use_ard=True, num_steps=5, hypers={})

    assert model is not None
    assert not model.training

    test_X = torch.rand(7, X.shape[1], dtype=X.dtype, device=X.device)
    posterior = model(test_X)
    assert posterior.mean.shape == (7,)
    assert posterior.covariance_matrix.shape == (7, 7)

    for p in model.parameters():
        assert p.device == X.device
        assert p.dtype == X.dtype


def test_train_gp_no_ard():
    from acq.fit_gp_turbo import train_gp

    X, y = _mk_simple_data()

    model = train_gp(train_x=X, train_y=y, use_ard=False, num_steps=3, hypers={})
    assert model is not None

    test_X = torch.rand(3, X.shape[1], dtype=X.dtype, device=X.device)
    mvn = model(test_X)
    assert mvn.mean.shape == (3,)
    assert mvn.covariance_matrix.shape == (3, 3)


def test_train_gp_with_hypers():
    from acq.fit_gp_turbo import train_gp

    X, y = _mk_simple_data()
    model0 = train_gp(train_x=X, train_y=y, use_ard=True, num_steps=1, hypers={})
    hypers = model0.state_dict()
    model = train_gp(train_x=X, train_y=y, use_ard=True, num_steps=2, hypers=hypers)
    assert model is not None
