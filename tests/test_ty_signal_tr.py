import torch


def test_ty_signal_tr_factory_factory():
    from acq.turbo_yubo.ty_signal_tr import ty_signal_tr_factory_factory

    factory = ty_signal_tr_factory_factory(use_gumbel=False)
    tr = factory(num_dim=5, num_arms=2)
    assert tr.num_dim == 5
    assert tr.num_arms == 2


def test_ty_signal_tr_init():
    from acq.turbo_yubo.ty_signal_tr import TYSignalTR

    tr = TYSignalTR(num_dim=5, num_arms=2)
    assert tr.num_dim == 5
    assert tr.num_arms == 2


def test_ty_signal_tr_update_from_model():
    from acq.turbo_yubo.ty_signal_tr import TYSignalTR

    tr = TYSignalTR(num_dim=5, num_arms=2)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    tr.update_from_model(y)
    assert tr._signal is not None


def test_ty_signal_tr_create_trust_region():
    from acq.turbo_yubo.ty_signal_tr import TYSignalTR

    tr = TYSignalTR(num_dim=5, num_arms=2)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    tr.update_from_model(y)
    x_center = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
    lb, ub = tr.create_trust_region(x_center, kernel=None, num_obs=10)
    assert lb.shape == (1, 5)
    assert ub.shape == (1, 5)
