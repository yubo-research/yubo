import torch


def test_full_space_tr_init():
    from acq.turbo_yubo.ty_full_space import FullSpaceTR

    tr = FullSpaceTR(num_dim=5, num_arms=2)
    assert tr.num_dim == 5
    assert tr.num_arms == 2


def test_full_space_tr_update_from_model():
    from acq.turbo_yubo.ty_full_space import FullSpaceTR

    tr = FullSpaceTR(num_dim=5, num_arms=2)
    Y = torch.tensor([1.0, 2.0, 3.0])
    tr.update_from_model(Y)


def test_full_space_tr_pre_draw():
    from acq.turbo_yubo.ty_full_space import FullSpaceTR

    tr = FullSpaceTR(num_dim=5, num_arms=2)
    tr.pre_draw()


def test_full_space_tr_create_trust_region():
    from acq.turbo_yubo.ty_full_space import FullSpaceTR

    tr = FullSpaceTR(num_dim=5, num_arms=2)
    x_center = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    lb, ub = tr.create_trust_region(x_center, kernel=None, num_obs=10)
    assert lb.shape == (5,)
    assert ub.shape == (5,)


def test_partial_targeter():
    from acq.turbo_yubo.ty_full_space import PartialTargeter

    pt = PartialTargeter(alpha=0.5)
    x_center = torch.tensor([0.0, 0.0])
    x_target = torch.tensor([1.0, 1.0])
    result = pt(x_center, x_target)
    assert torch.allclose(result, torch.tensor([0.5, 0.5]))
