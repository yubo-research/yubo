import numpy as np
import torch


def test_ty_shrink_tr_init():
    from acq.turbo_yubo.ty_shrink_tr import TYShrinkTR

    tr = TYShrinkTR(num_dim=5, num_arms=2)
    assert tr.num_dim == 5
    assert tr.num_arms == 2


def test_ty_shrink_tr_update_from_model():
    from acq.turbo_yubo.ty_shrink_tr import TYShrinkTR

    tr = TYShrinkTR(num_dim=5, num_arms=2)
    Y = np.array([1.0, 2.0, 3.0])
    tr.update_from_model(Y)


def test_ty_shrink_tr_create_trust_region():
    from acq.turbo_yubo.ty_shrink_tr import TYShrinkTR

    tr = TYShrinkTR(num_dim=5, num_arms=2)
    x_center = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
    lb, ub = tr.create_trust_region(x_center, kernel=None, num_obs=10)
    assert lb.shape == (1, 5)
    assert ub.shape == (1, 5)
