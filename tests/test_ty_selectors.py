import torch


def test_ty_thompson_empty():
    from acq.turbo_yubo.ty_selectors import ty_thompson

    train_x = torch.tensor([])
    x_cand = torch.rand(10, 3)
    result = ty_thompson(train_x, model=None, x_cand=x_cand, num_arms=2)
    assert result.shape[0] == 2


def test_ty_pareto_import():
    from acq.turbo_yubo.ty_selectors import ty_pareto

    assert ty_pareto is not None
