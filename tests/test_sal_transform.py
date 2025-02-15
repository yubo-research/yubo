def test_sal_transform():
    import torch

    from acq.sal_transform import SALTransform

    num_metrics = 2
    Y = torch.rand(size=(10, num_metrics))
    Yv = 0 * Y + torch.exp(torch.randn(size=(10, num_metrics)))

    sal = SALTransform(1, 2, 3, 4)
    y_sal, yv_sal = sal(Y, Yv)
    print(y_sal, yv_sal)

    Y_check, Yv_check = sal.untransform(y_sal, yv_sal)
    print(Y_check, Yv_check)

    assert torch.abs(Y - Y_check).mean() < 1e-6
    assert torch.abs(Yv - Yv_check).mean() < 1e-6
