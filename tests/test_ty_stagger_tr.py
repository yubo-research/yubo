def test_create_trust_region_no_kernel_uses_length():
    import numpy as np
    import torch

    import acq.turbo_yubo.ty_stagger_tr as ty_stagger_tr

    x_center = torch.tensor([0.6, 0.4, 0.5], dtype=torch.double)
    tr = ty_stagger_tr.TYStaggerTR(num_dim=3, _num_arms=5, s_min=0.1, length_sampler=lambda s_min, s_max: 0.5)
    tr.pre_draw()
    tr.update_from_model([0.1, 0.2])
    lb, ub = tr.create_trust_region(x_center, kernel=object())

    expected_lb = np.clip(x_center.cpu().numpy() - 0.25, 0.0, 1.0)
    expected_ub = np.clip(x_center.cpu().numpy() + 0.25, 0.0, 1.0)
    assert np.allclose(lb, expected_lb)
    assert np.allclose(ub, expected_ub)


def test_create_trust_region_with_kernel_weights():
    import numpy as np
    import torch

    import acq.turbo_yubo.ty_stagger_tr as ty_stagger_tr

    class K:
        def __init__(self):
            self.lengthscale = torch.tensor([2.0, 1.0, 0.5], dtype=torch.double)

    x_center = torch.tensor([0.5, 0.5, 0.5], dtype=torch.double)
    tr = ty_stagger_tr.TYStaggerTR(num_dim=3, _num_arms=3, s_min=0.1, length_sampler=lambda s_min, s_max: 1.0)
    lb, ub = tr.create_trust_region(x_center, kernel=K())

    weights = np.array([2.0, 1.0, 0.5])
    weights = weights / weights.mean()
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
    expected_lb = np.clip(x_center.cpu().numpy() - weights * 0.5, 0.0, 1.0)
    expected_ub = np.clip(x_center.cpu().numpy() + weights * 0.5, 0.0, 1.0)
    assert np.allclose(lb, expected_lb)
    assert np.allclose(ub, expected_ub)


def test_length_changes_between_calls_stateless():
    import numpy as np
    import torch

    import acq.turbo_yubo.ty_stagger_tr as ty_stagger_tr

    values = [0.2, 0.8]

    def next_length(s_min, s_max):
        return values.pop(0)

    x_center = torch.tensor([0.5, 0.5], dtype=torch.double)
    tr = ty_stagger_tr.TYStaggerTR(num_dim=2, _num_arms=2, s_min=0.1, length_sampler=next_length)
    lb1, ub1 = tr.create_trust_region(x_center, kernel=object())
    lb2, ub2 = tr.create_trust_region(x_center, kernel=object())

    w = np.ones(2)
    exp_lb1 = np.clip(x_center.cpu().numpy() - w * 0.2 / 2.0, 0.0, 1.0)
    exp_ub1 = np.clip(x_center.cpu().numpy() + w * 0.2 / 2.0, 0.0, 1.0)
    exp_lb2 = np.clip(x_center.cpu().numpy() - w * 0.8 / 2.0, 0.0, 1.0)
    exp_ub2 = np.clip(x_center.cpu().numpy() + w * 0.8 / 2.0, 0.0, 1.0)

    assert np.allclose(lb1, exp_lb1)
    assert np.allclose(ub1, exp_ub1)
    assert np.allclose(lb2, exp_lb2)
    assert np.allclose(ub2, exp_ub2)
