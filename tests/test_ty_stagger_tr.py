def test_create_trust_region_no_kernel_uses_length():
    import numpy as np
    import torch

    import acq.turbo_yubo.ty_stagger_tr as ty_stagger_tr

    x_center = torch.tensor([[0.6, 0.4, 0.5]], dtype=torch.double)
    tr = ty_stagger_tr.TYStaggerTR(num_dim=3, num_arms=5, s_min=0.1)
    tr.pre_draw()
    tr.update_from_model([0.1, 0.2])
    lb, ub = tr.create_trust_region(x_center, kernel=object(), num_obs=2)

    assert lb.shape == (1, 3)
    assert ub.shape == (1, 3)
    assert np.all(lb >= 0.0) and np.all(lb <= 1.0)
    assert np.all(ub >= 0.0) and np.all(ub <= 1.0)
    assert np.all(lb <= ub)


def test_create_trust_region_with_kernel_weights():
    import numpy as np
    import torch

    import acq.turbo_yubo.ty_stagger_tr as ty_stagger_tr

    class K:
        def __init__(self):
            self.lengthscale = torch.tensor([2.0, 1.0, 0.5], dtype=torch.double)

    x_center = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.double)
    tr = ty_stagger_tr.TYStaggerTR(num_dim=3, num_arms=3, s_min=0.1)
    lb, ub = tr.create_trust_region(x_center, kernel=K(), num_obs=3)

    assert lb.shape == (1, 3)
    assert ub.shape == (1, 3)
    assert np.all(lb >= 0.0) and np.all(lb <= 1.0)
    assert np.all(ub >= 0.0) and np.all(ub <= 1.0)
    assert np.all(lb <= ub)

    weights = np.array([2.0, 1.0, 0.5])
    weights = weights / weights.mean()
    weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))

    assert np.all((ub - lb) > 0)


def test_length_changes_between_calls_stateless():
    import numpy as np
    import torch

    import acq.turbo_yubo.ty_stagger_tr as ty_stagger_tr

    x_center = torch.tensor([[0.5, 0.5]], dtype=torch.double)
    tr = ty_stagger_tr.TYStaggerTR(num_dim=2, num_arms=2, s_min=0.1)
    lb1, ub1 = tr.create_trust_region(x_center, kernel=object(), num_obs=2)
    lb2, ub2 = tr.create_trust_region(x_center, kernel=object(), num_obs=2)

    assert lb1.shape == (1, 2)
    assert ub1.shape == (1, 2)
    assert lb2.shape == (1, 2)
    assert ub2.shape == (1, 2)
    assert np.all(lb1 >= 0.0) and np.all(lb1 <= 1.0)
    assert np.all(ub1 >= 0.0) and np.all(ub1 <= 1.0)
    assert np.all(lb2 >= 0.0) and np.all(ub2 <= 1.0)
    assert np.all(ub2 >= 0.0) and np.all(ub2 <= 1.0)
    assert np.all(lb1 <= ub1)
    assert np.all(lb2 <= ub2)
