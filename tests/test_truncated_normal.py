def test_truncated_normal():
    import numpy as np
    import torch
    from scipy.stats import truncnorm

    from torch_truncnorm.TruncatedNormal import TruncatedNormal

    torch.manual_seed(17)
    np.random.seed(17)

    mu = torch.tensor([0.1])
    sigma = torch.tensor([0.3])

    tn = TruncatedNormal(
        loc=mu,
        scale=sigma,
        a=torch.zeros_like(mu),
        b=torch.ones_like(mu),
    )

    loc = mu.numpy()[0]
    scale = sigma.numpy()[0]
    rv = truncnorm(a=(0 - loc) / scale, b=(1 - loc) / scale, loc=loc, scale=scale)

    x_rv = rv.rvs(1000)
    x_tn = tn.rsample((1000,))
    assert np.abs(x_rv.std() / x_tn.std() - 1) < 0.05

    assert x_tn.min() >= 0
    assert x_tn.max() <= 1


def test_truncated_normal_multiple():
    import numpy as np
    import torch

    from torch_truncnorm.TruncatedNormal import TruncatedNormal

    torch.manual_seed(17)
    np.random.seed(17)

    mu = torch.tensor([0.1] * 1000 + [0.3] * 1000)
    sigma = torch.tensor([0.01] * 1000 + [0.06] * 1000)

    tn = TruncatedNormal(
        loc=mu,
        scale=sigma,
        a=torch.zeros_like(mu),
        b=torch.ones_like(mu),
    )

    x_tn = tn.rsample((1,)).flatten()

    dev = x_tn - mu
    assert torch.abs(dev[:1000].std() - 0.01) < 0.001
    assert torch.abs(dev[1000:].std() - 0.06) < 0.001

    assert x_tn.min() >= 0
    assert x_tn.max() <= 1
