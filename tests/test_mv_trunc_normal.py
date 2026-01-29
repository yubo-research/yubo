def test_mv_trunc_normal():
    import numpy as np
    import torch
    from scipy.stats import truncnorm

    from sampling.mv_truncated_normal import MVTruncatedNormal

    torch.manual_seed(17)

    mu = torch.tensor([0.1, 0.7])
    sigma = torch.tensor([0.3, 0.01])
    mvtn = MVTruncatedNormal(
        loc=mu,
        scale=sigma,
    )

    loc = mu.numpy()
    scale = sigma.numpy()
    rv = truncnorm(
        a=(0 - loc) / scale, b=(1 - loc) / scale, loc=mu.numpy(), scale=sigma.numpy()
    )

    x = 0.001 + (1 - 2 * 0.001) * np.random.uniform(size=(1000, len(mu)))
    sp = rv.pdf(x)
    sp = sp / sp.sum()
    tn = torch.exp(mvtn._tn.log_prob(torch.tensor(x))).numpy()
    tn = tn / tn.sum()

    assert abs(sp - tn).max() < 1e-6

    x_samples = mvtn._tn.rsample(torch.Size([1000])).numpy()
    assert x_samples.shape == (1000, 2)
    assert x_samples.min() >= 0
    assert x_samples.max() <= 1
