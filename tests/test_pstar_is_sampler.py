def test_pstar_sampler():
    import numpy as np
    import torch

    from sampling.pstar_is_sampler import PStarISSampler

    from .test_util import gp_parabola

    torch.manual_seed(17)
    model = gp_parabola()[0]

    pss = PStarISSampler(
        k_mcmc=5,
        model=model,
    )
    X = pss(num_X_samples=16)[1].numpy()
    print(pss.appx_normal.mu, pss.appx_normal.sigma)
    assert np.all(X.flatten() >= 0) and np.all(X.flatten() <= 1)
