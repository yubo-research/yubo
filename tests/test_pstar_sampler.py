def test_pstar_sampler():
    from acq.acq_util import find_max
    from sampling.pstar_sampler import PStarSampler

    from .test_util import gp_parabola

    model = gp_parabola()[0]

    pss = PStarSampler(
        k_mcmc=5,
        model=model,
        X_max=find_max(model),
    )
    pss(num_X_samples=10)
