def test_pstar_sampler():
    from sampling.pstar_is_sampler import PStarISSampler

    from .test_util import gp_parabola

    model = gp_parabola()[0]

    PStarISSampler(
        k_mcmc=5,
        model=model,
        # X_max=find_max(model),
    )
    # pss(num_X_samples=10)
    breakpoint()
