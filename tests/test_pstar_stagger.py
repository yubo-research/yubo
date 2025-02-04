def test_pstar_stagger():
    import torch

    from sampling.stagger_thompson_sampler import StaggerThompsonSampler
    from tests.test_util import gp_parabola

    torch.manual_seed(17)

    model, X_max = gp_parabola()
    X_max = torch.atleast_2d(X_max)
    num_dim = X_max.shape[-1]

    pss = StaggerThompsonSampler(model, X_max, num_samples=16)

    for _ in range(3):
        pss.refine()
        assert pss.samples().shape == (16, num_dim)
