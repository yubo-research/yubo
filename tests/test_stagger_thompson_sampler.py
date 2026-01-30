def test_stagger_thompson_sampler():
    import torch

    from sampling.stagger_thompson_sampler import StaggerThompsonSampler
    from tests.test_util import gp_parabola

    torch.manual_seed(17)

    model, X_max = gp_parabola()
    X_max = torch.atleast_2d(X_max)
    num_dim = X_max.shape[-1]

    pss = StaggerThompsonSampler(model, X_max, num_samples=16)

    for _ in range(3):
        pss.refine(num_refinements=10)
        assert pss.samples().shape == (16, num_dim)


def test_stagger_thompson_sampler_ts_chain():
    import torch

    from sampling.stagger_thompson_sampler import StaggerThompsonSampler
    from tests.test_util import gp_parabola

    torch.manual_seed(42)

    model, X_max = gp_parabola()
    X_max = torch.atleast_2d(X_max)
    num_dim = X_max.shape[-1]

    pss = StaggerThompsonSampler(model, X_max, num_samples=8)
    pss.refine(num_refinements=5)
    pss.ts_chain()
    assert pss.samples().shape == (1, num_dim)
