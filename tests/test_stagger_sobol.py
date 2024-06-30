def test_stagger_sobol():
    import torch

    from sampling.stagger_sobol import StaggerSobol

    num_dim = 2
    X_control = torch.rand(size=(1, num_dim))
    ss = StaggerSobol(X_control)

    sampler = ss.get_sampler(num_proposal_points=100)
    X = sampler.ask(num_samples=100)

    assert X.min() >= 0
    assert X.max() <= 1
