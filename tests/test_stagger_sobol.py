def test_stagger_sobol():
    import torch

    from sampling.stagger_sobol import StaggerSobol

    num_dim = 2
    X_control = torch.rand(size=(1, num_dim))
    ss = StaggerSobol(X_control)

    ss.propose(10)
