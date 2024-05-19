def test_boot():
    import torch

    from sampling.stagger import boot

    num_dim = 3
    X = torch.rand(size=torch.Size([5, num_dim]))

    for x in boot(X):
        assert x in X


def test_proposal_stagger():
    import torch

    from sampling.stagger import proposal_stagger

    torch.manual_seed(17)
    num_dim = 2
    X_0 = torch.rand(size=torch.Size([num_dim]))
    pi, X = proposal_stagger(X_0, sigma_min=torch.tensor([1e-6, 0.01]), sigma_max=torch.tensor([1e-5, 0.1]), num_samples_per_dimension=5)

    dev = X - X_0

    assert torch.abs(dev[:5, 0]).max() < 1e-5
    assert torch.abs(dev[5:, 1]).mean() > 1e-5


def test_stagger_is():
    import torch

    from sampling.stagger import StaggerIS

    torch.manual_seed(17)
    num_dim = 2
    X_0 = torch.rand(size=torch.Size([num_dim]))

    def p_normal(X):
        nonlocal X_0
        sigma_0 = torch.tensor([1e-6, 0.1])
        d2 = ((X - X_0) / sigma_0) ** 2
        p = torch.exp(-(d2 / 2).sum(dim=1))
        # p = torch.maximum(torch.tensor(1e-9), p)
        return p / p.sum()

    iis = StaggerIS(X_0)
    num_samples_per_dimension = 10

    X = iis.ask(num_samples_per_dimension=num_samples_per_dimension)
    X_and_p_target = (X, p_normal(X))
    X = iis.ask(num_samples_per_dimension=num_samples_per_dimension, X_and_p_target=X_and_p_target)

    dev = X - X_0
    assert torch.abs(dev[:5, 0]).mean() < 1e-3
    assert torch.abs(dev[5:, 1]).mean() > 1e-3
