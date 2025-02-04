def test_proposal_stagger():
    import torch

    from sampling.stagger_distribution import StaggerDistribution

    torch.manual_seed(17)
    num_dim = 2
    num_samples_per_dimension = 5
    X_0 = torch.rand(size=torch.Size([num_dim]), dtype=torch.double)
    pi, X = StaggerDistribution(
        X_0,
        num_samples_per_dimension=5,
    ).propose(
        sigma_min=torch.tensor([1e-6, 0.01]),
        sigma_max=torch.tensor([1e-5, 0.1]),
    )

    num_samples = num_samples_per_dimension * num_dim
    assert pi.shape == (num_samples,)
    assert X.shape == (num_samples, num_dim)

    dev = X - X_0

    assert torch.abs(dev[:5, 1]).max() < 1e-5
    assert torch.abs(dev[5:, 0]).max() < 1e-5


def test_proposal_stagger_narrow():
    import torch

    from sampling.stagger_distribution import StaggerDistribution

    torch.manual_seed(17)
    num_dim = 2
    num_samples_per_dimension = 1000
    X_0 = torch.rand(size=torch.Size([num_dim]), dtype=torch.double)
    pi, X = StaggerDistribution(
        X_0,
        num_samples_per_dimension=num_samples_per_dimension,
    ).propose(
        sigma_min=torch.tensor([1e-3, 0.01]),
        sigma_max=torch.tensor([1.1e-3, 0.011]),
    )

    num_samples = num_samples_per_dimension * num_dim
    assert pi.shape == (num_samples,)
    assert X.shape == (num_samples, num_dim)

    dev = X - X_0

    assert torch.abs(dev[:num_samples_per_dimension, 0].std() - 1.05e-3) < 0.2e-3
    assert torch.abs(dev[num_samples_per_dimension:, 1].std() - 0.0105) < 0.2
