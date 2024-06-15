def _p_normal(X_0, X):
    import torch

    num_dim = len(X_0)
    if num_dim == 1:
        sigma_0 = torch.tensor([0.1])
    elif num_dim == 2:
        sigma_0 = torch.tensor([1e-6, 0.1])
    elif num_dim == 3:
        sigma_0 = torch.tensor([1e-6, 5e-3, 0.1])
    else:
        assert False, num_dim
    d2 = ((X - X_0) / sigma_0) ** 2
    p = torch.exp(-(d2 / 2).sum(dim=1))
    # p = torch.maximum(torch.tensor(1e-9), p)
    return p / p.sum()


def test_compare_3d():
    import torch

    from sampling.probe_mv import ProbeMV
    from sampling.stagger import StaggerIS

    torch.manual_seed(17)
    num_dim = 3
    num_samples = 100
    X_0 = torch.rand(size=torch.Size([num_dim]))
    sigma_min = torch.tensor([1e-6, 1e-6, 1e-6])
    sigma_max = torch.tensor([10, 10, 10])

    pm = ProbeMV(X_0, sigma_min, sigma_max)
    iis = StaggerIS(X_0, sigma_min, sigma_max)

    X_pm = None
    X_iis = None

    print()
    for _ in range(30):
        X_pm = pm.ask(num_samples=num_samples, X_and_p_target=(X_pm, _p_normal(X_0, X_pm)) if X_pm is not None else None)
        X_iis = iis.ask(num_samples_per_dimension=num_samples, X_and_p_target=(X_iis, _p_normal(X_0, X_iis)) if X_iis is not None else None)

        print()
        print("TSP:", (X_pm - X_0).std(), pm.sigma_estimate(), pm.convergence_criterion())
        print("TSS:", (X_iis - X_0).std(), iis.sigma_estimate(), iis.convergence_criterion())

    assert torch.abs(pm.sigma_estimate() / iis.sigma_estimate() - 1).max() < 0.3, torch.abs(pm.sigma_estimate() / iis.sigma_estimate() - 1)


def test_compare_1d():
    import numpy as np
    import torch

    from sampling.probe_mv import ProbeMV
    from sampling.stagger import StaggerIS

    torch.manual_seed(17)
    num_dim = 1
    num_samples = 100
    X_0 = torch.rand(size=torch.Size([num_dim]))
    sigma_min = torch.tensor([1e-6])
    sigma_max = torch.tensor([10])

    pm = ProbeMV(X_0, sigma_min, sigma_max)
    iis = StaggerIS(X_0, sigma_min, sigma_max)

    X_pm = None
    X_iis = None

    print()
    for _ in range(10):
        X_pm = pm.ask(num_samples=num_samples, X_and_p_target=(X_pm, _p_normal(X_0, X_pm)) if X_pm is not None else None)
        X_iis = iis.ask(num_samples_per_dimension=num_samples, X_and_p_target=(X_iis, _p_normal(X_0, X_iis)) if X_iis is not None else None)

        print()
        print("TSP:", (X_pm - X_0).std(), float(pm.sigma_estimate()), float(pm.convergence_criterion()))
        print("TSS:", (X_iis - X_0).std(), float(iis.sigma_estimate()), float(iis.convergence_criterion()))

    assert abs(float(pm.sigma_estimate()) - float(iis.sigma_estimate())) < 0.01
    assert not np.isnan(iis.convergence_criterion())


def test_proposal_stagger():
    import torch

    from sampling.stagger import _proposal_stagger

    torch.manual_seed(17)
    num_dim = 2
    num_samples_per_dimension = 5
    X_0 = torch.rand(size=torch.Size([num_dim]), dtype=torch.double)
    pi, X = _proposal_stagger(
        X_0,
        sigma_min=torch.tensor([1e-6, 0.01]),
        sigma_max=torch.tensor([1e-5, 0.1]),
        num_samples_per_dimension=5,
    )

    num_samples = num_samples_per_dimension * num_dim
    assert pi.shape == (num_samples,)
    assert X.shape == (num_samples, num_dim)

    dev = X - X_0

    assert torch.abs(dev[:5, 1]).max() < 1e-5
    assert torch.abs(dev[5:, 0]).max() < 1e-5


def test_proposal_stagger_narrow():
    import torch

    from sampling.stagger import _proposal_stagger

    torch.manual_seed(17)
    num_dim = 2
    num_samples_per_dimension = 1000
    X_0 = torch.rand(size=torch.Size([num_dim]), dtype=torch.double)
    pi, X = _proposal_stagger(
        X_0,
        sigma_min=torch.tensor([1e-3, 0.01]),
        sigma_max=torch.tensor([1.1e-3, 0.011]),
        num_samples_per_dimension=num_samples_per_dimension,
    )

    num_samples = num_samples_per_dimension * num_dim
    assert pi.shape == (num_samples,)
    assert X.shape == (num_samples, num_dim)

    dev = X - X_0

    assert torch.abs(dev[:num_samples_per_dimension, 0].std() - 1.05e-3) < 0.2e-3
    assert torch.abs(dev[num_samples_per_dimension:, 1].std() - 0.0105) < 0.2


def test_stagger_is():
    import torch

    from sampling.stagger import StaggerIS

    torch.manual_seed(17)
    num_dim = 2
    X_0 = torch.rand(size=torch.Size([num_dim]))

    iis = StaggerIS(X_0)
    num_samples_per_dimension = 10

    X = iis.ask(num_samples_per_dimension=num_samples_per_dimension)
    X_and_p_target = (X, _p_normal(X_0, X))
    X = iis.ask(num_samples_per_dimension=num_samples_per_dimension, X_and_p_target=X_and_p_target)

    dev = X - X_0
    assert torch.abs(dev[:num_samples_per_dimension, 1]).max() < 1e-5
    assert torch.abs(dev[num_samples_per_dimension:, 0]).max() < 1e-5

    XX = iis._unmk_1d(iis._mk_1d(X_0, 99), 99)
    assert torch.all(XX[0, :] == X_0)

    iis.sampler(num_base_samples=100)
