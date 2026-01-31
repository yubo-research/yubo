import torch


def test_proposal_stagger():
    from sampling.probe_mv import proposal_stagger

    X_0 = torch.tensor([0.5, 0.5])
    pi, X = proposal_stagger(X_0, sigma_min=0.01, sigma_max=0.1, num_samples=10)
    assert X.shape == (10, 2)
    assert pi.shape == (10,)


def test_proposal_normal():
    from sampling.probe_mv import proposal_normal

    X_0 = torch.tensor([0.5, 0.5])
    sigma = torch.tensor([0.1, 0.1])
    pi, X = proposal_normal(X_0, sigma, num_samples=10)
    assert X.shape == (10, 2)
    assert pi.shape == (10,)


def test_probe_mv_init():
    from sampling.probe_mv import ProbeMV

    X_0 = torch.tensor([0.5, 0.5])
    pm = ProbeMV(X_0)
    assert pm.convergence_criterion() == 1000


def test_probe_mv_sigma_estimate():
    from sampling.probe_mv import ProbeMV

    X_0 = torch.tensor([0.5, 0.5])
    pm = ProbeMV(X_0)
    sigma = pm.sigma_estimate()
    assert sigma.shape == (2,)


def test_probe_mv_ask():
    from sampling.probe_mv import ProbeMV

    X_0 = torch.tensor([0.5, 0.5])
    pm = ProbeMV(X_0)
    X = pm.ask(num_samples=10)
    assert X.shape == (10, 2)
