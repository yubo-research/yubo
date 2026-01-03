import torch

from model.enn_botorch_t import EpistemicNearestNeighborsBoTorchT, EpistemicNearestNeighborsWeighterBoTorchT
from model.enn_t import EpistemicNearestNeighborsT
from model.enn_weighter_t import ENNWeighterT


def test_ennt_botorch_matches_ennt_posterior_cpu():
    g = torch.Generator(device="cpu").manual_seed(11)
    n = 40
    d = 4
    m = 1
    train_X = torch.rand((n, d), generator=g)
    train_Y = torch.randn((n, m), generator=g)
    train_Yvar = torch.zeros_like(train_Y)
    Xq = torch.rand((13, d), generator=g)
    base = EpistemicNearestNeighborsT(k=3)
    base.add(train_X, train_Y, train_Yvar)
    mvn_base = base.posterior(Xq)
    model = EpistemicNearestNeighborsBoTorchT(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar, k=3)
    posterior = model.posterior(Xq)
    mvn = posterior.distribution
    torch.testing.assert_close(mvn.mean, mvn_base.mu.squeeze(-1), atol=1e-6, rtol=0)
    torch.testing.assert_close(mvn.variance.sqrt(), mvn_base.se.squeeze(-1), atol=1e-6, rtol=0)


def test_ennt_weighter_botorch_matches_weighter_posterior_cpu():
    g = torch.Generator(device="cpu").manual_seed(13)
    n = 40
    d = 5
    m = 1
    train_X = torch.rand((n, d), generator=g)
    train_Y = torch.randn((n, m), generator=g)
    Xq = torch.rand((11, d), generator=g)
    base = ENNWeighterT(weighting="sobol_over_sigma", k=3)
    base.add(train_X, train_Y)
    mvn_base = base.posterior(Xq)
    model = EpistemicNearestNeighborsWeighterBoTorchT(train_X=train_X, train_Y=train_Y, weighting="sobol_over_sigma", k=3)
    posterior = model.posterior(Xq)
    mvn = posterior.distribution
    torch.testing.assert_close(mvn.mean, mvn_base.mu.squeeze(-1), atol=1e-5, rtol=0)
    torch.testing.assert_close(mvn.variance.sqrt(), mvn_base.se.squeeze(-1), atol=1e-5, rtol=0)
