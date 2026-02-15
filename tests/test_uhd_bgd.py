import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd_bgd import UHDBGD


def _make_bgd(sigma_0=1.0, lr=0.01, alpha=0.1):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bgd = UHDBGD(gp, sigma_0=sigma_0, dim=dim, lr=lr, alpha=alpha)
    return module, gp, bgd


def test_ask_perturbs_module():
    module, _, bgd = _make_bgd()
    orig = module.weight.data.clone()
    bgd.ask()
    assert not torch.equal(module.weight.data, orig)


def test_first_tell_no_step():
    """First tell has no previous mu, so no gradient step."""
    module, _, bgd = _make_bgd()
    orig = module.weight.data.clone()

    bgd.ask()
    bgd.tell(5.0, 0.1)

    assert torch.allclose(module.weight.data, orig)


def test_positive_surprise_moves_params():
    """When delta exceeds EWMA, signal > 0, params should change."""
    module, _, bgd = _make_bgd(lr=1.0, alpha=0.0)

    bgd.ask()
    bgd.tell(1.0, 0.0)  # first: sets mu_prev=1.0, no step
    orig = module.weight.data.clone()

    bgd.ask()
    bgd.tell(3.0, 0.0)  # delta=2.0, ewma=0, signal=2.0

    assert not torch.allclose(module.weight.data, orig)


def test_y_best_tracks_maximum():
    _, _, bgd = _make_bgd()

    bgd.ask()
    bgd.tell(3.0, 0.0)
    assert abs(bgd.y_best - 3.0) < 1e-10

    bgd.ask()
    bgd.tell(1.0, 0.0)
    assert abs(bgd.y_best - 3.0) < 1e-10

    bgd.ask()
    bgd.tell(5.0, 0.0)
    assert abs(bgd.y_best - 5.0) < 1e-10


def test_multiple_steps_finite():
    module, _, bgd = _make_bgd(sigma_0=0.5, lr=0.001)

    for _ in range(20):
        bgd.ask()
        mu = float(torch.randn(1).item())
        bgd.tell(mu, 0.1)

    assert torch.isfinite(module.weight.data).all()
