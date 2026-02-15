import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd_simple import UHDSimple


def _make_uhd(sigma_0=1.0):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    uhd = UHDSimple(gp, sigma_0=sigma_0, dim=dim)
    return module, gp, uhd


def test_ask_perturbs_module():
    module, _, uhd = _make_uhd()
    orig = module.weight.data.clone()
    uhd.ask()
    assert not torch.equal(module.weight.data, orig)


def test_tell_first_always_accepts():
    module, _, uhd = _make_uhd()
    orig = module.weight.data.clone()

    uhd.ask()
    perturbed = module.weight.data.clone()
    uhd.tell(1.0, 0.0)

    assert torch.equal(module.weight.data, perturbed)
    assert not torch.equal(module.weight.data, orig)


def test_tell_worse_reverts():
    module, _, uhd = _make_uhd()

    uhd.ask()
    uhd.tell(10.0, 0.0)
    accepted_weight = module.weight.data.clone()

    uhd.ask()
    uhd.tell(5.0, 0.0)  # worse than y_best, should revert

    assert torch.allclose(module.weight.data, accepted_weight)


def test_tell_improvement_keeps_new_params():
    module, _, uhd = _make_uhd()

    uhd.ask()
    uhd.tell(1.0, 0.0)

    uhd.ask()
    better_weight = module.weight.data.clone()
    uhd.tell(2.0, 0.0)

    assert torch.equal(module.weight.data, better_weight)


def test_y_best_tracks_maximum():
    _, _, uhd = _make_uhd()

    uhd.ask()
    uhd.tell(1.0, 0.0)
    assert uhd.y_best == 1.0

    uhd.ask()
    uhd.tell(3.0, 0.0)
    assert uhd.y_best == 3.0

    uhd.ask()
    uhd.tell(2.0, 0.0)  # worse, y_best unchanged
    assert uhd.y_best == 3.0


def test_y_best_starts_none():
    _, _, uhd = _make_uhd()
    assert uhd.y_best is None


def test_mu_se_avg_reflect_last_tell():
    _, _, uhd = _make_uhd()

    uhd.ask()
    uhd.tell(1.5, 0.3)
    assert uhd.mu_avg == 1.5
    assert uhd.se_avg == 0.3


def test_multiple_steps():
    module, _, uhd = _make_uhd(sigma_0=0.5)

    for _ in range(10):
        uhd.ask()
        mu = float(torch.randn(1).item())
        uhd.tell(mu, 0.0)

    assert torch.isfinite(module.weight.data).all()
