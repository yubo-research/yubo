import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd_hoeffding import UHDHoeffding


def _make_uhd(sigma_0=1.0, alpha=0.1):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    uhd = UHDHoeffding(gp, sigma_0=sigma_0, dim=dim, alpha=alpha)
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


def test_y_best_tracks_accepted():
    """y_best should reflect the best accepted average mu."""
    _, _, uhd = _make_uhd(alpha=2.0)

    uhd.ask()
    uhd.tell(1.0, 0.0)
    assert abs(uhd.y_best - 1.0) < 1e-10

    # Accumulator carries: weighted avg of (1.0, 3.0) = 2.0 > 1.0 → accepted
    uhd.ask()
    uhd.tell(3.0, 0.0)
    assert abs(uhd.y_best - 2.0) < 1e-10


def test_multiple_steps():
    module, _, uhd = _make_uhd(sigma_0=0.5)

    for i in range(10):
        uhd.ask()
        mu = float(torch.randn(1).item())
        uhd.tell(mu, 0.0)

    assert torch.isfinite(module.weight.data).all()
