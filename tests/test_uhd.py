import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd import UHD


def _make_uhd(sigma_0=1.0):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    uhd = UHD(gp, sigma_0=sigma_0, dim=dim)
    return module, gp, uhd


def test_ask_returns_incrementing_seeds():
    _, _, uhd = _make_uhd()
    s0 = uhd.ask()
    uhd.tell(s0, 1.0)
    s1 = uhd.ask()
    uhd.tell(s1, 2.0)
    assert s0 == 0
    assert s1 == 1


def test_tell_improvement_accepts():
    module, _, uhd = _make_uhd()
    orig = module.weight.data.clone()

    seed = uhd.ask()
    perturbed = module.weight.data.clone()
    uhd.tell(seed, 1.0)  # First call, always accepted

    # Params should stay at the perturbed values
    assert torch.equal(module.weight.data, perturbed)
    assert not torch.equal(module.weight.data, orig)


def test_tell_no_improvement_reverts():
    module, _, uhd = _make_uhd()

    # First step: accepted (no prior y_max)
    seed0 = uhd.ask()
    uhd.tell(seed0, 10.0)
    accepted_weight = module.weight.data.clone()

    # Second step: worse score, should revert
    seed1 = uhd.ask()
    uhd.tell(seed1, 5.0)

    assert torch.allclose(module.weight.data, accepted_weight)


def test_tell_improvement_keeps_new_params():
    module, _, uhd = _make_uhd()

    seed0 = uhd.ask()
    uhd.tell(seed0, 1.0)

    seed1 = uhd.ask()
    better_weight = module.weight.data.clone()
    uhd.tell(seed1, 2.0)  # Improvement

    assert torch.equal(module.weight.data, better_weight)


def test_multiple_steps():
    module, _, uhd = _make_uhd(sigma_0=0.5)

    for i in range(10):
        seed = uhd.ask()
        y = float(torch.randn(1).item())
        uhd.tell(seed, y)

    # Just verify it doesn't crash and module has valid params
    assert torch.isfinite(module.weight.data).all()
