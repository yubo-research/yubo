import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd_hoeffding import UHDHoeffding
from tests.uhd_ask_tell_helpers import (
    assert_ask_perturbs_module,
    assert_tell_first_always_accepts,
    assert_tell_improvement_keeps_new_params,
    assert_tell_worse_reverts,
)


def _make_uhd(sigma_0=1.0, alpha=0.1):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    uhd = UHDHoeffding(gp, sigma_0=sigma_0, dim=dim, alpha=alpha)
    return module, gp, uhd


def test_ask_perturbs_module():
    assert_ask_perturbs_module(_make_uhd)


def test_tell_first_always_accepts():
    assert_tell_first_always_accepts(_make_uhd)


def test_tell_worse_reverts():
    assert_tell_worse_reverts(_make_uhd)


def test_tell_improvement_keeps_new_params():
    assert_tell_improvement_keeps_new_params(_make_uhd)


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
