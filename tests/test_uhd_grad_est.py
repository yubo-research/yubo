import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.lr_scheduler import ConstantLR
from optimizer.uhd_mezo import UHDMeZO


def _make(lr=0.001):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    scheduler = ConstantLR(lr)
    uhd = UHDMeZO(gp, dim=dim, lr_scheduler=scheduler)
    return module, gp, uhd


def test_positive_phase_perturbs():
    module, _, uhd = _make()
    orig = module.weight.data.clone()
    uhd.ask()
    assert not torch.equal(module.weight.data, orig)


def test_positive_tell_unperturbs():
    module, _, uhd = _make()
    orig = module.weight.data.clone()
    uhd.ask()
    uhd.tell(1.0, 0.0)
    # After positive tell, params should be back to original.
    assert torch.allclose(module.weight.data, orig)


def test_negative_phase_perturbs_opposite():
    module, _, uhd = _make()
    orig = module.weight.data.clone()

    uhd.ask()
    pos_delta = module.weight.data - orig
    uhd.tell(1.0, 0.0)

    uhd.ask()
    neg_delta = module.weight.data - orig
    # Negative perturbation should be the mirror of positive (within float round-trip).
    assert torch.allclose(neg_delta, -pos_delta, atol=1e-6)


def test_full_step_moves_params():
    module, _, uhd = _make(lr=0.1)
    orig = module.weight.data.clone()

    # Positive phase
    uhd.ask()
    uhd.tell(2.0, 0.0)  # mu_plus = 2

    # Negative phase
    uhd.ask()
    uhd.tell(1.0, 0.0)  # mu_minus = 1, advantage > 0

    # Params should have moved from original.
    assert not torch.equal(module.weight.data, orig)


def test_zero_advantage_no_movement():
    module, _, uhd = _make(lr=0.1)
    orig = module.weight.data.clone()

    uhd.ask()
    uhd.tell(5.0, 0.0)

    uhd.ask()
    uhd.tell(5.0, 0.0)  # same mu → advantage = 0

    # step_scale = 0, so perturb(seed, 0) leaves params unchanged.
    assert torch.allclose(module.weight.data, orig)


def test_positive_advantage_moves_toward_positive():
    """When mu_plus > mu_minus, step should be in the positive-epsilon direction."""
    module, _, uhd = _make(lr=1.0)
    orig = module.weight.data.clone()

    uhd.ask()
    pos_delta = (module.weight.data - orig).clone()
    uhd.tell(10.0, 0.0)

    uhd.ask()
    uhd.tell(0.0, 0.0)  # mu_minus = 0, advantage = 10

    step = module.weight.data - orig
    # Step should be positively correlated with pos_delta.
    assert float((step * pos_delta).sum()) > 0


def test_y_best_tracks_maximum():
    _, _, uhd = _make()
    assert uhd.y_best is None

    uhd.ask()
    uhd.tell(3.0, 0.0)
    assert uhd.y_best == 3.0

    uhd.ask()
    uhd.tell(1.0, 0.0)
    assert uhd.y_best == 3.0

    uhd.ask()
    uhd.tell(5.0, 0.0)
    assert uhd.y_best == 5.0


def test_mu_se_reflect_last_tell():
    _, _, uhd = _make()

    uhd.ask()
    uhd.tell(1.5, 0.3)
    assert uhd.mu_avg == 1.5
    assert uhd.se_avg == 0.3

    uhd.ask()
    uhd.tell(2.5, 0.1)
    assert uhd.mu_avg == 2.5
    assert uhd.se_avg == 0.1


def test_multiple_gradient_steps():
    module, _, uhd = _make(lr=0.01)

    for _ in range(10):
        # Two ask/tell cycles per gradient step.
        uhd.ask()
        uhd.tell(float(torch.randn(1).item()), 0.0)
        uhd.ask()
        uhd.tell(float(torch.randn(1).item()), 0.0)

    assert torch.isfinite(module.weight.data).all()


def test_seed_increments_per_gradient_step():
    """Seed should increment once per gradient step (pair of ask/tells)."""
    _, _, uhd = _make()

    # After first pair, seed should be 1.
    uhd.ask()
    uhd.tell(1.0, 0.0)
    uhd.ask()
    uhd.tell(0.0, 0.0)

    # After second pair, seed should be 2.
    uhd.ask()
    uhd.tell(2.0, 0.0)
    uhd.ask()
    uhd.tell(1.0, 0.0)

    assert uhd._seed == 2
