import torch
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder
from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd_simple_be import UHDMeZOBE


def _make_mezo_be(sigma=0.1, lr=0.01, warmup=5, num_candidates=3, fit_interval=1):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bounds = torch.zeros(2, 3)
    bounds[1] = 1.0
    embedder = BehavioralEmbedder(bounds, num_probes=4, seed=0)
    uhd = UHDMeZOBE(
        gp,
        dim,
        module,
        embedder,
        sigma=sigma,
        lr=lr,
        num_candidates=num_candidates,
        warmup=warmup,
        fit_interval=fit_interval,
    )
    return module, uhd


def test_ask_perturbs_module():
    module, uhd = _make_mezo_be()
    orig = module.weight.data.clone()
    uhd.ask()
    assert not torch.equal(module.weight.data, orig)


def test_positive_then_negative_phases():
    _, uhd = _make_mezo_be()
    assert uhd.positive_phase
    uhd.ask()
    assert uhd.positive_phase
    uhd.tell(1.0, 0.0)
    assert not uhd.positive_phase
    uhd.ask()
    assert not uhd.positive_phase
    uhd.tell(0.5, 0.0)
    assert uhd.positive_phase


def test_gradient_step_changes_params():
    module, uhd = _make_mezo_be()
    orig = module.weight.data.clone()
    uhd.ask()
    uhd.tell(2.0, 0.0)
    uhd.ask()
    uhd.tell(1.0, 0.0)
    assert not torch.equal(module.weight.data, orig)


def test_y_best_tracks_maximum():
    _, uhd = _make_mezo_be()
    assert uhd.y_best is None
    uhd.ask()
    uhd.tell(1.0, 0.0)
    assert uhd.y_best == 1.0
    uhd.ask()
    uhd.tell(3.0, 0.0)
    assert uhd.y_best == 3.0
    uhd.ask()
    uhd.tell(2.0, 0.0)
    assert uhd.y_best == 3.0


def test_properties():
    _, uhd = _make_mezo_be(sigma=0.05)
    assert uhd.y_best is None
    assert uhd.sigma == 0.05
    assert isinstance(uhd.eval_seed, int)
    uhd.ask()
    uhd.tell(1.5, 0.3)
    assert uhd.mu_avg == 1.5
    assert uhd.se_avg == 0.3


def test_runs_many_steps():
    module, uhd = _make_mezo_be(warmup=6, num_candidates=3, fit_interval=2)
    for _ in range(40):
        uhd.ask()
        mu = float(torch.randn(1).item())
        uhd.tell(mu, 0.0)
    assert torch.isfinite(module.weight.data).all()


def test_seed_selection_activates_after_warmup():
    warmup = 6
    _, uhd = _make_mezo_be(warmup=warmup, num_candidates=3)
    seeds_during_warmup = []
    for _ in range(warmup):
        uhd.ask()
        seeds_during_warmup.append(uhd.eval_seed)
        uhd.tell(float(torch.randn(1).item()), 0.0)

    assert seeds_during_warmup == [0, 0, 1, 1, 2, 2]

    uhd.ask()
    seed_after = uhd.eval_seed
    assert seed_after >= 3
