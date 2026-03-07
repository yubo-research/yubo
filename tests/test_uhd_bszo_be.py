import torch
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder
from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd_bszo_be import UHDBSZOBE


def _make_bszo_be(
    *,
    k=2,
    lr=0.01,
    warmup=6,
    num_candidates=3,
    fit_interval=2,
    epsilon=0.01,
):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bounds = torch.zeros(2, 3)
    bounds[1] = 1.0
    embedder = BehavioralEmbedder(bounds, num_probes=4, seed=0)
    uhd = UHDBSZOBE(
        gp,
        dim,
        module,
        embedder,
        epsilon=epsilon,
        k=k,
        lr=lr,
        num_candidates=num_candidates,
        warmup=warmup,
        fit_interval=fit_interval,
    )
    return module, uhd


def _run_step(uhd, mu_fn=None):
    for _ in range(uhd.k + 1):
        uhd.ask()
        mu = mu_fn() if mu_fn else float(torch.randn(1).item())
        uhd.tell(mu, 0.0)


def test_ask_perturbs_module():
    module, uhd = _make_bszo_be()
    orig = module.weight.data.clone()
    uhd.ask()
    uhd.tell(1.0, 0.0)
    uhd.ask()
    assert not torch.equal(module.weight.data, orig)


def test_phase_transitions():
    _, uhd = _make_bszo_be(k=2)
    assert uhd.phase == 0
    uhd.ask()
    uhd.tell(1.0, 0.0)
    assert uhd.phase == 1
    uhd.ask()
    uhd.tell(1.1, 0.0)
    assert uhd.phase == 2
    uhd.ask()
    uhd.tell(0.9, 0.0)
    assert uhd.phase == 0


def test_gradient_step_changes_params():
    module, uhd = _make_bszo_be()
    orig = module.weight.data.clone()
    _run_step(uhd)
    assert not torch.equal(module.weight.data, orig)


def test_y_best_tracks_maximum():
    _, uhd = _make_bszo_be()
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
    _, uhd = _make_bszo_be(epsilon=0.05)
    assert uhd.y_best is None
    assert uhd.sigma == 0.05
    assert uhd.k == 2
    assert isinstance(uhd.eval_seed, int)
    uhd.ask()
    uhd.tell(1.5, 0.3)
    assert uhd.mu_avg == 1.5
    assert uhd.se_avg == 0.3


def test_runs_many_steps():
    torch.manual_seed(0)
    module, uhd = _make_bszo_be(warmup=6, num_candidates=3, fit_interval=2)
    for _ in range(30):
        _run_step(uhd)
    assert torch.isfinite(module.weight.data).all()


def test_seed_selection_activates_after_warmup():
    warmup = 6
    _, uhd = _make_bszo_be(warmup=warmup, num_candidates=3, k=2)

    eval_seeds_during_warmup = []
    for _ in range(warmup):
        _run_step(uhd, mu_fn=lambda: float(torch.randn(1).item()))
        eval_seeds_during_warmup.append(uhd.eval_seed)

    assert eval_seeds_during_warmup == list(range(1, warmup + 1))

    _run_step(uhd, mu_fn=lambda: float(torch.randn(1).item()))
    seed_after = uhd.eval_seed
    assert seed_after == warmup + 1


def test_enn_data_accumulates():
    _, uhd = _make_bszo_be(warmup=100, k=2)

    for _ in range(5):
        _run_step(uhd, mu_fn=lambda: float(torch.randn(1).item()))

    assert len(uhd._zs) == 10
    assert len(uhd._ys) == 10
