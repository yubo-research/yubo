import torch
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder
from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.uhd_simple_be import UHDSimpleBE, _SimpleENN, _SimplePosterior


def _make_uhd_be(sigma_0=1.0, warmup=5, num_candidates=3, fit_interval=1):
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bounds = torch.zeros(2, 3)
    bounds[1] = 1.0
    embedder = BehavioralEmbedder(bounds, num_probes=4, seed=0)
    uhd = UHDSimpleBE(
        gp,
        sigma_0=sigma_0,
        dim=dim,
        module=module,
        embedder=embedder,
        num_candidates=num_candidates,
        warmup=warmup,
        fit_interval=fit_interval,
    )
    return module, uhd


def test_ask_perturbs_module():
    module, uhd = _make_uhd_be()
    orig = module.weight.data.clone()
    uhd.ask()
    assert not torch.equal(module.weight.data, orig)


def test_tell_first_always_accepts():
    module, uhd = _make_uhd_be()
    uhd.ask()
    perturbed = module.weight.data.clone()
    uhd.tell(1.0, 0.0)
    assert torch.equal(module.weight.data, perturbed)


def test_tell_worse_reverts():
    module, uhd = _make_uhd_be()
    uhd.ask()
    uhd.tell(10.0, 0.0)
    accepted = module.weight.data.clone()

    uhd.ask()
    uhd.tell(5.0, 0.0)
    assert torch.allclose(module.weight.data, accepted)


def test_y_best_tracks_maximum():
    _, uhd = _make_uhd_be()
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
    _, uhd = _make_uhd_be()
    assert uhd.y_best is None
    uhd.ask()
    uhd.tell(1.5, 0.3)
    assert uhd.mu_avg == 1.5
    assert uhd.se_avg == 0.3
    assert isinstance(uhd.eval_seed, int)
    assert isinstance(uhd.sigma, float)


def test_warmup_seeds_sequential():
    _, uhd = _make_uhd_be(warmup=100)
    seeds = []
    for _ in range(5):
        uhd.ask()
        seeds.append(uhd.eval_seed)
        uhd.tell(float(torch.randn(1).item()), 0.0)
    assert seeds == [0, 1, 2, 3, 4]


def test_after_warmup_seeds_advance_by_num_candidates():
    num_candidates = 3
    warmup = 3
    _, uhd = _make_uhd_be(warmup=warmup, num_candidates=num_candidates)

    for i in range(warmup):
        uhd.ask()
        uhd.tell(float(i), 0.0)

    uhd.ask()
    seed_after_warmup = uhd.eval_seed
    assert seed_after_warmup >= warmup
    assert seed_after_warmup < warmup + num_candidates


def test_runs_many_steps():
    module, uhd = _make_uhd_be(warmup=5, num_candidates=3, fit_interval=2)
    for _ in range(20):
        uhd.ask()
        mu = float(torch.randn(1).item())
        uhd.tell(mu, 0.0)
    assert torch.isfinite(module.weight.data).all()


def test_simple_enn_posterior():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3))
    y = x[:, 0] + 0.1 * rng.standard_normal(20)
    enn = _SimpleENN(x=x, y=y, k=5)

    x_cand = rng.standard_normal((4, 3))
    post = enn.posterior(x_cand)
    assert post.mu.shape == (4,)
    assert post.se.shape == (4,)
    assert np.all(post.se >= 0)


def test_simple_posterior():
    import numpy as np

    post = _SimplePosterior(mu=np.array([1.0, 2.0]), se=np.array([0.1, 0.2]))
    assert post.mu[0] == 1.0
    assert post.se[1] == 0.2
